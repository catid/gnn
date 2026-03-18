from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def test_v9_selection_components_respect_grad_and_success_weights() -> None:
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=0)
    tracker.start_stage(stage)
    tracker.update(
        step=4,
        stage=stage,
        all_visit_counts=torch.tensor([4.0, 4.0, 4.0, 1.0]),
        task_visit_counts=torch.tensor([4.0, 4.0, 4.0, 1.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([0.5, 3.0, 0.5, 0.1]),
        task_gradient_signal=torch.tensor([0.5, 3.0, 0.5, 0.1]),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.tensor([0.0, 0.0, 3.0, 0.0]),
    )
    topology = build_initial_topology(load_config("configs/v9_utility_selective_long.yaml"), 4)

    full = tracker.selection_components(topology, utility_alpha=0.75, utility_grad_weight=1.0)
    no_success = tracker.selection_components(topology, utility_alpha=0.0, utility_grad_weight=1.0)
    no_grad = tracker.selection_components(topology, utility_alpha=0.75, utility_grad_weight=0.0)

    assert full[2]["score"] > no_grad[2]["score"]
    assert full[3]["score"] > no_success[3]["score"]
    assert no_grad[2]["grad"] == full[2]["grad"]
    assert no_success[3]["success"] == full[3]["success"]


def test_v9_selection_components_use_stage_tail_only() -> None:
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=0.25,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=12, bootstrap_steps=2)
    tracker.start_stage(stage)
    tracker.update(
        step=4,
        stage=stage,
        all_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
    )
    tracker.update(
        step=10,
        stage=stage,
        all_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
    )
    topology = build_initial_topology(load_config("configs/v9_utility_selective_long.yaml"), 4)
    components = tracker.selection_components(topology, utility_alpha=0.75, utility_grad_weight=1.0)

    assert components[1]["visit"] == 0.0
    assert components[2]["visit"] > 0.0
    assert components[2]["score"] > components[1]["score"]


def test_v9_selected_vs_unselected_child_usefulness_logging() -> None:
    config = load_config("configs/v9_utility_selective_long.yaml")
    topology = build_initial_topology(config, 8)
    utility_components = {
        1: {"visit": 0.1, "grad": 0.1, "success": 0.0, "score": -1.0},
        2: {"visit": 0.2, "grad": 0.2, "success": 0.0, "score": -0.5},
        3: {"visit": 1.0, "grad": 1.0, "success": 1.0, "score": 3.0},
        4: {"visit": 0.9, "grad": 0.9, "success": 0.8, "score": 2.8},
        5: {"visit": 0.3, "grad": 0.3, "success": 0.0, "score": 0.0},
        6: {"visit": 0.4, "grad": 0.4, "success": 0.0, "score": 0.1},
        7: {"visit": 0.5, "grad": 0.5, "success": 0.0, "score": 0.2},
        8: {"visit": 0.6, "grad": 0.6, "success": 0.0, "score": 0.3},
    }
    topology, split_stats = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=0.75,
        utility_grad_weight=1.0,
        seed=1234,
        future_active_counts=(16, 24, 32),
    )
    tracker = CoverageTracker(
        num_compute_nodes=config.model.num_compute_nodes,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=1, active_compute_nodes=12, start_step=1, end_step=20, bootstrap_steps=0)
    tracker.start_stage(stage, split_stats=split_stats, topology=topology)
    task_visits = torch.zeros(config.model.num_compute_nodes)
    task_grad = torch.zeros(config.model.num_compute_nodes)
    success = torch.zeros(config.model.num_compute_nodes)
    for parent_id in split_stats["selected_parents"]:
        for child_id in split_stats["parent_to_children"][parent_id]:
            task_visits[child_id - 1] = 8.0
            task_grad[child_id - 1] = 4.0
            success[child_id - 1] = 3.0
    for parent_id in split_stats["unselected_parents"]:
        task_visits[parent_id - 1] = 1.0
        task_grad[parent_id - 1] = 0.5
        success[parent_id - 1] = 0.25
    tracker.update(
        step=10,
        stage=stage,
        all_visit_counts=task_visits,
        task_visit_counts=task_visits,
        query_visit_counts=torch.zeros_like(task_visits),
        bootstrap_visit_counts=torch.zeros_like(task_visits),
        all_gradient_signal=task_grad,
        task_gradient_signal=task_grad,
        query_gradient_signal=torch.zeros_like(task_visits),
        bootstrap_gradient_signal=torch.zeros_like(task_visits),
        success_visit_counts=success,
    )
    tracker.finalize()
    stats = tracker.to_dict()["stages"][0]["split_stats"]
    expected_children = {
        child_id
        for parent_id in split_stats["selected_parents"]
        for child_id in split_stats["parent_to_children"][parent_id]
    }

    assert stats["selected_parent_child_usefulness_mean"] > stats["unselected_parent_child_usefulness_mean"]
    assert set(stats["parent_child_usefulness"]) == set(split_stats["eligible_parents"])
    assert set(stats["child_visit_share"]) == expected_children


def test_v9_transfer_h1_config_changes_task_density_only() -> None:
    base = load_config("configs/v9_utility_selective_long.yaml")
    transfer = load_config("configs/v9_transfer_h1_utility_long.yaml")

    assert base.model == transfer.model
    assert base.growth == transfer.growth
    assert base.task.start_node_pool_size == transfer.task.start_node_pool_size
    assert transfer.task.writers_per_episode == 4
    assert transfer.task.train_eval_writers == [4, 8, 12, 14]
