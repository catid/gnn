from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import (
    CoverageTracker,
    GrowthStage,
    build_initial_topology,
    transition_model_for_growth,
    transition_topology_for_growth,
)
from apsgnn.model import APSGNNModel
from apsgnn.tasks import GrowthMemoryRoutingTask


def _flatten_node_cell(model: APSGNNModel, node_id: int) -> torch.Tensor:
    parts = [parameter.detach().reshape(-1).cpu() for parameter in model.node_cells[node_id - 1].parameters()]
    return torch.cat(parts)


def test_irregular_projected_target_mapping_under_selective_schedule() -> None:
    config = load_config("configs/v8_clone_selective_hard.yaml")
    topology = build_initial_topology(config, 4)
    topology, _ = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="balanced",
        utility_components={},
        utility_alpha=0.75,
        seed=1234,
    )
    leaves = torch.tensor([1, 4, 5, 8, 9, 12, 13, 16, 17, 24, 25, 32], dtype=torch.long)
    projected = topology.project_home_leaves(leaves)
    assert torch.equal(projected, torch.tensor([1, 1, 5, 5, 2, 2, 6, 6, 3, 3, 4, 4], dtype=torch.long))


def test_staged_static_activate_mode_does_not_use_inheritance() -> None:
    staged_config = load_config("configs/v8_staged_static_selective_hard.yaml")
    clone_config = load_config("configs/v8_clone_selective_hard.yaml")

    torch.manual_seed(11)
    staged_model = APSGNNModel(staged_config)
    torch.manual_seed(11)
    clone_model = APSGNNModel(clone_config)

    topology = build_initial_topology(staged_config, 4)
    topology, split_stats = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="balanced",
        utility_components={},
        utility_alpha=0.75,
        seed=1234,
    )
    parent_child_pairs = split_stats["sibling_pairs"]
    child_id = parent_child_pairs[0][1]
    parent_id = parent_child_pairs[0][0]

    staged_child_before = _flatten_node_cell(staged_model, child_id).clone()
    parent_before = _flatten_node_cell(staged_model, parent_id).clone()

    transition_model_for_growth(
        staged_model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=6,
        transition_mode=staged_config.growth.transition_mode,
        split_mode=staged_config.growth.split_mode,
        mutation_scale=staged_config.growth.split_mutation_scale,
        seed=staged_config.train.seed,
        selective_parent_child_pairs=parent_child_pairs,
        transition_stats=split_stats,
    )
    transition_model_for_growth(
        clone_model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=6,
        transition_mode=clone_config.growth.transition_mode,
        split_mode=clone_config.growth.split_mode,
        mutation_scale=clone_config.growth.split_mutation_scale,
        seed=clone_config.train.seed,
        selective_parent_child_pairs=parent_child_pairs,
        transition_stats=split_stats,
    )

    assert torch.equal(_flatten_node_cell(staged_model, child_id), staged_child_before)
    assert torch.equal(_flatten_node_cell(clone_model, child_id), parent_before)


def test_balanced_random_and_utility_modes_differ_only_in_parent_selection_logic() -> None:
    config = load_config("configs/v8_clone_selective_hard.yaml")
    topology = build_initial_topology(config, 8)
    utility_components = {
        1: {"visit": 0.1, "grad": 0.1, "success": 0.0, "score": -1.0},
        2: {"visit": 0.2, "grad": 0.2, "success": 0.0, "score": -0.5},
        3: {"visit": 1.0, "grad": 1.0, "success": 1.0, "score": 3.0},
        4: {"visit": 0.9, "grad": 0.9, "success": 0.9, "score": 2.8},
        5: {"visit": 0.3, "grad": 0.3, "success": 0.0, "score": 0.0},
        6: {"visit": 0.4, "grad": 0.4, "success": 0.0, "score": 0.1},
        7: {"visit": 0.5, "grad": 0.5, "success": 0.0, "score": 0.2},
        8: {"visit": 0.6, "grad": 0.6, "success": 0.0, "score": 0.3},
    }

    _, balanced_stats = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="balanced",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=1234,
    )
    _, utility_stats = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=1234,
    )

    assert balanced_stats["selected_parents"] == [1, 2, 3, 4]
    assert utility_stats["selected_parents"] == [3, 4, 7, 8]
    assert balanced_stats["child_intervals"] != {}
    assert utility_stats["child_intervals"] != {}


def test_utility_accounting_excludes_bootstrap_packets() -> None:
    tracker = CoverageTracker(num_compute_nodes=6, gradient_norm_threshold=1.0e-8, utility_ema_decay=0.9)
    stage = GrowthStage(index=0, active_compute_nodes=6, start_step=1, end_step=20, bootstrap_steps=5)
    tracker.start_stage(stage)
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(6),
        bootstrap_visit_counts=torch.tensor([2.0, 2.0, 2.0, 2.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(6),
        query_gradient_signal=torch.zeros(6),
        bootstrap_gradient_signal=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0]),
        success_visit_counts=torch.tensor([5.0, 5.0, 5.0, 5.0, 0.0, 0.0]),
    )
    tracker.update(
        step=8,
        stage=stage,
        all_visit_counts=torch.tensor([1.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([1.0, 3.0, 0.0, 0.0, 0.0, 0.0]),
        query_visit_counts=torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        bootstrap_visit_counts=torch.zeros(6),
        all_gradient_signal=torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([1.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        query_gradient_signal=torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(6),
        success_visit_counts=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
    topology = build_initial_topology(load_config("configs/v8_utility_selective_hard.yaml"), 4)
    components = tracker.selection_components(topology, utility_alpha=0.75)

    assert components[1]["visit"] > 0.0
    assert components[2]["visit"] > components[1]["visit"]
    assert components[3]["visit"] == 0.0
    assert components[1]["success"] > 0.0


def test_eligible_parent_detection_and_selected_parent_logging_are_correct() -> None:
    config = load_config("configs/v8_clone_selective_hard.yaml")
    topology = build_initial_topology(config, 4)
    topology, split_stats = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="balanced",
        utility_components={},
        utility_alpha=0.75,
        seed=1234,
    )
    eligible = topology.eligible_split_parents()

    assert split_stats["eligible_parents"] == [1, 2, 3, 4]
    assert split_stats["selected_parents"] == [1, 2]
    assert sorted(eligible) == [1, 2, 3, 4, 5, 6]
    assert split_stats["parent_to_child"][1] == 5
    assert split_stats["parent_to_child"][2] == 6


def test_inactive_nodes_do_not_receive_normal_task_traffic_before_activation_under_selective_schedule() -> None:
    config = load_config("configs/v8_staged_static_selective_hard.yaml")
    task = GrowthMemoryRoutingTask(config)
    topology = build_initial_topology(config, 4)
    topology, _ = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="balanced",
        utility_components={},
        utility_alpha=0.75,
        seed=1234,
    )
    batch = task.generate(batch_size=4, seed=1234, active_compute_nodes=6, bootstrap_mode=False, topology=topology)
    active_ids = set(topology.ring_node_ids)

    assert set(batch.writer_start_nodes.reshape(-1).tolist()).issubset(active_ids)
    assert set(batch.query_start_nodes.reshape(-1).tolist()).issubset(active_ids)
    assert set(batch.writer_home_nodes.reshape(-1).tolist()).issubset(active_ids)
    assert set(batch.query_home_nodes.reshape(-1).tolist()).issubset(active_ids)


def test_random_and_utility_selection_are_reproducible_given_seed() -> None:
    config = load_config("configs/v8_clone_selective_hard.yaml")
    topology = build_initial_topology(config, 8)
    utility_components = {
        node_id: {"visit": float(node_id), "grad": float(node_id) * 0.5, "success": 0.0, "score": float(node_id)}
        for node_id in topology.eligible_split_parents()
    }

    _, random_a = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="random",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=4234,
    )
    _, random_b = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="random",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=4234,
    )
    _, utility_a = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=1234,
    )
    _, utility_b = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=0.75,
        seed=9999,
    )

    assert random_a["selected_parents"] == random_b["selected_parents"]
    assert utility_a["selected_parents"] == utility_b["selected_parents"]


def test_random_selection_preserves_future_schedule_feasibility() -> None:
    config = load_config("configs/v8_random_selective_hard.yaml")
    stage_counts = list(config.growth.stage_active_counts)
    topology = build_initial_topology(config, stage_counts[0])
    base_seed = 5234

    for stage_index, next_active in enumerate(stage_counts[1:], start=1):
        topology, split_stats = transition_topology_for_growth(
            topology,
            next_active,
            split_parent_policy="random",
            utility_components={},
            utility_alpha=config.growth.utility_success_alpha,
            seed=base_seed + stage_index,
            future_active_counts=stage_counts[stage_index + 1 :],
        )
        assert len(split_stats["selected_parents"]) == next_active - stage_counts[stage_index - 1]
        assert split_stats["feasible_parent_subset_count"] > 0

    assert topology.active_compute_nodes == config.model.num_compute_nodes
    assert all(interval.size == 1 for interval in topology.node_intervals.values())


def test_utility_selection_respects_future_schedule_constraints() -> None:
    config = load_config("configs/v8_utility_selective_hard.yaml")
    topology = build_initial_topology(config, 4)
    for selected in ([1, 2], [1, 5], [1, 7, 5, 2]):
        topology, _, _, _ = topology.split_selected(list(selected))

    utility_components = {
        1: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        9: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        7: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        10: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        5: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        11: {"visit": 0.0, "grad": 0.0, "success": 0.0, "score": 0.0},
        8: {"visit": 2.0, "grad": 2.0, "success": 2.0, "score": 10.0},
        2: {"visit": 1.5, "grad": 1.5, "success": 1.5, "score": 9.0},
        12: {"visit": 1.0, "grad": 1.0, "success": 1.0, "score": 8.0},
        6: {"visit": 0.8, "grad": 0.8, "success": 0.8, "score": 7.0},
        3: {"visit": 0.7, "grad": 0.7, "success": 0.7, "score": 6.0},
        4: {"visit": 0.6, "grad": 0.6, "success": 0.6, "score": 5.0},
    }

    _, split_stats = transition_topology_for_growth(
        topology,
        16,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=config.growth.utility_success_alpha,
        seed=1234,
        future_active_counts=[24, 32],
    )

    assert split_stats["feasible_parent_subset_count"] == 3
    assert split_stats["selected_parents"] == [8, 6, 3, 4]
    assert 2 not in split_stats["selected_parents"]
    assert 12 not in split_stats["selected_parents"]
