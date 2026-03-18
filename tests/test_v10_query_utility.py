from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def test_v10_selection_components_include_query_tail_weights() -> None:
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
        all_visit_counts=torch.tensor([4.0, 4.0, 4.0, 4.0]),
        task_visit_counts=torch.tensor([4.0, 4.0, 4.0, 4.0]),
        query_visit_counts=torch.tensor([0.0, 0.0, 3.0, 0.0]),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([2.0, 2.0, 2.0, 2.0]),
        task_gradient_signal=torch.tensor([2.0, 2.0, 2.0, 2.0]),
        query_gradient_signal=torch.tensor([0.0, 5.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.zeros(4),
    )
    topology = build_initial_topology(load_config("configs/v10_utility_querymix_longplus.yaml"), 4)

    base = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    query = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.5,
        utility_query_grad_weight=1.0,
    )

    assert base[2]["score"] == base[1]["score"]
    assert query[2]["score"] > query[1]["score"]
    assert query[3]["score"] > base[3]["score"]
    assert query[2]["query_grad"] > 0.0
    assert query[3]["query_visit"] > 0.0


def test_v10_selection_components_keep_bootstrap_excluded_from_query_tail() -> None:
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=3)
    tracker.start_stage(stage)
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.zeros(4),
        query_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        bootstrap_visit_counts=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(4),
        query_gradient_signal=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.tensor([9.0, 0.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(4),
    )
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        query_visit_counts=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        query_gradient_signal=torch.tensor([0.0, 4.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.zeros(4),
    )
    topology = build_initial_topology(load_config("configs/v10_utility_querymix_longplus.yaml"), 4)
    components = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.5,
        utility_query_grad_weight=1.0,
    )

    assert components[1]["query_visit"] == 0.0
    assert components[2]["query_visit"] > 0.0
    assert components[2]["query_grad"] > 0.0


def test_v10_child_usefulness_uses_query_terms() -> None:
    config = load_config("configs/v10_utility_querymix_longplus.yaml")
    topology = build_initial_topology(config, 4)
    topology, split_stats = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="utility",
        utility_components={
            1: {"visit": 1.0, "grad": 1.0, "success": 0.0, "query_visit": 1.0, "query_grad": 1.0, "score": 2.0},
            2: {"visit": 1.0, "grad": 1.0, "success": 0.0, "query_visit": 1.0, "query_grad": 1.0, "score": 2.0},
            3: {"visit": 0.0, "grad": 0.0, "success": 0.0, "query_visit": 0.0, "query_grad": 0.0, "score": -1.0},
            4: {"visit": 0.0, "grad": 0.0, "success": 0.0, "query_visit": 0.0, "query_grad": 0.0, "score": -1.1},
        },
        utility_alpha=0.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.5,
        utility_query_grad_weight=1.0,
        seed=1234,
        future_active_counts=(8, 12, 16, 24, 32),
    )
    tracker = CoverageTracker(
        num_compute_nodes=config.model.num_compute_nodes,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=1, active_compute_nodes=6, start_step=1, end_step=20, bootstrap_steps=0)
    tracker.start_stage(stage, split_stats=split_stats, topology=topology)
    visits = torch.zeros(config.model.num_compute_nodes)
    grads = torch.zeros(config.model.num_compute_nodes)
    qvisits = torch.zeros(config.model.num_compute_nodes)
    qgrads = torch.zeros(config.model.num_compute_nodes)
    left_child, right_child = split_stats["sibling_pairs"][0]
    visits[left_child - 1] = 2.0
    visits[right_child - 1] = 2.0
    grads[left_child - 1] = 1.0
    grads[right_child - 1] = 1.0
    qvisits[right_child - 1] = 4.0
    qgrads[right_child - 1] = 5.0
    tracker.update(
        step=10,
        stage=stage,
        all_visit_counts=visits,
        task_visit_counts=visits,
        query_visit_counts=qvisits,
        bootstrap_visit_counts=torch.zeros_like(visits),
        all_gradient_signal=grads,
        task_gradient_signal=grads,
        query_gradient_signal=qgrads,
        bootstrap_gradient_signal=torch.zeros_like(grads),
        success_visit_counts=torch.zeros_like(visits),
    )
    tracker.finalize()
    finalized = tracker.to_dict()["stages"][0]["split_stats"]

    assert finalized["child_usefulness_share"][right_child] > finalized["child_usefulness_share"][left_child]
