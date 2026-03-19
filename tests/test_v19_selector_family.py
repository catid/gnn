from __future__ import annotations

from itertools import pairwise

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def _tracker() -> tuple[CoverageTracker, GrowthStage]:
    tracker = CoverageTracker(
        num_compute_nodes=64,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=3)
    tracker.start_stage(stage)
    return tracker, stage


def _wide(values: list[float]) -> torch.Tensor:
    tensor = torch.zeros(64)
    tensor[: len(values)] = torch.tensor(values, dtype=torch.float32)
    return tensor


def test_v19_selector_score_computation_distinguishes_v_vt_vq_vtq_q() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        task_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        query_visit_counts=torch.zeros(64),
        bootstrap_visit_counts=torch.zeros(64),
        all_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        task_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        query_gradient_signal=_wide([0.0, 0.0, 6.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(64),
        success_visit_counts=torch.zeros(64),
    )
    topology = build_initial_topology(load_config("configs/v19_core_full_querygrad_scale.yaml"), 4)

    visit_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    visit_task_grad_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    visit_query_grad_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )
    full_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )
    query_grad_only_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=0.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )

    assert visit_scores[1]["score"] > visit_scores[2]["score"]
    assert visit_task_grad_scores[2]["score"] > visit_task_grad_scores[3]["score"]
    assert visit_query_grad_scores[3]["score"] > visit_query_grad_scores[2]["score"]
    assert full_scores[2]["score"] > full_scores[4]["score"]
    assert query_grad_only_scores[3]["score"] > query_grad_only_scores[1]["score"]


def test_v19_bootstrap_excluded_from_selector_accounting() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.zeros(64),
        query_visit_counts=torch.zeros(64),
        bootstrap_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(64),
        query_gradient_signal=torch.zeros(64),
        bootstrap_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(64),
    )
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        task_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(64),
        bootstrap_visit_counts=torch.zeros(64),
        all_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        task_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(64),
        bootstrap_gradient_signal=torch.zeros(64),
        success_visit_counts=torch.zeros(64),
    )
    topology = build_initial_topology(load_config("configs/v19_core_visitonly_scale.yaml"), 4)
    scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )

    assert scores[1]["visit"] == 0.0
    assert scores[2]["visit"] > 0.0


def test_v19_transfer_regimes_match_with_intended_differences_only() -> None:
    t1_v = load_config("configs/v19_transfer_t1_visitonly_scale.yaml")
    t1_vt = load_config("configs/v19_transfer_t1_visit_taskgrad_scale.yaml")
    t1_vq = load_config("configs/v19_transfer_t1_visit_querygrad_scale.yaml")
    t1_vtq = load_config("configs/v19_transfer_t1_full_querygrad_scale.yaml")
    t1_q = load_config("configs/v19_transfer_t1_querygradonly_scale.yaml")

    assert t1_v.model == t1_vt.model == t1_vq.model == t1_vtq.model == t1_q.model
    assert t1_v.task == t1_vt.task == t1_vq.task == t1_vtq.task == t1_q.task
    assert t1_v.train == t1_vt.train == t1_vq.train == t1_vtq.train == t1_q.train
    assert t1_v.growth.stage_steps == [250, 250, 300, 300, 400, 500, 700, 900, 6400]
    assert t1_v.growth.stage_active_counts == [4, 6, 8, 12, 16, 24, 32, 48, 64]


def test_v19_larger_benchmark_target_mapping_is_contiguous_and_complete() -> None:
    config = load_config("configs/v19_core_visitonly_scale.yaml")
    topology = build_initial_topology(config, 4)
    future = [6, 8, 12, 16, 24, 32, 48, 64]
    for index, next_active in enumerate(future):
        topology, _ = transition_topology_for_growth(
            topology,
            next_active,
            split_parent_policy="balanced",
            utility_components=None,
            utility_alpha=0.0,
            utility_visit_weight=1.0,
            utility_grad_weight=0.0,
            utility_query_visit_weight=0.0,
            utility_query_grad_weight=0.0,
            seed=1234,
            future_active_counts=future[index + 1 :],
        )
        intervals = [topology.node_intervals[node_id] for node_id in topology.ring_node_ids]
        assert intervals[0].start == 1
        assert intervals[-1].end == 64
        for left, right in pairwise(intervals):
            assert left.end + 1 == right.start


def test_v19_reproducible_utility_selection_given_seed() -> None:
    config = load_config("configs/v19_core_full_querygrad_scale.yaml")
    topology = build_initial_topology(config, 4)
    utility_components = {
        1: {"score": 3.0},
        2: {"score": 2.0},
        3: {"score": 1.0},
        4: {"score": 0.0},
    }
    stats = []
    for _ in range(2):
        _, split_stats = transition_topology_for_growth(
            topology,
            6,
            split_parent_policy="utility",
            utility_components=utility_components,
            utility_alpha=0.0,
            utility_visit_weight=1.0,
            utility_grad_weight=1.0,
            utility_query_visit_weight=0.0,
            utility_query_grad_weight=1.0,
            seed=1234,
            future_active_counts=[8, 12, 16, 24, 32, 48, 64],
        )
        stats.append(split_stats)

    assert stats[0]["selected_parents"] == stats[1]["selected_parents"]
    assert stats[0]["selected_parent_scores"] == stats[1]["selected_parent_scores"]
