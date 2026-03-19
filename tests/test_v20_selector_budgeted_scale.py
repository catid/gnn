from __future__ import annotations

from itertools import pairwise

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def _tracker(num_compute_nodes: int = 40) -> tuple[CoverageTracker, GrowthStage]:
    tracker = CoverageTracker(
        num_compute_nodes=num_compute_nodes,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=3)
    tracker.start_stage(stage)
    return tracker, stage


def _wide(values: list[float], width: int = 40) -> torch.Tensor:
    tensor = torch.zeros(width)
    tensor[: len(values)] = torch.tensor(values, dtype=torch.float32)
    return tensor


def test_v20_selector_family_score_computation_distinguishes_all_live_members() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        task_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        query_visit_counts=torch.zeros(40),
        bootstrap_visit_counts=torch.zeros(40),
        all_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        task_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        query_gradient_signal=_wide([0.0, 0.0, 6.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(40),
        success_visit_counts=torch.zeros(40),
    )
    topology = build_initial_topology(load_config("configs/v20_core_full_querygrad_40_s.yaml"), 4)

    visit = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    visit_task = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    visit_query = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )
    full = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )
    query_only = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=0.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )

    assert visit[1]["score"] > visit[2]["score"]
    assert visit_task[2]["score"] > visit_task[3]["score"]
    assert visit_query[3]["score"] > visit_query[2]["score"]
    assert full[2]["score"] > full[4]["score"]
    assert query_only[3]["score"] > query_only[1]["score"]


def test_v20_bootstrap_is_excluded_from_selector_accounting() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.zeros(40),
        query_visit_counts=torch.zeros(40),
        bootstrap_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(40),
        query_gradient_signal=torch.zeros(40),
        bootstrap_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(40),
    )
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        task_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(40),
        bootstrap_visit_counts=torch.zeros(40),
        all_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        task_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(40),
        bootstrap_gradient_signal=torch.zeros(40),
        success_visit_counts=torch.zeros(40),
    )
    topology = build_initial_topology(load_config("configs/v20_core_visitonly_40_s.yaml"), 4)
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


def test_v20_schedule_and_budget_match_across_40_scale_screening_family() -> None:
    names = [
        "visitonly",
        "visit_taskgrad",
        "visit_querygrad",
        "full_querygrad",
        "querygradonly",
    ]
    core = [load_config(f"configs/v20_core_{name}_40_s.yaml") for name in names]
    t1 = [load_config(f"configs/v20_transfer_t1_{name}_40_s.yaml") for name in names]

    assert all(cfg.model == core[0].model for cfg in core[1:])
    assert all(cfg.train == core[0].train for cfg in core[1:])
    assert all(cfg.task == core[0].task for cfg in core[1:])
    assert core[0].growth.stage_active_counts == [4, 6, 8, 12, 16, 24, 32, 40]
    assert core[0].growth.stage_steps == [100, 100, 150, 150, 200, 250, 350, 1700]

    assert all(cfg.model == t1[0].model for cfg in t1[1:])
    assert all(cfg.train == t1[0].train for cfg in t1[1:])
    assert all(cfg.task == t1[0].task for cfg in t1[1:])
    assert t1[0].task.writers_per_episode == 6
    assert t1[0].task.start_node_pool_size == 2
    assert t1[0].growth.stage_steps == [100, 100, 150, 150, 200, 250, 350, 1700]


def test_v20_larger_scale_40_target_mapping_is_contiguous_and_complete() -> None:
    config = load_config("configs/v20_core_visitonly_40_s.yaml")
    topology = build_initial_topology(config, 4)
    future = [6, 8, 12, 16, 24, 32, 40]
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
        assert intervals[-1].end == 40
        for left, right in pairwise(intervals):
            assert left.end + 1 == right.start


def test_v20_reproducible_selection_given_seed() -> None:
    config = load_config("configs/v20_core_full_querygrad_40_s.yaml")
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
            future_active_counts=[8, 12, 16, 24, 32, 40],
        )
        stats.append(split_stats)

    assert stats[0]["selected_parents"] == stats[1]["selected_parents"]
    assert stats[0]["selected_parent_scores"] == stats[1]["selected_parent_scores"]
