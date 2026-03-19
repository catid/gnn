from __future__ import annotations

from itertools import pairwise

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def _tracker(num_compute_nodes: int = 32) -> tuple[CoverageTracker, GrowthStage]:
    tracker = CoverageTracker(
        num_compute_nodes=num_compute_nodes,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=3)
    tracker.start_stage(stage)
    return tracker, stage


def _wide(values: list[float], width: int = 32) -> torch.Tensor:
    tensor = torch.zeros(width)
    tensor[: len(values)] = torch.tensor(values, dtype=torch.float32)
    return tensor


def test_v21_selector_score_computation_distinguishes_all_seven_members() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        task_visit_counts=_wide([10.0, 3.0, 1.0, 1.0]),
        query_visit_counts=torch.zeros(32),
        bootstrap_visit_counts=torch.zeros(32),
        all_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        task_gradient_signal=_wide([1.0, 5.0, 1.0, 1.0]),
        query_gradient_signal=_wide([0.0, 0.0, 6.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(32),
        success_visit_counts=torch.zeros(32),
    )
    topology = build_initial_topology(load_config("configs/v21_core_full_querygrad_32_s.yaml"), 4)

    def score(visit: float, grad: float, qgrad: float) -> dict[int, dict[str, float]]:
        return tracker.selection_components(
            topology,
            utility_alpha=0.0,
            utility_visit_weight=visit,
            utility_grad_weight=grad,
            utility_query_visit_weight=0.0,
            utility_query_grad_weight=qgrad,
        )

    visit = score(1.0, 0.0, 0.0)
    visit_task = score(1.0, 1.0, 0.0)
    visit_query = score(1.0, 0.0, 1.0)
    full = score(1.0, 1.0, 1.0)
    query_only = score(0.0, 0.0, 1.0)
    visit_task_half = score(1.0, 0.5, 0.0)
    visit_query_half = score(1.0, 0.0, 0.5)

    assert visit[1]["score"] > visit[2]["score"]
    assert visit_task[2]["score"] > visit_task[3]["score"]
    assert visit_query[3]["score"] > visit_query[2]["score"]
    assert full[2]["score"] > full[4]["score"]
    assert query_only[3]["score"] > query_only[1]["score"]
    assert visit_task_half[2]["score"] > visit[2]["score"]
    assert visit_query_half[3]["score"] > visit[3]["score"]


def test_v21_bootstrap_excluded_from_selector_accounting() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.zeros(32),
        query_visit_counts=torch.zeros(32),
        bootstrap_visit_counts=_wide([9.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(32),
        query_gradient_signal=torch.zeros(32),
        bootstrap_gradient_signal=_wide([9.0, 0.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(32),
    )
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        task_visit_counts=_wide([0.0, 3.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(32),
        bootstrap_visit_counts=torch.zeros(32),
        all_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        task_gradient_signal=_wide([0.0, 2.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(32),
        bootstrap_gradient_signal=torch.zeros(32),
        success_visit_counts=torch.zeros(32),
    )
    topology = build_initial_topology(load_config("configs/v21_core_visitonly_32_s.yaml"), 4)
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


def test_v21_schedule_budget_matching_across_core_screening_family() -> None:
    names = [
        "visitonly",
        "visit_taskgrad",
        "visit_querygrad",
        "full_querygrad",
        "querygradonly",
        "visit_taskgrad_half",
        "visit_querygrad_half",
    ]
    configs = [load_config(f"configs/v21_core_{name}_32_s.yaml") for name in names]
    assert all(cfg.model == configs[0].model for cfg in configs[1:])
    assert all(cfg.task == configs[0].task for cfg in configs[1:])
    assert all(cfg.train == configs[0].train for cfg in configs[1:])
    assert configs[0].growth.stage_steps == [100, 100, 120, 120, 150, 170, 770]
    assert configs[0].train.train_steps == 1530


def test_v21_transfer_schedule_budget_matching_for_promoted_family() -> None:
    names = [
        "visitonly",
        "visit_taskgrad",
        "visit_querygrad",
        "full_querygrad",
    ]
    configs = [load_config(f"configs/v21_t1_{name}_32_m.yaml") for name in names]
    assert all(cfg.model == configs[0].model for cfg in configs[1:])
    assert all(cfg.task == configs[0].task for cfg in configs[1:])
    assert all(cfg.train == configs[0].train for cfg in configs[1:])
    assert configs[0].task.writers_per_episode == 4
    assert configs[0].growth.stage_steps == [120, 120, 150, 150, 180, 220, 1610]


def test_v21_promotion_logic_prefers_dense_eval_last_and_stability() -> None:
    rows = {
        "visitonly": {"dense": 0.20, "last": 0.18, "rolling": 0.19, "best": 0.22},
        "querygrad": {"dense": 0.18, "last": 0.15, "rolling": 0.16, "best": 0.24},
        "querygradonly": {"dense": 0.12, "last": 0.10, "rolling": 0.11, "best": 0.26},
    }

    def composite(row: dict[str, float]) -> float:
        return 0.45 * row["dense"] + 0.35 * row["last"] + 0.20 * row["rolling"]

    ranked = sorted(rows, key=lambda key: composite(rows[key]), reverse=True)
    assert ranked[0] == "visitonly"
    assert ranked[1] == "querygrad"


def test_v21_32_leaf_target_mapping_is_contiguous_and_complete() -> None:
    config = load_config("configs/v21_core_visitonly_32_s.yaml")
    topology = build_initial_topology(config, 4)
    future = [6, 8, 12, 16, 24, 32]
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
        assert intervals[-1].end == 32
        for left, right in pairwise(intervals):
            assert left.end + 1 == right.start
