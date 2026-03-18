from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology, transition_topology_for_growth


def _tracker() -> tuple[CoverageTracker, GrowthStage]:
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=3)
    tracker.start_stage(stage)
    return tracker, stage


def test_v18_selector_score_computation_distinguishes_v_q_g() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=torch.tensor([8.0, 1.0, 1.0, 1.0]),
        task_visit_counts=torch.tensor([8.0, 1.0, 1.0, 1.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        task_gradient_signal=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        query_gradient_signal=torch.tensor([0.0, 6.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.zeros(4),
    )
    topology = build_initial_topology(load_config("configs/v18_core_querygrad_long.yaml"), 4)

    visit_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    querygrad_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=1.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )
    querygradonly_scores = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=0.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )

    assert visit_scores[1]["score"] > visit_scores[2]["score"]
    assert querygradonly_scores[2]["score"] > querygradonly_scores[1]["score"]
    assert querygrad_scores[2]["score"] > querygradonly_scores[1]["score"]


def test_v18_bootstrap_excluded_from_selector_accounting() -> None:
    tracker, stage = _tracker()
    tracker.update(
        step=2,
        stage=stage,
        all_visit_counts=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        task_visit_counts=torch.zeros(4),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(4),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.tensor([10.0, 0.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(4),
    )
    tracker.update(
        step=6,
        stage=stage,
        all_visit_counts=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([0.0, 2.0, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.zeros(4),
    )
    topology = build_initial_topology(load_config("configs/v18_core_visitonly_long.yaml"), 4)
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


def test_v18_transfer_regimes_match_with_intended_differences_only() -> None:
    t1_v = load_config("configs/v18_transfer_t1_visitonly_long.yaml")
    t1_q = load_config("configs/v18_transfer_t1_querygrad_long.yaml")
    t1_g = load_config("configs/v18_transfer_t1_querygradonly_long.yaml")
    t2_q = load_config("configs/v18_transfer_t2a_querygrad_long.yaml")

    assert t1_v.model == t1_q.model == t1_g.model
    assert t1_v.train == t1_q.train == t1_g.train
    assert t1_v.task == t1_q.task == t1_g.task
    assert t1_v.growth.stage_steps == [250, 250, 300, 300, 400, 600, 6400]
    assert t2_q.train == t1_q.train
    assert t2_q.growth == t1_q.growth
    assert t2_q.task.writers_per_episode == t1_q.task.writers_per_episode
    assert t2_q.task.start_node_pool_size == 1
    assert t1_q.task.start_node_pool_size == 2


def test_v18_transition_logging_and_seed_reproducibility() -> None:
    config = load_config("configs/v18_core_querygrad_long.yaml")
    topology = build_initial_topology(config, 4)
    utility_components = {
        1: {"visit": 1.0, "grad": 1.0, "success": 0.0, "query_visit": 0.0, "query_grad": 1.0, "score": 3.0},
        2: {"visit": 0.8, "grad": 0.9, "success": 0.0, "query_visit": 0.0, "query_grad": 0.7, "score": 2.4},
        3: {"visit": 0.1, "grad": 0.1, "success": 0.0, "query_visit": 0.0, "query_grad": 0.1, "score": -0.1},
        4: {"visit": 0.0, "grad": 0.0, "success": 0.0, "query_visit": 0.0, "query_grad": 0.0, "score": -0.2},
    }
    _, stats_a = transition_topology_for_growth(
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
        future_active_counts=(8, 12, 16, 24, 32),
    )
    _, stats_b = transition_topology_for_growth(
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
        future_active_counts=(8, 12, 16, 24, 32),
    )

    assert stats_a["selected_parents"] == stats_b["selected_parents"]
    assert stats_a["selected_parent_scores"] == stats_b["selected_parent_scores"]
    assert stats_a["unselected_parent_scores"] == stats_b["unselected_parent_scores"]
    assert stats_a["utility_visit_weight"] == 1.0
    assert stats_a["utility_query_grad_weight"] == 1.0
