from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, build_initial_topology


def _make_tracker() -> tuple[CoverageTracker, GrowthStage]:
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.0,
        utility_tail_fraction=1.0,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=0)
    tracker.start_stage(stage)
    return tracker, stage


def test_v17_querygrad_component_weights_can_flip_preference() -> None:
    tracker, stage = _make_tracker()
    tracker.update(
        step=4,
        stage=stage,
        all_visit_counts=torch.tensor([9.0, 2.0, 1.0, 1.0]),
        task_visit_counts=torch.tensor([9.0, 2.0, 1.0, 1.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_gradient_signal=torch.zeros(4),
        task_gradient_signal=torch.zeros(4),
        query_gradient_signal=torch.tensor([1.0, 8.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.zeros(4),
        success_visit_counts=torch.zeros(4),
    )
    topology = build_initial_topology(load_config("configs/v10_utility_querygrad_longplus.yaml"), 4)

    visit_only = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=0.0,
    )
    querygrad_only = tracker.selection_components(
        topology,
        utility_alpha=0.0,
        utility_visit_weight=0.0,
        utility_grad_weight=0.0,
        utility_query_visit_weight=0.0,
        utility_query_grad_weight=1.0,
    )

    assert visit_only[1]["score"] > visit_only[2]["score"]
    assert querygrad_only[2]["score"] > querygrad_only[1]["score"]


def test_v17_querygradonly_transfer_config_changes_task_density_only() -> None:
    base = load_config("configs/v17_utility_querygradonly_longplus.yaml")
    transfer = load_config("configs/v17_transfer_h1_utility_querygradonly_longplus.yaml")

    assert base.model == transfer.model
    assert base.growth == transfer.growth
    assert transfer.task.writers_per_episode == 4
    assert transfer.task.train_eval_writers == [4, 8, 12, 14]
    assert base.task.start_node_pool_size == transfer.task.start_node_pool_size
