from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import CoverageTracker, GrowthStage, clockwise_successor, project_home_leaves


def test_projected_home_target_mapping_across_32_leaf_stages() -> None:
    leaves = torch.tensor([1, 8, 9, 16, 17, 24, 25, 32], dtype=torch.long)

    stage4 = project_home_leaves(leaves, active_compute_nodes=4, final_compute_nodes=32)
    stage8 = project_home_leaves(leaves, active_compute_nodes=8, final_compute_nodes=32)
    stage16 = project_home_leaves(leaves, active_compute_nodes=16, final_compute_nodes=32)
    stage32 = project_home_leaves(leaves, active_compute_nodes=32, final_compute_nodes=32)

    assert torch.equal(stage4, torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.long))
    assert torch.equal(stage8, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long))
    assert torch.equal(stage16, torch.tensor([1, 4, 5, 8, 9, 12, 13, 16], dtype=torch.long))
    assert torch.equal(stage32, leaves)


def test_task_only_coverage_accounting_excludes_bootstrap_packets() -> None:
    tracker = CoverageTracker(num_compute_nodes=4, gradient_norm_threshold=1.0e-8, utility_ema_decay=0.9)
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=20, bootstrap_steps=5)
    tracker.start_stage(stage)
    tracker.update(
        step=10,
        stage=stage,
        all_visit_counts=torch.tensor([3.0, 3.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        query_visit_counts=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bootstrap_visit_counts=torch.tensor([2.0, 3.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        query_gradient_signal=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.tensor([0.0, 1.0, 0.0, 0.0]),
    )
    tracker.finalize()
    stage_summary = tracker.to_dict()["stages"][0]

    assert stage_summary["all_visit_coverage_at"]["10"] == 0.5
    assert stage_summary["task_visit_coverage_at"]["10"] == 0.25
    assert stage_summary["query_visit_coverage_at"]["10"] == 0.25
    assert stage_summary["task_visit_histogram"] == [1.0, 0.0, 0.0, 0.0]
    assert stage_summary["bootstrap_visit_histogram"] == [2.0, 3.0, 0.0, 0.0]


def test_task_only_gradient_coverage_uses_task_signal_not_bootstrap_signal() -> None:
    tracker = CoverageTracker(num_compute_nodes=4, gradient_norm_threshold=0.5, utility_ema_decay=0.9)
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=20, bootstrap_steps=5)
    tracker.start_stage(stage)
    tracker.update(
        step=10,
        stage=stage,
        all_visit_counts=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        task_visit_counts=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        query_visit_counts=torch.tensor([0.0, 1.0, 0.0, 0.0]),
        bootstrap_visit_counts=torch.tensor([0.0, 0.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([1.0, 0.1, 0.0, 0.0]),
        query_gradient_signal=torch.tensor([0.0, 0.1, 0.0, 0.0]),
        bootstrap_gradient_signal=torch.tensor([0.0, 0.9, 0.0, 0.0]),
    )
    tracker.finalize()
    stage_summary = tracker.to_dict()["stages"][0]

    assert stage_summary["all_grad_coverage_at"]["10"] == 0.5
    assert stage_summary["task_grad_coverage_at"]["10"] == 0.25
    assert stage_summary["query_grad_coverage_at"]["10"] == 0.0


def test_harder_benchmark_configs_differ_only_in_intended_task_parameters() -> None:
    moderate = load_config("configs/v6_static_moderate.yaml")
    hard = load_config("configs/v6_static_hard.yaml")

    assert moderate.model.nodes_total == hard.model.nodes_total == 33
    assert moderate.model.cache_read_variant == hard.model.cache_read_variant == "learned_implicit"
    assert moderate.model.first_hop_router_variant == hard.model.first_hop_router_variant == "key_mlp_ce"
    assert moderate.growth.stage_active_counts == hard.growth.stage_active_counts == [32]
    assert moderate.task.writers_per_episode == 6
    assert hard.task.writers_per_episode == 2
    assert moderate.task.start_node_pool_size == 8
    assert hard.task.start_node_pool_size == 2
    assert moderate.train.batch_size_per_gpu == 2
    assert hard.train.batch_size_per_gpu == 1
    assert (moderate.task.query_ttl_min, moderate.task.query_ttl_max) == (3, 5)
    assert (hard.task.query_ttl_min, hard.task.query_ttl_max) == (2, 3)


def test_growth_clone_and_mutate_configs_keep_split_semantics() -> None:
    clone = load_config("configs/v6_growth_clone_hard.yaml")
    mutate = load_config("configs/v6_growth_mutate_followup.yaml")

    assert clone.growth.stage_active_counts == [4, 8, 16, 32]
    assert mutate.growth.stage_active_counts == [4, 8, 16, 32]
    assert clone.growth.stage_steps == mutate.growth.stage_steps
    assert clone.growth.split_mode == "clone"
    assert mutate.growth.split_mode == "mutate"


def test_clockwise_successor_tracks_active_ring_after_stage_change() -> None:
    current_8 = torch.tensor([1, 4, 8], dtype=torch.long)
    current_32 = torch.tensor([1, 16, 32], dtype=torch.long)

    assert torch.equal(clockwise_successor(current_8, active_compute_nodes=8), torch.tensor([2, 5, 1]))
    assert torch.equal(clockwise_successor(current_32, active_compute_nodes=32), torch.tensor([2, 17, 1]))
