from __future__ import annotations

import torch

from apsgnn.config import ExperimentConfig, load_config
from apsgnn.growth import CoverageTracker, GrowthSchedule, GrowthStage, clockwise_successor, project_home_leaves, split_model_for_growth
from apsgnn.model import APSGNNModel
from apsgnn.tasks import GrowthMemoryRoutingTask


def _v5_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.nodes_total = 9
    config.model.max_ttl = 8
    config.model.cache_capacity = 8
    config.model.use_first_hop_key_hint = False
    config.model.use_learned_first_hop_router = True
    config.model.first_hop_router_variant = "key_mlp_ce"
    config.model.first_hop_router_hidden_dim = 64
    config.model.first_hop_router_layers = 2
    config.model.first_hop_router_use_residual = False
    config.model.first_hop_router_separate_heads = True
    config.model.cache_read_variant = "learned_implicit"
    config.model.cache_read_hidden_dim = 64
    config.model.cache_read_layers = 2
    config.task.name = "memory_growth"
    config.task.writers_per_episode = 2
    config.task.query_ttl_min = 3
    config.task.query_ttl_max = 4
    config.task.max_rollout_steps = 10
    config.train.batch_size_per_gpu = 1
    config.train.train_steps = 120
    config.growth.enabled = True
    config.growth.stage_active_counts = [4, 8]
    config.growth.stage_steps = [60, 60]
    config.growth.bootstrap_steps = 10
    config.growth.clock_prior_bias = 0.5
    config.growth.delay_zero_bias = 0.25
    config.growth.bootstrap_clock_prior_bias = 6.0
    config.growth.bootstrap_delay_zero_bias = 6.0
    config.growth.split_mode = "clone"
    config.growth.gradient_norm_threshold = 1.0e-8
    return config


def test_projected_home_target_mapping_across_stages() -> None:
    leaves = torch.tensor([1, 4, 5, 8, 9, 12, 13, 16], dtype=torch.long)

    stage4 = project_home_leaves(leaves, active_compute_nodes=4, final_compute_nodes=16)
    stage8 = project_home_leaves(leaves, active_compute_nodes=8, final_compute_nodes=16)
    stage16 = project_home_leaves(leaves, active_compute_nodes=16, final_compute_nodes=16)

    assert torch.equal(stage4, torch.tensor([1, 1, 2, 2, 3, 3, 4, 4], dtype=torch.long))
    assert torch.equal(stage8, torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.long))
    assert torch.equal(stage16, leaves)


def test_clockwise_transport_prior_points_to_successor() -> None:
    current_node = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    successor = clockwise_successor(current_node, active_compute_nodes=4)
    assert torch.equal(successor, torch.tensor([2, 3, 4, 1], dtype=torch.long))


def test_split_operator_preserves_parent_weights_for_clone_split() -> None:
    config = _v5_config()
    model = APSGNNModel(config)
    with torch.no_grad():
        for child_index in range(4):
            model.node_cells[child_index].cache_ff.net[0].weight.fill_(child_index + 1)
            model.start_node_embed.weight[child_index + 1].fill_(child_index + 1)
            model.first_hop_router.writer_head.weight[child_index].fill_(child_index + 1)

    split_model_for_growth(
        model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=8,
        split_mode="clone",
        mutation_scale=0.02,
        seed=0,
    )

    assert torch.allclose(model.node_cells[0].cache_ff.net[0].weight, model.node_cells[1].cache_ff.net[0].weight)
    assert torch.allclose(model.node_cells[2].cache_ff.net[0].weight, model.node_cells[3].cache_ff.net[0].weight)
    assert torch.allclose(model.start_node_embed.weight[1], model.start_node_embed.weight[2])
    assert torch.allclose(model.first_hop_router.writer_head.weight[0], model.first_hop_router.writer_head.weight[1])


def test_mutate_split_changes_only_intended_child_and_local_subset() -> None:
    config = _v5_config()
    model = APSGNNModel(config)
    split_model_for_growth(
        model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=8,
        split_mode="mutate",
        mutation_scale=0.05,
        seed=7,
    )

    left_child = model.node_cells[0]
    right_child = model.node_cells[1]
    assert torch.allclose(left_child.cache_attn.in_proj_weight, right_child.cache_attn.in_proj_weight)
    assert not torch.allclose(left_child.cache_ff.net[0].weight, right_child.cache_ff.net[0].weight)
    assert not torch.allclose(model.start_node_embed.weight[1], model.start_node_embed.weight[2])
    assert not torch.allclose(model.first_hop_router.writer_head.weight[0], model.first_hop_router.writer_head.weight[1])


def test_stage_bootstrap_gives_traffic_to_all_active_nodes() -> None:
    config = _v5_config()
    task = GrowthMemoryRoutingTask(config)
    batch = task.generate(batch_size=1, seed=0, active_compute_nodes=4, bootstrap_mode=True)
    model = APSGNNModel(config)
    model.set_growth_context(active_compute_nodes=4, bootstrap_active=True)
    model.eval()
    output = model(batch.to(torch.device("cpu")))

    visits = output["diagnostics"]["visit_counts"][:4]
    assert torch.all(visits > 0)


def test_coverage_metrics_are_computed_correctly() -> None:
    tracker = CoverageTracker(num_compute_nodes=4, gradient_norm_threshold=1.0e-8, utility_ema_decay=0.9)
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=20, bootstrap_steps=10)
    tracker.start_stage(stage)
    tracker.update(
        step=1,
        stage=stage,
        visit_counts=torch.tensor([1.0, 0.0, 0.0, 0.0]),
        gradient_norms=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    for step in range(2, 11):
        tracker.update(
            step=step,
            stage=stage,
            visit_counts=torch.tensor([1.0, 1.0, 1.0, 1.0]),
            gradient_norms=torch.tensor([1.0, 1.0, 1.0, 1.0]),
        )
    tracker.finalize()
    stage_summary = tracker.to_dict()["stages"][0]

    assert stage_summary["time_to_full_visit"] == 2
    assert stage_summary["time_to_full_grad"] == 2
    assert stage_summary["visit_coverage_at"]["10"] == 1.0
    assert stage_summary["grad_coverage_at"]["10"] == 1.0


def test_growth_configs_differ_only_in_intended_ways() -> None:
    static_sparse = load_config("configs/v5_static_sparse.yaml")
    static_bootstrap = load_config("configs/v5_static_bootstrap.yaml")
    growth_clone = load_config("configs/v5_growth_clone.yaml")
    growth_mutate = load_config("configs/v5_growth_mutate.yaml")

    assert static_sparse.model.nodes_total == static_bootstrap.model.nodes_total == growth_clone.model.nodes_total == growth_mutate.model.nodes_total
    assert static_sparse.model.cache_read_variant == static_bootstrap.model.cache_read_variant == growth_clone.model.cache_read_variant == growth_mutate.model.cache_read_variant
    assert static_sparse.model.first_hop_router_variant == growth_clone.model.first_hop_router_variant
    assert static_sparse.growth.bootstrap_steps == 0
    assert static_bootstrap.growth.bootstrap_steps > 0
    assert growth_clone.growth.stage_active_counts == [4, 8, 16]
    assert growth_mutate.growth.stage_active_counts == [4, 8, 16]
    assert growth_clone.growth.split_mode == "clone"
    assert growth_mutate.growth.split_mode == "mutate"
    assert GrowthSchedule.from_config(growth_clone).final_active_compute_nodes == 16


def test_growth_schedule_rescales_for_short_smoke_runs() -> None:
    config = load_config("configs/v5_growth_clone.yaml")
    config.train.train_steps = 120
    schedule = GrowthSchedule.from_config(config)

    assert [stage.active_compute_nodes for stage in schedule.stages] == [4, 8, 16]
    assert [stage.end_step - stage.start_step + 1 for stage in schedule.stages] == [30, 30, 60]
