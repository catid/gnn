from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import GrowthSchedule, transition_model_for_growth
from apsgnn.model import APSGNNModel
from apsgnn.tasks import GrowthMemoryRoutingTask


def _flatten_node_cell(model: APSGNNModel, node_id: int) -> torch.Tensor:
    parts = [parameter.detach().reshape(-1).cpu() for parameter in model.node_cells[node_id - 1].parameters()]
    return torch.cat(parts)


def test_staged_static_activation_schedule_correctness() -> None:
    config = load_config("configs/v7_staged_static_hard.yaml")
    schedule = GrowthSchedule.from_config(config)

    assert [stage.active_compute_nodes for stage in schedule.stages] == [4, 8, 16, 32]
    assert [stage.start_step for stage in schedule.stages] == [1, 251, 601, 1101]
    assert [stage.end_step for stage in schedule.stages] == [250, 600, 1100, 3000]


def test_staged_static_projected_targets_match_growth() -> None:
    staged = load_config("configs/v7_staged_static_hard.yaml")
    growth = load_config("configs/v7_growth_clone_hard.yaml")
    staged_task = GrowthMemoryRoutingTask(staged)
    growth_task = GrowthMemoryRoutingTask(growth)

    batch_staged = staged_task.generate(batch_size=2, seed=1234, active_compute_nodes=8, bootstrap_mode=False)
    batch_growth = growth_task.generate(batch_size=2, seed=1234, active_compute_nodes=8, bootstrap_mode=False)

    assert torch.equal(batch_staged.writer_home_nodes, batch_growth.writer_home_nodes)
    assert torch.equal(batch_staged.query_home_nodes, batch_growth.query_home_nodes)
    assert torch.equal(batch_staged.writer_start_nodes, batch_growth.writer_start_nodes)
    assert torch.equal(batch_staged.query_start_nodes, batch_growth.query_start_nodes)


def test_staged_static_does_not_use_split_inheritance() -> None:
    config = load_config("configs/v7_staged_static_hard.yaml")
    torch.manual_seed(7)
    model = APSGNNModel(config)
    node5_before = _flatten_node_cell(model, 5).clone()

    stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=8,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=config.train.seed,
    )

    assert stats["sibling_pairs"] == []
    assert torch.equal(_flatten_node_cell(model, 5), node5_before)


def test_inactive_nodes_do_not_receive_normal_task_traffic_before_activation() -> None:
    config = load_config("configs/v7_staged_static_hard.yaml")
    task = GrowthMemoryRoutingTask(config)
    batch = task.generate(batch_size=4, seed=1234, active_compute_nodes=4, bootstrap_mode=False)

    assert int(batch.writer_start_nodes.max().item()) <= 4
    assert int(batch.query_start_nodes.max().item()) <= 4
    assert int(batch.writer_home_nodes.max().item()) <= 4
    assert int(batch.query_home_nodes.max().item()) <= 4
    assert batch.bootstrap_start_nodes is None


def test_growth_clone_and_staged_static_differ_only_in_inheritance_mechanism() -> None:
    staged = load_config("configs/v7_staged_static_hard.yaml")
    growth = load_config("configs/v7_growth_clone_hard.yaml")

    assert staged.model == growth.model
    assert staged.task == growth.task
    assert staged.train == growth.train
    assert staged.growth.stage_active_counts == growth.growth.stage_active_counts
    assert staged.growth.stage_steps == growth.growth.stage_steps
    assert staged.growth.bootstrap_steps == growth.growth.bootstrap_steps
    assert staged.growth.clock_prior_bias == growth.growth.clock_prior_bias
    assert staged.growth.bootstrap_clock_prior_bias == growth.growth.bootstrap_clock_prior_bias
    assert staged.growth.transition_mode == "activate"
    assert growth.growth.transition_mode == "split"
    assert staged.growth.split_mode == growth.growth.split_mode == "clone"


def test_growth_clone_rewrites_newly_active_nodes_but_staged_static_keeps_original_init() -> None:
    staged_config = load_config("configs/v7_staged_static_hard.yaml")
    growth_config = load_config("configs/v7_growth_clone_hard.yaml")

    torch.manual_seed(11)
    staged_model = APSGNNModel(staged_config)
    torch.manual_seed(11)
    growth_model = APSGNNModel(growth_config)

    original_parent3 = _flatten_node_cell(staged_model, 3).clone()
    original_node5 = _flatten_node_cell(staged_model, 5).clone()

    transition_model_for_growth(
        staged_model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=8,
        transition_mode=staged_config.growth.transition_mode,
        split_mode=staged_config.growth.split_mode,
        mutation_scale=staged_config.growth.split_mutation_scale,
        seed=staged_config.train.seed,
    )
    transition_model_for_growth(
        growth_model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=8,
        transition_mode=growth_config.growth.transition_mode,
        split_mode=growth_config.growth.split_mode,
        mutation_scale=growth_config.growth.split_mutation_scale,
        seed=growth_config.train.seed,
    )

    assert torch.equal(_flatten_node_cell(staged_model, 5), original_node5)
    assert torch.equal(_flatten_node_cell(growth_model, 5), original_parent3)
    assert not torch.equal(_flatten_node_cell(growth_model, 5), original_node5)
