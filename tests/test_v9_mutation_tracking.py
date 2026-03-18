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


def _flatten_node(model: APSGNNModel, node_id: int) -> torch.Tensor:
    chunks = [parameter.detach().reshape(-1).cpu() for parameter in model.node_cells[node_id - 1].parameters()]
    return torch.cat(chunks)


def test_v9_mutation_diagnostics_record_child_shares_and_usefulness_wins() -> None:
    config = load_config("configs/v9_utility_mutate_long.yaml")
    topology = build_initial_topology(config, 4)
    topology, topology_stats = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="utility",
        utility_components={
            1: {"visit": 1.0, "grad": 1.0, "success": 1.0, "score": 3.0},
            2: {"visit": 0.9, "grad": 0.9, "success": 0.9, "score": 2.7},
            3: {"visit": 0.1, "grad": 0.1, "success": 0.1, "score": -1.0},
            4: {"visit": 0.1, "grad": 0.1, "success": 0.1, "score": -1.1},
        },
        utility_alpha=config.growth.utility_success_alpha,
        utility_grad_weight=config.growth.utility_grad_weight,
        seed=1234,
        future_active_counts=(8, 12, 16, 24, 32),
    )
    torch.manual_seed(7)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=6,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=config.train.seed,
        selective_parent_child_pairs=topology_stats["sibling_pairs"],
        transition_stats=topology_stats,
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
    success = torch.zeros(config.model.num_compute_nodes)
    mutated_children = set(split_stats["mutated_children"])
    for left_child, right_child in split_stats["sibling_pairs"]:
        visits[left_child - 1] = 1.0
        grads[left_child - 1] = 0.5
        success[left_child - 1] = 0.2
        visits[right_child - 1] = 5.0 if right_child in mutated_children else 1.0
        grads[right_child - 1] = 2.5 if right_child in mutated_children else 0.5
        success[right_child - 1] = 1.0 if right_child in mutated_children else 0.2
    tracker.update(
        step=12,
        stage=stage,
        all_visit_counts=visits,
        task_visit_counts=visits,
        query_visit_counts=torch.zeros_like(visits),
        bootstrap_visit_counts=torch.zeros_like(visits),
        all_gradient_signal=grads,
        task_gradient_signal=grads,
        query_gradient_signal=torch.zeros_like(visits),
        bootstrap_gradient_signal=torch.zeros_like(visits),
        success_visit_counts=success,
    )
    tracker.finalize()
    finalized = tracker.to_dict()["stages"][0]["split_stats"]

    assert finalized["mutated_children_with_traffic"] == len(mutated_children)
    assert finalized["mutated_child_more_usefulness_pairs"] == len(mutated_children)
    assert finalized["mutated_child_usefulness_win_rate"] == 1.0
    assert finalized["mutated_child_visit_share_mean"] > 0.5
    assert finalized["mutated_child_grad_share_mean"] > 0.5
    assert finalized["mutated_child_usefulness_share_mean"] > 0.5
    for child_id in mutated_children:
        assert finalized["child_visit_share"][child_id] > 0.5
        assert finalized["child_grad_share"][child_id] > 0.5


def test_v9_mutation_is_reproducible_given_seed() -> None:
    config = load_config("configs/v9_utility_mutate_long.yaml")
    topology = build_initial_topology(config, 4)
    topology, topology_stats = transition_topology_for_growth(
        topology,
        6,
        split_parent_policy="balanced",
        utility_components={},
        utility_alpha=config.growth.utility_success_alpha,
        utility_grad_weight=config.growth.utility_grad_weight,
        seed=1234,
        future_active_counts=(8, 12, 16, 24, 32),
    )

    torch.manual_seed(5)
    model_a = APSGNNModel(config)
    torch.manual_seed(5)
    model_b = APSGNNModel(config)

    stats_a = transition_model_for_growth(
        model_a,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=6,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=4234,
        selective_parent_child_pairs=topology_stats["sibling_pairs"],
        transition_stats=topology_stats,
    )
    stats_b = transition_model_for_growth(
        model_b,
        previous_active_compute_nodes=4,
        next_active_compute_nodes=6,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=4234,
        selective_parent_child_pairs=topology_stats["sibling_pairs"],
        transition_stats=topology_stats,
    )

    assert stats_a["mutated_children"] == stats_b["mutated_children"]
    for child_id in stats_a["mutated_children"]:
        assert torch.equal(_flatten_node(model_a, child_id), _flatten_node(model_b, child_id))
