from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import (
    build_uniform_topology,
    transition_model_for_growth,
    transition_topology_for_growth,
)
from apsgnn.model import APSGNNModel


def _selected_top_half(selected_scores: dict[int, float]) -> list[int]:
    ordered = sorted(selected_scores, key=lambda node_id: (-selected_scores[node_id], node_id))
    return ordered[: max(1, len(ordered) // 2)]


def test_v11_conditional_mutation_mutates_top_half_on_late_stage() -> None:
    config = load_config("configs/v11_utility_querygrad_condmut_longplus.yaml")
    topology = build_uniform_topology(config.model.num_compute_nodes, 16)
    utility_components = {
        node_id: {
            "visit": float(node_id),
            "grad": float(node_id),
            "success": 0.0,
            "query_visit": 0.0,
            "query_grad": float(node_id),
            "score": float(node_id),
        }
        for node_id in topology.eligible_split_parents()
    }
    topology, topology_stats = transition_topology_for_growth(
        topology,
        24,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=config.growth.utility_success_alpha,
        utility_grad_weight=config.growth.utility_grad_weight,
        utility_query_visit_weight=config.growth.utility_query_visit_weight,
        utility_query_grad_weight=config.growth.utility_query_grad_weight,
        seed=1234,
        future_active_counts=(32,),
    )
    torch.manual_seed(7)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=16,
        next_active_compute_nodes=24,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=topology_stats["sibling_pairs"],
        transition_stats=topology_stats,
        next_stage_index=5,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
    )

    expected = sorted(_selected_top_half({int(k): float(v) for k, v in topology_stats["selected_parent_scores"].items()}))
    assert split_stats["mutated_parent_ids"] == expected
    assert len(split_stats["mutated_children"]) == len(expected)


def test_v11_conditional_mutation_skips_early_stages() -> None:
    config = load_config("configs/v11_utility_querygrad_condmut_longplus.yaml")
    topology = build_uniform_topology(config.model.num_compute_nodes, 8)
    utility_components = {
        node_id: {
            "visit": float(node_id),
            "grad": float(node_id),
            "success": 0.0,
            "query_visit": 0.0,
            "query_grad": float(node_id),
            "score": float(node_id),
        }
        for node_id in topology.eligible_split_parents()
    }
    topology, topology_stats = transition_topology_for_growth(
        topology,
        12,
        split_parent_policy="utility",
        utility_components=utility_components,
        utility_alpha=config.growth.utility_success_alpha,
        utility_grad_weight=config.growth.utility_grad_weight,
        utility_query_visit_weight=config.growth.utility_query_visit_weight,
        utility_query_grad_weight=config.growth.utility_query_grad_weight,
        seed=1234,
        future_active_counts=(16, 24, 32),
    )
    torch.manual_seed(5)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=8,
        next_active_compute_nodes=12,
        transition_mode=config.growth.transition_mode,
        split_mode=config.growth.split_mode,
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=topology_stats["sibling_pairs"],
        transition_stats=topology_stats,
        next_stage_index=3,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
    )

    assert split_stats["mutated_parent_ids"] == []
    assert split_stats["mutated_children"] == []
