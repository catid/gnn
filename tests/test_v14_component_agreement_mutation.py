from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import transition_model_for_growth
from apsgnn.model import APSGNNModel


def test_v14_component_gate_requires_final_stage() -> None:
    config = load_config("configs/v14_utility_querygrad_agree_longplus.yaml")
    torch.manual_seed(31)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=16,
        next_active_compute_nodes=24,
        transition_mode="split",
        split_mode="mutate",
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=[(1, 17), (2, 18)],
        transition_stats={
            "selected_parent_scores": {1: 2.0, 2: 1.8},
            "unselected_parent_scores": {5: 0.8},
            "parent_components": {
                1: {"visit_z": 1.0, "query_grad_z": 1.0, "score": 2.0},
                2: {"visit_z": 1.0, "query_grad_z": 1.0, "score": 1.8},
            },
        },
        next_stage_index=5,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
        mutation_min_visit_z=config.growth.mutation_min_visit_z,
        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
    )

    assert split_stats["mutated_parent_ids"] == []


def test_v14_component_gate_requires_visit_and_query_grad_agreement() -> None:
    config = load_config("configs/v14_utility_querygrad_agree_longplus.yaml")
    torch.manual_seed(37)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=24,
        next_active_compute_nodes=32,
        transition_mode="split",
        split_mode="mutate",
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=[(1, 25), (2, 26), (3, 27), (4, 28)],
        transition_stats={
            "selected_parent_scores": {1: 2.0, 2: 1.9, 3: 1.4, 4: 1.2},
            "unselected_parent_scores": {5: 1.1, 6: 0.7},
            "parent_components": {
                1: {"visit_z": 0.6, "query_grad_z": 0.7, "score": 2.0},
                2: {"visit_z": 0.1, "query_grad_z": 0.8, "score": 1.9},
                3: {"visit_z": 0.7, "query_grad_z": 0.2, "score": 1.4},
                4: {"visit_z": 0.7, "query_grad_z": 0.8, "score": 1.2},
            },
        },
        next_stage_index=6,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
        mutation_min_visit_z=config.growth.mutation_min_visit_z,
        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
    )

    assert split_stats["mutation_reference_kind"] == "best_unselected"
    assert split_stats["mutation_reference_score"] == 1.1
    assert split_stats["mutated_parent_ids"] == [1]
    assert split_stats["mutation_min_visit_z"] == config.growth.mutation_min_visit_z
    assert split_stats["mutation_min_query_grad_z"] == config.growth.mutation_min_query_grad_z
