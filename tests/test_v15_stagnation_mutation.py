from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import mutation_stagnation_info, transition_model_for_growth
from apsgnn.model import APSGNNModel


def test_v15_stagnation_info_requires_flat_tail() -> None:
    stalled = mutation_stagnation_info([0.31, 0.325], window=2, delta=0.02)
    improving = mutation_stagnation_info([0.31, 0.34], window=2, delta=0.02)

    assert stalled["stagnated"] is True
    assert stalled["range"] == 0.015000000000000013
    assert improving["stagnated"] is False
    assert improving["tail_values"] == [0.31, 0.34]


def test_v15_stagnation_gate_blocks_without_flat_stage_tail() -> None:
    config = load_config("configs/v15_utility_querygrad_stagnate_longplus.yaml")
    torch.manual_seed(23)
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
            "selected_parent_scores": {1: 2.0, 2: 1.8, 3: 1.4, 4: 1.2},
            "unselected_parent_scores": {5: 1.0, 6: 0.9},
            "stage_stagnated": False,
            "stage_stagnation_window": 2,
            "stage_stagnation_delta": config.growth.mutation_stagnation_delta,
            "stage_val_tail": [0.42, 0.46],
            "stage_val_range": 0.04,
        },
        next_stage_index=6,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
        mutation_min_visit_z=config.growth.mutation_min_visit_z,
        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
        mutation_require_stagnation=config.growth.mutation_require_stagnation,
    )

    assert split_stats["mutated_parent_ids"] == []
    assert split_stats["mutation_requires_stagnation"] is True
    assert split_stats["mutation_stagnated"] is False


def test_v15_stagnation_gate_allows_final_stage_mutation_when_flat() -> None:
    config = load_config("configs/v15_utility_querygrad_stagnate_longplus.yaml")
    torch.manual_seed(29)
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
            "selected_parent_scores": {1: 2.0, 2: 1.8, 3: 1.4, 4: 1.2},
            "unselected_parent_scores": {5: 1.0, 6: 0.9},
            "stage_stagnated": True,
            "stage_stagnation_window": 2,
            "stage_stagnation_delta": config.growth.mutation_stagnation_delta,
            "stage_val_tail": [0.42, 0.43],
            "stage_val_range": 0.01,
        },
        next_stage_index=6,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
        mutation_min_visit_z=config.growth.mutation_min_visit_z,
        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
        mutation_require_stagnation=config.growth.mutation_require_stagnation,
    )

    assert split_stats["mutation_reference_kind"] == "best_unselected"
    assert split_stats["mutation_reference_score"] == 1.0
    assert split_stats["mutated_parent_ids"] == [1, 2]
    assert split_stats["mutation_stagnated"] is True
    assert split_stats["mutation_stage_val_tail"] == [0.42, 0.43]
