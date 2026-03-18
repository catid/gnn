from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import transition_model_for_growth
from apsgnn.model import APSGNNModel


def test_v12_adaptive_mutation_uses_best_unselected_margin() -> None:
    config = load_config("configs/v12_utility_querygrad_adaptmut_longplus.yaml")
    torch.manual_seed(11)
    model = APSGNNModel(config)
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=16,
        next_active_compute_nodes=24,
        transition_mode="split",
        split_mode="mutate",
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=[(1, 17), (2, 18), (3, 19), (4, 20)],
        transition_stats={
            "selected_parent_scores": {1: 1.5, 2: 0.9, 3: 0.6, 4: 0.1},
            "unselected_parent_scores": {5: 0.5, 6: 0.2},
        },
        next_stage_index=5,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
    )

    assert split_stats["mutation_reference_kind"] == "best_unselected"
    assert split_stats["mutation_reference_score"] == 0.5
    assert split_stats["mutated_parent_ids"] == [1, 2]


def test_v12_adaptive_mutation_uses_selected_mean_when_all_selected() -> None:
    config = load_config("configs/v12_utility_querygrad_adaptmut_longplus.yaml")
    torch.manual_seed(13)
    model = APSGNNModel(config)
    split_pairs = [(parent_id, 24 + index + 1) for index, parent_id in enumerate(range(1, 9))]
    split_stats = transition_model_for_growth(
        model,
        previous_active_compute_nodes=24,
        next_active_compute_nodes=32,
        transition_mode="split",
        split_mode="mutate",
        mutation_scale=config.growth.split_mutation_scale,
        seed=1234,
        selective_parent_child_pairs=split_pairs,
        transition_stats={
            "selected_parent_scores": {
                split_pairs[0][0]: 1.4,
                split_pairs[1][0]: 1.1,
                split_pairs[2][0]: 0.8,
                split_pairs[3][0]: 0.3,
                split_pairs[4][0]: 0.1,
                split_pairs[5][0]: -0.1,
                split_pairs[6][0]: -0.4,
                split_pairs[7][0]: -0.8,
            },
            "unselected_parent_scores": {},
        },
        next_stage_index=6,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
    )

    assert split_stats["mutation_reference_kind"] == "selected_mean"
    assert abs(split_stats["mutation_reference_score"] - 0.3) < 1.0e-6
    assert split_stats["mutated_parent_ids"] == [split_pairs[0][0], split_pairs[1][0], split_pairs[2][0]]
