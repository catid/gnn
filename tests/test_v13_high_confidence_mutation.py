from __future__ import annotations

import torch

from apsgnn.config import load_config
from apsgnn.growth import transition_model_for_growth
from apsgnn.model import APSGNNModel


def test_v13_high_confidence_gate_skips_pre_final_transition() -> None:
    config = load_config("configs/v13_utility_querygrad_hiconf_longplus.yaml")
    torch.manual_seed(17)
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
            "selected_parent_scores": {1: 2.0, 2: 1.8, 3: 1.4, 4: 1.2},
            "unselected_parent_scores": {5: 0.5, 6: 0.2},
        },
        next_stage_index=5,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
    )

    assert split_stats["mutated_parent_ids"] == []
    assert split_stats["mutation_reference_kind"] is None
    assert split_stats["mutation_reference_score"] is None


def test_v13_high_confidence_gate_requires_large_margin_in_final_stage() -> None:
    config = load_config("configs/v13_utility_querygrad_hiconf_longplus.yaml")
    torch.manual_seed(19)
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
            "selected_parent_scores": {1: 1.8, 2: 1.4, 3: 1.2, 4: 0.9},
            "unselected_parent_scores": {5: 0.8, 6: 0.7},
        },
        next_stage_index=6,
        mutation_stage_index_min=config.growth.mutation_stage_index_min,
        mutation_selected_fraction=config.growth.mutation_selected_fraction,
        mutation_score_margin=config.growth.mutation_score_margin,
    )

    assert split_stats["mutation_reference_kind"] == "best_unselected"
    assert split_stats["mutation_reference_score"] == 0.8
    assert split_stats["mutated_parent_ids"] == [1]
