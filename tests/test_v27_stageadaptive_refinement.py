from __future__ import annotations

import runpy
from pathlib import Path

import yaml

from apsgnn.config import GrowthConfig, selector_weights_for_stage


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_stageadaptive_selector_switch_points() -> None:
    early = GrowthConfig(
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        adaptive_selector_stage_index_min=4,
        adaptive_utility_visit_weight=1.0,
        adaptive_utility_grad_weight=0.5,
    )
    final = GrowthConfig(
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        adaptive_selector_stage_index_min=6,
        adaptive_utility_visit_weight=1.0,
        adaptive_utility_grad_weight=0.5,
    )
    assert selector_weights_for_stage(early, 3)["utility_grad_weight"] == 0.0
    assert selector_weights_for_stage(early, 4)["utility_grad_weight"] == 0.5
    assert selector_weights_for_stage(final, 5)["utility_grad_weight"] == 0.0
    assert selector_weights_for_stage(final, 6)["utility_grad_weight"] == 0.5


def test_v27_generator_outputs_expected_configs() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v27_configs.py"), run_name="__main__")

    early = load_yaml("v27_t1_stageadaptive_early_half_32_s.yaml")
    late375 = load_yaml("v27_core_stageadaptive_late_0375_32_m.yaml")
    final = load_yaml("v27_t2a_stageadaptive_final_half_32_l.yaml")
    baseline = load_yaml("v27_core_visit_taskgrad_half_32_m.yaml")

    assert early["growth"]["adaptive_selector_stage_index_min"] == 4
    assert early["growth"]["adaptive_utility_grad_weight"] == 0.5
    assert late375["growth"]["adaptive_selector_stage_index_min"] == 5
    assert late375["growth"]["adaptive_utility_grad_weight"] == 0.375
    assert final["growth"]["adaptive_selector_stage_index_min"] == 6
    assert final["task"]["start_node_pool_size"] == 1
    assert final["train"]["train_steps"] == 3570
    assert baseline["growth"]["utility_grad_weight"] == 0.5
    assert baseline["growth"].get("adaptive_selector_stage_index_min", -1) == -1


def test_v27_schedule_and_regime_matching() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v27_configs.py"), run_name="__main__")

    core_s = load_yaml("v27_core_stageadaptive_late_half_32_s.yaml")
    t1_m = load_yaml("v27_t1_stageadaptive_late_half_32_m.yaml")
    t2a_l = load_yaml("v27_t2a_stageadaptive_late_half_32_l.yaml")

    assert core_s["growth"]["stage_steps"] == [60, 60, 75, 75, 90, 120, 870]
    assert t1_m["growth"]["stage_steps"] == [90, 90, 120, 120, 150, 180, 1800]
    assert t2a_l["growth"]["stage_steps"] == [120, 120, 150, 150, 180, 220, 2630]
    assert core_s["task"]["train_eval_writers"] == [2, 6, 10]
    assert t1_m["task"]["train_eval_writers"] == [4, 8, 12, 14]
    assert t2a_l["task"]["start_node_pool_size"] == 1
