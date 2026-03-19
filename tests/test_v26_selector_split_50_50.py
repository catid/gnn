from __future__ import annotations

import json
import runpy
from pathlib import Path

import yaml

from apsgnn.config import GrowthConfig, selector_weights_for_stage


ROOT = Path(__file__).resolve().parents[1]
LR_CHOICES_PATH = ROOT / "reports" / "v26_lr_choices.json"
REPORT_HELPERS = runpy.run_path(str(ROOT / "scripts" / "build_v26_report.py"))
choose_lr_multipliers = REPORT_HELPERS["choose_lr_multipliers"]
combined_rank = REPORT_HELPERS["combined_rank"]
screening_composite = REPORT_HELPERS["screening_composite"]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_selector_weights_for_stage_switching() -> None:
    growth = GrowthConfig(
        utility_visit_weight=1.0,
        utility_grad_weight=0.0,
        utility_query_grad_weight=0.0,
        adaptive_selector_stage_index_min=5,
        adaptive_utility_visit_weight=1.0,
        adaptive_utility_grad_weight=0.5,
        adaptive_utility_query_grad_weight=0.0,
    )
    early = selector_weights_for_stage(growth, 4)
    late = selector_weights_for_stage(growth, 5)
    assert early["utility_visit_weight"] == 1.0
    assert early["utility_grad_weight"] == 0.0
    assert late["utility_visit_weight"] == 1.0
    assert late["utility_grad_weight"] == 0.5


def test_v26_generator_outputs_expected_weights_and_lr_choices() -> None:
    backup = LR_CHOICES_PATH.read_text() if LR_CHOICES_PATH.exists() else None
    try:
        LR_CHOICES_PATH.write_text(
            json.dumps(
                {
                    "visitonly": 0.8,
                    "visit_taskgrad_half": 1.0,
                    "stageadaptive_vt": 0.8,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        runpy.run_path(str(ROOT / "scripts" / "gen_v26_configs.py"), run_name="__main__")

        visit = load_yaml("v26_core_visitonly_32_m.yaml")
        vt_half = load_yaml("v26_core_visit_taskgrad_half_32_m.yaml")
        adaptive = load_yaml("v26_core_stageadaptive_vt_32_m.yaml")
        pilot_query = load_yaml("v26_core_querygradonly_32_p_lr080.yaml")

        assert abs(float(visit["train"]["lr"]) - 1.6e-4) < 1e-12
        assert abs(float(vt_half["train"]["lr"]) - 2.0e-4) < 1e-12
        assert abs(float(pilot_query["train"]["lr"]) - 1.6e-4) < 1e-12

        assert vt_half["growth"]["utility_visit_weight"] == 1.0
        assert vt_half["growth"]["utility_grad_weight"] == 0.5
        assert vt_half["growth"]["utility_query_grad_weight"] == 0.0

        assert adaptive["growth"]["utility_visit_weight"] == 1.0
        assert adaptive["growth"]["utility_grad_weight"] == 0.0
        assert adaptive["growth"]["adaptive_selector_stage_index_min"] == 5
        assert adaptive["growth"]["adaptive_utility_grad_weight"] == 0.5
    finally:
        if backup is None:
            LR_CHOICES_PATH.unlink(missing_ok=True)
        else:
            LR_CHOICES_PATH.write_text(backup, encoding="utf-8")


def test_v26_schedule_and_regime_matching() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v26_configs.py"), run_name="__main__")

    visit = load_yaml("v26_t1_visitonly_32_m.yaml")
    vt_half = load_yaml("v26_t1_visit_taskgrad_half_32_m.yaml")
    t2a = load_yaml("v26_t2a_visitonly_32_l.yaml")
    t2c = load_yaml("v26_t2c_visitonly_32_l.yaml")

    assert visit["train"]["train_steps"] == 2550
    assert vt_half["train"]["train_steps"] == 2550
    assert visit["growth"]["stage_steps"] == [90, 90, 120, 120, 150, 180, 1800]
    assert vt_half["growth"]["stage_steps"] == [90, 90, 120, 120, 150, 180, 1800]
    assert visit["task"]["train_eval_writers"] == [4, 8, 12, 14]
    assert t2a["task"]["start_node_pool_size"] == 1
    assert t2c["task"]["writers_per_episode"] == 6
    assert t2c["task"]["train_eval_writers"] == [6, 10, 14, 16]


def test_v26_promotion_helpers() -> None:
    record = {"dense_mean": 0.2, "last_val": 0.1, "last5_val_mean": 0.3}
    assert abs(screening_composite(record) - (0.45 * 0.2 + 0.35 * 0.1 + 0.20 * 0.3)) < 1e-12

    lr_choice = choose_lr_multipliers(
        [
            {"selector": "visitonly", "lr_multiplier": 0.8, "screen_composite": 0.10},
            {"selector": "visitonly", "lr_multiplier": 1.0, "screen_composite": 0.09},
            {"selector": "querygradonly", "lr_multiplier": 0.8, "screen_composite": 0.08},
            {"selector": "querygradonly", "lr_multiplier": 1.0, "screen_composite": 0.08},
        ]
    )
    assert lr_choice["visitonly"] == 0.8
    assert lr_choice["querygradonly"] == 1.0

    rank = combined_rank(
        {
            "visit_taskgrad_0375": {"screen_composite": {"mean": 0.03}},
            "querygradonly": {"screen_composite": {"mean": 0.02}},
        },
        {
            "visit_taskgrad_0375": {"screen_composite": {"mean": 0.04}},
            "querygradonly": {"screen_composite": {"mean": 0.01}},
        },
        ["querygradonly", "visit_taskgrad_0375"],
    )
    assert rank == ["visit_taskgrad_0375", "querygradonly"]
