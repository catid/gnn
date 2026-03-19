from __future__ import annotations

import runpy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_v25_generator_outputs_expected_weights() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v25_configs.py"), run_name="__main__")

    visit = load_yaml("v25_core_visitonly_32_l.yaml")
    query = load_yaml("v25_core_querygradonly_32_l.yaml")
    vt_half = load_yaml("v25_core_visit_taskgrad_half_32_l.yaml")

    assert visit["growth"]["utility_visit_weight"] == 1.0
    assert visit["growth"]["utility_grad_weight"] == 0.0
    assert visit["growth"]["utility_query_grad_weight"] == 0.0

    assert query["growth"]["utility_visit_weight"] == 0.0
    assert query["growth"]["utility_grad_weight"] == 0.0
    assert query["growth"]["utility_query_grad_weight"] == 1.0

    assert vt_half["growth"]["utility_visit_weight"] == 1.0
    assert vt_half["growth"]["utility_grad_weight"] == 0.5
    assert vt_half["growth"]["utility_query_grad_weight"] == 0.0


def test_v25_schedule_matching_and_regime_shapes() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v25_configs.py"), run_name="__main__")

    configs = {
        "visit": load_yaml("v25_t1_visitonly_32_xl.yaml"),
        "query": load_yaml("v25_t1_querygradonly_32_xl.yaml"),
        "vt_half": load_yaml("v25_t1_visit_taskgrad_half_32_xl.yaml"),
    }

    stage_steps = [120, 120, 150, 150, 180, 220, 3650]
    for cfg in configs.values():
        assert cfg["train"]["train_steps"] == 4590
        assert cfg["growth"]["stage_steps"] == stage_steps
        assert cfg["task"]["train_eval_writers"] == [4, 8, 12, 14]
        assert cfg["task"]["start_node_pool_size"] == 2

    t2a = load_yaml("v25_t2a_visit_taskgrad_half_32_xl.yaml")
    assert t2a["task"]["start_node_pool_size"] == 1
    assert t2a["task"]["train_eval_writers"] == [4, 8, 12, 14]


def test_v25_scripts_are_generated() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v25_configs.py"), run_name="__main__")
    for path in [
        ROOT / "scripts" / "train_v25_core_visitonly_32_l.sh",
        ROOT / "scripts" / "train_v25_core_querygradonly_32_xl.sh",
        ROOT / "scripts" / "train_v25_t1_visit_taskgrad_half_32_r.sh",
        ROOT / "scripts" / "train_v25_t2a_visit_taskgrad_half_32_xl.sh",
        ROOT / "scripts" / "smoke_v25_t1_querygradonly_32_l.sh",
    ]:
        assert path.exists(), path
