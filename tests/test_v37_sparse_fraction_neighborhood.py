from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v37_generator_outputs_expected_variants(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "gen_v37_configs.py", "gen_v37_configs")
    module.CONFIGS = tmp_path / "configs"
    module.SCRIPTS = tmp_path / "scripts"
    module.main()

    cfg_names = sorted(path.name for path in module.CONFIGS.glob("v37_*.yaml"))
    assert "v37_core_visit_taskgrad_half_agree_mutate_z00_m075_f0125_32_l.yaml" in cfg_names
    assert "v37_t1_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml" in cfg_names
    assert "v37_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f0375_32_xl.yaml" in cfg_names
    assert "v37_t1r_visit_taskgrad_half_32_xl.yaml" in cfg_names


def test_v37_fraction_knobs_are_wired_correctly() -> None:
    cfg_0125 = yaml.safe_load(
        (ROOT / "configs" / "v37_core_visit_taskgrad_half_agree_mutate_z00_m075_f0125_32_l.yaml").read_text()
    )
    assert cfg_0125["growth"]["mutation_selected_fraction"] == 0.125
    assert cfg_0125["growth"]["mutation_score_margin"] == 0.75

    cfg_0375 = yaml.safe_load(
        (ROOT / "configs" / "v37_t1_visit_taskgrad_half_agree_mutate_z00_m075_f0375_32_xl.yaml").read_text()
    )
    assert cfg_0375["growth"]["mutation_selected_fraction"] == 0.375
    assert cfg_0375["growth"]["mutation_stage_index_min"] == 6
    assert cfg_0375["task"]["train_eval_writers"] == [4, 8, 12, 14]


def test_v37_report_completion_filter(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "build_v37_report.py", "build_v37_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "last.pt").write_text("checkpoint", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is True
