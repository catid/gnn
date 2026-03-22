from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v40_generator_emits_fraction_variants():
    module = load_module("gen_v40_configs", "scripts/gen_v40_configs.py")
    cfg_125 = module.build_config("visit_taskgrad_half_agree_mutate_z00_m075_f0125", "t1")
    cfg_25 = module.build_config("visit_taskgrad_half_agree_mutate_z00_m075_f025", "t1r")
    assert cfg_125["growth"]["mutation_selected_fraction"] == 0.125
    assert cfg_25["growth"]["mutation_selected_fraction"] == 0.25


def test_v40_eval_completion_filter(tmp_path):
    module = load_module("run_v40_eval_sweep", "scripts/run_v40_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v40_report_builder_knows_three_arms():
    module = load_module("build_v40_report", "scripts/build_v40_report.py")
    assert len(module.SELECTORS) == 3
    assert set(module.PHASES) == {"t1_xl", "t1r_xl"}
