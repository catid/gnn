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


def test_v56_generator_emits_package_friendly_slice():
    module = load_module("gen_v56_configs", "scripts/gen_v56_configs.py")
    assert set(module.ARMS) == {
        "stageadaptive_late_half",
        "stageadaptive_late_half_agree_mutate_z00_m075_f0125",
    }
    assert set(module.REGIMES) == {"t1", "t1r", "t2c"}
    cfg = module.build_config("stageadaptive_late_half_agree_mutate_z00_m075_f0125", "t2c")
    assert cfg["growth"]["mutation_selected_fraction"] == 0.125


def test_v56_eval_completion_filter(tmp_path):
    module = load_module("run_v56_eval_sweep", "scripts/run_v56_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v56_report_builder_tracks_fresh_and_pooled_package_slice():
    module = load_module("build_v56_report", "scripts/build_v56_report.py")
    assert set(module.SELECTORS) == {
        "stageadaptive_late_half",
        "stageadaptive_late_half_agree_mutate_z00_m075_f0125",
    }
    assert set(module.PHASES) == {
        "fresh_t1_xl",
        "fresh_t1r_xl",
        "fresh_t2c_xl",
        "pooled_t1_xl",
        "pooled_t1r_xl",
        "pooled_t2c_xl",
    }
