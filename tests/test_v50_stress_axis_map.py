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


def test_v50_generator_emits_t2b_t2c():
    module = load_module("gen_v50_configs", "scripts/gen_v50_configs.py")
    assert set(module.REGIMES) == {"t2b", "t2c"}
    cfg = module.build_config("stageadaptive_late_half", "t2c")
    assert cfg["task"]["writers_per_episode"] == 6
    assert cfg["task"]["train_eval_writers"] == [6, 10, 14, 16]


def test_v50_eval_completion_filter(tmp_path):
    module = load_module("run_v50_eval_sweep", "scripts/run_v50_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v50_report_builder_has_t2_axes():
    module = load_module("build_v50_report", "scripts/build_v50_report.py")
    assert set(module.PHASES) == {"t2b_xl", "t2c_xl"}
    assert set(module.SELECTORS) == {"visit_taskgrad_half", "stageadaptive_late_half"}
