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


def test_v41_generator_wires_stageadaptive_late_half():
    module = load_module("gen_v41_configs", "scripts/gen_v41_configs.py")
    cfg = module.build_config("stageadaptive_late_half", "t2a")
    growth = cfg["growth"]
    assert growth["utility_visit_weight"] == 1.0
    assert growth["utility_grad_weight"] == 0.0
    assert growth["adaptive_selector_stage_index_min"] == 5
    assert growth["adaptive_utility_grad_weight"] == 0.5
    assert cfg["task"]["start_node_pool_size"] == 1


def test_v41_eval_completion_filter(tmp_path):
    module = load_module("run_v41_eval_sweep", "scripts/run_v41_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v41_report_builder_knows_t1_t1r_t2a():
    module = load_module("build_v41_report", "scripts/build_v41_report.py")
    assert set(module.PHASES) == {"t1_xl", "t1r_xl", "t2a_xl"}
    assert set(module.SELECTORS) == {"visit_taskgrad_half", "stageadaptive_late_half"}
