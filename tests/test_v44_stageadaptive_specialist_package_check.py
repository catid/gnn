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


def test_v44_generator_emits_sparse_stageadaptive_arm():
    module = load_module("gen_v44_configs", "scripts/gen_v44_configs.py")
    cfg = module.build_config("stageadaptive_late_half_agree_mutate_z00_m075_f025", "t2a")
    assert cfg["growth"]["adaptive_selector_stage_index_min"] == 5
    assert cfg["growth"]["mutation_selected_fraction"] == 0.25
    assert cfg["task"]["start_node_pool_size"] == 1


def test_v44_eval_completion_filter(tmp_path):
    module = load_module("run_v44_eval_sweep", "scripts/run_v44_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v44_report_builder_has_transfer_specialist_phases():
    module = load_module("build_v44_report", "scripts/build_v44_report.py")
    assert set(module.PHASES) == {"t1_xl", "t1r_xl", "t2a_xl"}
    assert "stageadaptive_late_half_agree_mutate_z00_m075_f025" in module.SELECTORS
