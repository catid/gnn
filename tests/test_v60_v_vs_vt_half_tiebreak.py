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


def test_v57_generator_emits_all_rule_regimes():
    module = load_module("gen_v60_configs", "scripts/gen_v60_configs.py")
    assert set(module.ARMS) == {"visitonly", "visit_taskgrad_half"}
    assert set(module.REGIMES) == {"core", "t1", "t1r", "t2a", "t2b", "t2c"}
    cfg = module.build_config("visitonly", "t2c")
    assert cfg["task"]["writers_per_episode"] == 6
    assert cfg["growth"]["utility_grad_weight"] == 0.0


def test_v57_eval_completion_filter(tmp_path):
    module = load_module("run_v60_eval_sweep", "scripts/run_v60_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v57_report_builder_tracks_rules_and_scopes():
    module = load_module("build_v60_report", "scripts/build_v60_report.py")
    assert set(module.SELECTORS) == {"visitonly", "visit_taskgrad_half"}
    assert set(module.RULES) == {
        "single_selector_v",
        "single_selector_vt_half",
        "regime_keyed_vt_home_ingress_v_transfer_non_ingress",
    }
    assert module.PHASES["fresh_core_xl"]["family"] == "home"
    assert module.PHASES["fresh_t2a_xl"]["family"] == "ingress"
    assert module.PHASES["fresh_t2b_xl"]["family"] == "non_ingress_stress"
    assert module.PHASES["pooled_t2b_xl"]["family"] == "non_ingress_stress"
