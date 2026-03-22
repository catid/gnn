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


def test_v32_generator_outputs_expected_margin_variants(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "gen_v32_configs.py", "gen_v32_configs")
    module.CONFIGS = tmp_path / "configs"
    module.SCRIPTS = tmp_path / "scripts"
    module.main()

    cfg_names = sorted(path.name for path in module.CONFIGS.glob("v32_*.yaml"))
    assert "v32_core_visit_taskgrad_half_agree_mutate_z00_m050_32_l.yaml" in cfg_names
    assert "v32_core_visit_taskgrad_half_agree_mutate_z00_m075_32_l.yaml" in cfg_names
    assert "v32_core_visit_taskgrad_half_agree_mutate_z00_m100_32_l.yaml" in cfg_names
    assert "v32_t2a_visit_taskgrad_half_agree_mutate_z00_m075_32_xl.yaml" in cfg_names


def test_v32_margin_schedule_and_regime_matching() -> None:
    config_path = ROOT / "configs" / "v32_core_visit_taskgrad_half_agree_mutate_z00_m075_32_xl.yaml"
    config = yaml.safe_load(config_path.read_text())
    assert config["train"]["train_steps"] == 4590
    assert config["growth"]["stage_steps"] == [120, 120, 150, 150, 180, 220, 3650]
    assert config["growth"]["mutation_score_margin"] == 0.75
    assert config["growth"]["mutation_min_visit_z"] == 0.0
    assert config["task"]["start_node_pool_size"] == 2
    assert config["task"]["train_eval_writers"] == [2, 6, 10]

    t2a_path = ROOT / "configs" / "v32_t2a_visit_taskgrad_half_agree_mutate_z00_m100_32_xl.yaml"
    t2a = yaml.safe_load(t2a_path.read_text())
    assert t2a["task"]["start_node_pool_size"] == 1
    assert t2a["task"]["train_eval_writers"] == [4, 8, 12, 14]
    assert t2a["growth"]["mutation_score_margin"] == 1.0


def test_v32_report_completion_filter_excludes_smoke_and_in_progress(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "build_v32_report.py", "build_v32_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4000}) + "\n", encoding="utf-8")
    (run_dir / "last.pt").write_text("checkpoint", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is False

    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is True
