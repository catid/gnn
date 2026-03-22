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


def test_v36_generator_outputs_expected_configs(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "gen_v36_configs.py", "gen_v36_configs")
    module.CONFIGS = tmp_path / "configs"
    module.SCRIPTS = tmp_path / "scripts"
    module.main()
    cfg_names = sorted(path.name for path in module.CONFIGS.glob("v36_*.yaml"))
    assert "v36_core_visit_taskgrad_half_32_xl.yaml" in cfg_names
    assert "v36_t1_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml" in cfg_names
    assert "v36_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml" in cfg_names
    assert "v36_t1r_visit_taskgrad_half_32_xl.yaml" in cfg_names


def test_v36_sparse_package_mapping() -> None:
    cfg = yaml.safe_load(
        (ROOT / "configs" / "v36_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml").read_text()
    )
    assert cfg["train"]["train_steps"] == 4590
    assert cfg["growth"]["mutation_score_margin"] == 0.75
    assert cfg["growth"]["mutation_selected_fraction"] == 0.25
    assert cfg["growth"]["mutation_min_visit_z"] == 0.0
    assert cfg["task"]["start_node_pool_size"] == 1
    assert cfg["task"]["train_eval_writers"] == [4, 8, 12, 14]


def test_v36_report_completion_filter(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "build_v36_report.py", "build_v36_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4000}) + "\n", encoding="utf-8")
    (run_dir / "last.pt").write_text("checkpoint", encoding="utf-8")
    assert module.is_complete_run(run_dir) is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_run(run_dir) is True
