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


def test_v38_generator_outputs_expected_variants(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "gen_v38_configs.py", "gen_v38_configs")
    module.CONFIGS = tmp_path / "configs"
    module.SCRIPTS = tmp_path / "scripts"
    module.main()

    cfg_names = sorted(path.name for path in module.CONFIGS.glob("v38_*.yaml"))
    assert "v38_core_visit_taskgrad_half_32_xl.yaml" in cfg_names
    assert "v38_t1_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml" in cfg_names
    assert "v38_t1r_visit_taskgrad_half_32_xl.yaml" in cfg_names
    assert "v38_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml" in cfg_names


def test_v38_sparse_package_knobs_are_wired_correctly() -> None:
    cfg = yaml.safe_load(
        (ROOT / "configs" / "v38_t1_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_xl.yaml").read_text()
    )
    assert cfg["growth"]["split_mode"] == "mutate"
    assert cfg["growth"]["mutation_selected_fraction"] == 0.25
    assert cfg["growth"]["mutation_score_margin"] == 0.75
    assert cfg["growth"]["mutation_stage_index_min"] == 6

    base = yaml.safe_load((ROOT / "configs" / "v38_core_visit_taskgrad_half_32_xl.yaml").read_text())
    assert base["growth"]["split_mode"] == "clone"
    assert base["task"]["train_eval_writers"] == [2, 6, 10]


def test_v38_report_completion_filter(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "build_v38_report.py", "build_v38_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "last.pt").write_text("checkpoint", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is True
