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


def test_v34_generator_outputs_expected_variants(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "gen_v34_configs.py", "gen_v34_configs")
    module.CONFIGS = tmp_path / "configs"
    module.SCRIPTS = tmp_path / "scripts"
    module.main()

    cfg_names = sorted(path.name for path in module.CONFIGS.glob("v34_*.yaml"))
    assert "v34_core_visit_taskgrad_half_agree_mutate_z00_m075_f050_32_l.yaml" in cfg_names
    assert "v34_core_visit_taskgrad_half_agree_mutate_z00_m075_f025_32_l.yaml" in cfg_names
    assert "v34_core_visit_taskgrad_half_agree_mutate_z00_m075_stag_32_l.yaml" in cfg_names
    assert "v34_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f050_stag_32_xl.yaml" in cfg_names


def test_v34_variant_knobs_are_wired_correctly() -> None:
    sparse = yaml.safe_load(
        (ROOT / "configs" / "v34_core_visit_taskgrad_half_agree_mutate_z00_m075_f050_32_xl.yaml").read_text()
    )
    assert sparse["growth"]["mutation_selected_fraction"] == 0.5
    assert sparse["growth"]["mutation_score_margin"] == 0.75
    assert sparse["growth"]["mutation_stage_index_min"] == 6

    stag = yaml.safe_load(
        (ROOT / "configs" / "v34_t1_visit_taskgrad_half_agree_mutate_z00_m075_stag_32_l.yaml").read_text()
    )
    assert stag["growth"]["mutation_require_stagnation"] is True
    assert stag["growth"]["mutation_selected_fraction"] == 1.0
    assert stag["task"]["train_eval_writers"] == [4, 8, 12, 14]

    sparse_stag = yaml.safe_load(
        (ROOT / "configs" / "v34_t2a_visit_taskgrad_half_agree_mutate_z00_m075_f050_stag_32_xl.yaml").read_text()
    )
    assert sparse_stag["growth"]["mutation_selected_fraction"] == 0.5
    assert sparse_stag["growth"]["mutation_require_stagnation"] is True
    assert sparse_stag["task"]["start_node_pool_size"] == 1


def test_v34_report_completion_filter_excludes_incomplete_runs(tmp_path: Path) -> None:
    module = load_module(ROOT / "scripts" / "build_v34_report.py", "build_v34_report")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4300}) + "\n", encoding="utf-8")
    (run_dir / "last.pt").write_text("checkpoint", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is False

    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_run(run_dir, 4590) is True
