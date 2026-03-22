from __future__ import annotations

import importlib.util
import json
import runpy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(name: str) -> dict:
    return yaml.safe_load((ROOT / "configs" / name).read_text())


def test_v31_generator_outputs_expected_threshold_variants() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v31_configs.py"), run_name="__main__")

    baseline = load_yaml("v31_core_visit_taskgrad_half_32_l.yaml")
    loose = load_yaml("v31_core_visit_taskgrad_half_agree_mutate_z00_32_l.yaml")
    default = load_yaml("v31_core_visit_taskgrad_half_agree_mutate_z25_32_l.yaml")
    tight = load_yaml("v31_t1_visit_taskgrad_half_agree_mutate_z50_32_xl.yaml")

    assert baseline["growth"]["split_mode"] == "clone"
    assert baseline["growth"]["utility_grad_weight"] == 0.5

    assert loose["growth"]["split_mode"] == "mutate"
    assert loose["growth"]["mutation_min_visit_z"] == 0.0
    assert loose["growth"]["mutation_min_query_grad_z"] == 0.0

    assert default["growth"]["mutation_min_visit_z"] == 0.25
    assert default["growth"]["mutation_min_query_grad_z"] == 0.25
    assert default["growth"]["mutation_score_margin"] == 0.75

    assert tight["growth"]["mutation_min_visit_z"] == 0.5
    assert tight["growth"]["mutation_min_query_grad_z"] == 0.5


def test_v31_schedule_and_regime_matching() -> None:
    runpy.run_path(str(ROOT / "scripts" / "gen_v31_configs.py"), run_name="__main__")

    core_l = load_yaml("v31_core_visit_taskgrad_half_agree_mutate_z25_32_l.yaml")
    core_xl = load_yaml("v31_core_visit_taskgrad_half_agree_mutate_z25_32_xl.yaml")
    t1_l = load_yaml("v31_t1_visit_taskgrad_half_agree_mutate_z00_32_l.yaml")
    t2a_xl = load_yaml("v31_t2a_visit_taskgrad_half_agree_mutate_z50_32_xl.yaml")

    assert core_l["growth"]["stage_steps"] == [120, 120, 150, 150, 180, 220, 2630]
    assert core_xl["growth"]["stage_steps"] == [120, 120, 150, 150, 180, 220, 3650]
    assert t1_l["task"]["train_eval_writers"] == [4, 8, 12, 14]
    assert t2a_xl["task"]["start_node_pool_size"] == 1


def test_v31_report_completion_filter_excludes_smoke_and_in_progress(tmp_path: Path) -> None:
    script_path = ROOT / "scripts" / "build_v31_report.py"
    spec = importlib.util.spec_from_file_location("build_v31_report", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def make_run(name: str, train_steps: int, max_step: int) -> Path:
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "last.pt").write_text("x")
        (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": train_steps}}))
        rows = []
        for step in (min(50, max_step), max_step):
            rows.append({"step": step, "val/query_accuracy": 0.1})
        (run_dir / "metrics.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        return run_dir

    smoke = make_run("smoke", train_steps=20, max_step=20)
    in_progress = make_run("in_progress", train_steps=3570, max_step=750)
    complete = make_run("complete", train_steps=3570, max_step=3570)

    assert module.is_complete_run(smoke, 3570) is False
    assert module.is_complete_run(in_progress, 3570) is False
    assert module.is_complete_run(complete, 3570) is True
