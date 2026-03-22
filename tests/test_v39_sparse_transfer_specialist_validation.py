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


def test_v39_generator_emits_t2b_and_sparse_fraction(tmp_path):
    module = load_module("gen_v39_configs", "scripts/gen_v39_configs.py")
    config = module.build_config("visit_taskgrad_half_agree_mutate_z00_m075_f025", "t2b")
    assert config["task"]["query_ttl_min"] == 2
    assert config["task"]["query_ttl_max"] == 2
    assert config["growth"]["mutation_selected_fraction"] == 0.25
    assert config["growth"]["mutation_score_margin"] == 0.75
    path = tmp_path / "cfg.yaml"
    module.dump_yaml(config, path)
    roundtrip = yaml.safe_load(path.read_text())
    assert roundtrip["runtime"]["run_name"].startswith("v39-t2b-")


def test_v39_eval_sweep_filters_for_expected_xl_steps(tmp_path):
    module = load_module("run_v39_eval_sweep", "scripts/run_v39_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 4590}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 4590}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "xl") is True


def test_v39_report_builder_uses_transfer_only_phases(tmp_path):
    module = load_module("build_v39_report", "scripts/build_v39_report.py")
    assert set(module.PHASES) == {"t1_xl", "t1r_xl", "t2b_xl"}
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    cfg = {"train": {"train_steps": 4590}}
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    (run_dir / "last.pt").write_bytes(b"x")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"step": 2550, "val/query_accuracy": 0.0417}),
                json.dumps({"step": 4590, "val/query_accuracy": 0.0833}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for kind in ("best", "last"):
        for writer, acc in ((4, 0.0), (8, 0.0417), (12, 0.0625), (14, 0.0833)):
            payload = {"metrics": {"query_accuracy": acc}}
            (run_dir / f"eval_{kind}_k{writer}.json").write_text(json.dumps(payload), encoding="utf-8")
    record = module.summarize_run(run_dir, "visit_taskgrad_half", "t1_xl", 1234)
    assert record["dense_mean"] > 0.0
    assert record["score"] == record["last_val"] + record["dense_mean"]
