from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import yaml

from apsgnn.config import GrowthConfig, TaskConfig, selector_decision_for_stage
from apsgnn.growth import CoverageTracker, GrowthStage


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v62_generator_emits_exact_regimes_and_schedule_shapes():
    module = load_module("gen_v62_configs", "scripts/gen_v62_configs.py")
    assert set(module.REGIMES) == {"core", "t1", "t1r", "t2a", "t2b", "t2c", "hmid", "hmix"}
    assert set(module.SCHEDULES) == {"p", "m", "l", "xl"}
    assert set(module.ARMS) == {"visitonly", "visit_taskgrad_half"}

    for schedule, min_fraction in (("m", 0.60), ("l", 0.65), ("xl", 0.70)):
        cfg = module.build_config("visit_taskgrad_half", "hmix", schedule)
        stage_steps = cfg["growth"]["stage_steps"]
        train_steps = cfg["train"]["train_steps"]
        assert sum(stage_steps) == train_steps
        assert stage_steps[-1] / train_steps >= min_fraction

    assert module.REGIMES["hmid"]["train_eval_writers"] == [3, 7, 11]
    assert module.REGIMES["hmix"]["start_node_pool_size"] == 1


def test_v62_static_selector_scores_are_correct():
    task = TaskConfig(writers_per_episode=3, start_node_pool_size=2, query_ttl_min=2, query_ttl_max=3)

    static_v = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.0)
    static_vt = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.5)

    assert selector_decision_for_stage(static_v, 3, task=task)["selected_selector"] == "V"
    assert selector_decision_for_stage(static_v, 3, task=task)["selected_selector_label"] == "V"
    assert selector_decision_for_stage(static_vt, 3, task=task)["selected_selector"] == "VT-0.5"
    assert selector_decision_for_stage(static_vt, 3, task=task)["selected_selector_label"] == "VT-0.5"


def test_v62_bootstrap_exclusion_remains_intact():
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.95,
        utility_tail_fraction=0.5,
    )
    stage = GrowthStage(index=0, active_compute_nodes=4, start_step=1, end_step=20, bootstrap_steps=5)
    tracker.update(
        step=1,
        stage=stage,
        task_visit_counts=torch.zeros(4),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.tensor([5.0, 3.0, 0.0, 0.0]),
        all_visit_counts=torch.tensor([5.0, 3.0, 0.0, 0.0]),
        task_gradient_signal=torch.zeros(4),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.tensor([2.0, 1.0, 0.0, 0.0]),
        all_gradient_signal=torch.tensor([2.0, 1.0, 0.0, 0.0]),
        success_visit_counts=torch.zeros(4),
    )
    snapshot = tracker.current_snapshot()
    assert snapshot["task_node_visit_coverage"] == 0.0
    assert snapshot["task_node_gradient_coverage"] == 0.0


def test_v62_selector_logging_and_predictiveness_fields_exist():
    tracker = CoverageTracker(
        num_compute_nodes=4,
        gradient_norm_threshold=1.0e-8,
        utility_ema_decay=0.95,
        utility_tail_fraction=0.5,
    )
    stage = GrowthStage(index=1, active_compute_nodes=4, start_step=1, end_step=8, bootstrap_steps=0)
    tracker.start_stage(
        stage,
        split_stats={
            "utility_alpha": 0.0,
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "utility_query_visit_weight": 0.0,
            "utility_query_grad_weight": 0.0,
            "eligible_parents": [1, 2],
            "selected_parents": [1],
            "unselected_parents": [2],
            "parent_components": {1: {"score": 2.0}, 2: {"score": 1.0}},
            "parent_to_children": {1: [1], 2: [2]},
            "sibling_pairs": [],
            "mutated_children": [],
            "selected_parent_scores": {1: 2.0},
            "unselected_parent_scores": {2: 1.0},
        },
    )
    tracker.update(
        step=1,
        stage=stage,
        task_visit_counts=torch.tensor([3.0, 1.0, 0.0, 0.0]),
        query_visit_counts=torch.zeros(4),
        bootstrap_visit_counts=torch.zeros(4),
        all_visit_counts=torch.tensor([3.0, 1.0, 0.0, 0.0]),
        task_gradient_signal=torch.tensor([0.7, 0.2, 0.0, 0.0]),
        query_gradient_signal=torch.zeros(4),
        bootstrap_gradient_signal=torch.zeros(4),
        all_gradient_signal=torch.tensor([0.7, 0.2, 0.0, 0.0]),
        success_visit_counts=torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    tracker.finalize()
    split_stats = tracker.completed_stages[0]["split_stats"]
    assert "selected_parent_child_usefulness_mean" in split_stats
    assert "unselected_parent_child_usefulness_mean" in split_stats
    assert "utility_usefulness_correlation" in split_stats
    assert "selected_parent_scores" in split_stats
    assert "unselected_parent_scores" in split_stats


def test_v62_eval_completion_filter_and_budget_matching(tmp_path):
    module = load_module("run_v62_eval_sweep", "scripts/run_v62_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_text("", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 2000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 2268}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "m") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 2268}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "m") is True


def test_v62_report_builder_ambiguity_logic():
    module = load_module("build_v62_report", "scripts/build_v62_report.py")
    main_summary = {
        "core": {
            "visitonly": {"composite": {"mean": 0.040, "std": 0.010}},
            "visit_taskgrad_half": {"composite": {"mean": 0.039, "std": 0.010}},
        },
        "t1": {
            "visitonly": {"composite": {"mean": 0.030, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.020, "std": 0.001}},
        },
        "t1r": {
            "visitonly": {"composite": {"mean": 0.030, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.020, "std": 0.001}},
        },
        "t2a": {
            "visitonly": {"composite": {"mean": 0.020, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.035, "std": 0.001}},
        },
        "t2b": {
            "visitonly": {"composite": {"mean": 0.020, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.021, "std": 0.001}},
        },
        "t2c": {
            "visitonly": {"composite": {"mean": 0.020, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.021, "std": 0.001}},
        },
        "hmid": {
            "visitonly": {"composite": {"mean": 0.020, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.0205, "std": 0.001}},
        },
        "hmix": {
            "visitonly": {"composite": {"mean": 0.019, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.021, "std": 0.001}},
        },
    }
    anchor_summary = {
        "core": {
            "visitonly": {"composite": {"mean": 0.030, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.040, "std": 0.001}},
        },
        "t1": {
            "visitonly": {"composite": {"mean": 0.030, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.020, "std": 0.001}},
        },
        "t2a": {
            "visitonly": {"composite": {"mean": 0.020, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.030, "std": 0.001}},
        },
        "hmix": {
            "visitonly": {"composite": {"mean": 0.025, "std": 0.001}},
            "visit_taskgrad_half": {"composite": {"mean": 0.023, "std": 0.001}},
        },
    }
    assert module.should_trigger_xl(main_summary, anchor_summary) is True
    picked = module.choose_ambiguity_regimes(main_summary, anchor_summary)
    assert len(picked) == 2
    assert "core" in picked


def test_v62_report_builder_selects_best_lr_per_selector():
    module = load_module("build_v62_report_rank", "scripts/build_v62_report.py")
    records = [
        {
            "arm": "visitonly",
            "arm_label": "V",
            "lr_multiplier": 0.6,
            "composite": 0.03,
            "dense_mean": 0.02,
            "last_dense_mean": 0.02,
            "last_val": 0.04,
            "last5_val_mean": 0.04,
            "best_val": 0.04,
            "best_to_last_drop": 0.0,
            "score": 0.06,
            "query_first_hop_home_rate": 0.1,
            "delivery_rate": 1.0,
            "home_to_out_rate": 0.8,
            "regime": "core",
            "run": "run-a",
        },
        {
            "arm": "visitonly",
            "arm_label": "V",
            "lr_multiplier": 1.0,
            "composite": 0.05,
            "dense_mean": 0.03,
            "last_dense_mean": 0.03,
            "last_val": 0.06,
            "last5_val_mean": 0.06,
            "best_val": 0.06,
            "best_to_last_drop": 0.0,
            "score": 0.09,
            "query_first_hop_home_rate": 0.1,
            "delivery_rate": 1.0,
            "home_to_out_rate": 0.8,
            "regime": "core",
            "run": "run-b",
        },
        {
            "arm": "visit_taskgrad_half",
            "arm_label": "VT-0.5",
            "lr_multiplier": 0.8,
            "composite": 0.04,
            "dense_mean": 0.02,
            "last_dense_mean": 0.02,
            "last_val": 0.05,
            "last5_val_mean": 0.05,
            "best_val": 0.05,
            "best_to_last_drop": 0.0,
            "score": 0.07,
            "query_first_hop_home_rate": 0.1,
            "delivery_rate": 1.0,
            "home_to_out_rate": 0.8,
            "regime": "t1",
            "run": "run-c",
        },
        {
            "arm": "visit_taskgrad_half",
            "arm_label": "VT-0.5",
            "lr_multiplier": 1.0,
            "composite": 0.06,
            "dense_mean": 0.03,
            "last_dense_mean": 0.03,
            "last_val": 0.07,
            "last5_val_mean": 0.07,
            "best_val": 0.07,
            "best_to_last_drop": 0.0,
            "score": 0.10,
            "query_first_hop_home_rate": 0.1,
            "delivery_rate": 1.0,
            "home_to_out_rate": 0.8,
            "regime": "t1",
            "run": "run-d",
        },
    ]
    picked = module.rank_candidates(records)
    assert picked["visitonly"]["lr_multiplier"] == 1.0
    assert picked["visit_taskgrad_half"]["lr_multiplier"] == 1.0


def test_v62_build_config_is_reproducible():
    module = load_module("gen_v62_configs_again", "scripts/gen_v62_configs.py")
    assert module.build_config("visitonly", "core", "m") == module.build_config("visitonly", "core", "m")
