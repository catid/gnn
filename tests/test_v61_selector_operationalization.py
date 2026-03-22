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


def test_v61_generator_emits_holdouts_and_matching_schedules():
    module = load_module("gen_v61_configs", "scripts/gen_v61_configs.py")
    assert {"core", "t1", "t1r", "t2a", "t2b", "t2c", "hmid", "hmix"} == set(module.REGIMES)
    assert {"p", "s", "m", "l"} == set(module.SCHEDULES)
    assert module.REGIMES["hmid"]["writers_per_episode"] == 3
    assert module.REGIMES["hmix"]["writers_per_episode"] == 3
    assert module.REGIMES["hmid"]["start_node_pool_size"] == 2
    assert module.REGIMES["hmix"]["start_node_pool_size"] == 1
    assert module.REGIMES["hmid"]["train_eval_writers"] == [3, 7, 11]
    assert module.REGIMES["hmix"]["train_eval_writers"] == [3, 7, 11]

    for arm in ("visitonly", "visit_taskgrad_half", "gate_meta_a", "gate_online_b"):
        cfg = module.build_config(arm, "hmix", "l")
        assert cfg["train"]["train_steps"] == 3360
        assert sum(cfg["growth"]["stage_steps"]) == 3360
        assert cfg["growth"]["stage_steps"][-1] / 3360 >= 0.65
        assert cfg["task"]["writers_per_episode"] == 3


def test_v61_gate_logic_for_static_and_gate_arms():
    task_core = TaskConfig(writers_per_episode=2, start_node_pool_size=2, query_ttl_min=2, query_ttl_max=3)
    task_t1 = TaskConfig(writers_per_episode=4, start_node_pool_size=2, query_ttl_min=2, query_ttl_max=3)
    task_t2a = TaskConfig(writers_per_episode=4, start_node_pool_size=1, query_ttl_min=2, query_ttl_max=3)
    task_t2b = TaskConfig(writers_per_episode=4, start_node_pool_size=2, query_ttl_min=2, query_ttl_max=2)

    static_v = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.0)
    assert selector_decision_for_stage(static_v, 3, task=task_core)["selected_selector_label"] == "V"

    static_vt = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.5)
    assert selector_decision_for_stage(static_vt, 3, task=task_core)["selected_selector_label"] == "VT-0.5"

    gate_writers = GrowthConfig(selector_gate_kind="writers", selector_gate_writers_threshold=2)
    assert selector_decision_for_stage(gate_writers, 3, task=task_core)["selected_selector_label"] == "VT-0.5"
    assert selector_decision_for_stage(gate_writers, 3, task=task_t1)["selected_selector_label"] == "V"

    gate_ingress = GrowthConfig(
        selector_gate_kind="ingress",
        selector_gate_ingress_start_node_pool_threshold=1,
        selector_gate_ingress_allow_tight_ttl=False,
    )
    assert selector_decision_for_stage(gate_ingress, 3, task=task_t2a)["selected_selector_label"] == "VT-0.5"
    assert selector_decision_for_stage(gate_ingress, 3, task=task_t1)["selected_selector_label"] == "V"

    gate_meta = GrowthConfig(
        selector_gate_kind="meta",
        selector_gate_meta_writer_weight=1.0,
        selector_gate_meta_ingress_bonus=-2.5,
        selector_gate_meta_tight_ttl_bonus=1.0,
        selector_gate_meta_threshold=2.5,
    )
    assert selector_decision_for_stage(gate_meta, 3, task=task_core)["selected_selector_label"] == "VT-0.5"
    assert selector_decision_for_stage(gate_meta, 3, task=task_t2a)["selected_selector_label"] == "VT-0.5"
    assert selector_decision_for_stage(gate_meta, 3, task=task_t2b)["selected_selector_label"] == "V"

    gate_online = GrowthConfig(
        selector_gate_kind="online",
        selector_gate_online_stage_index_min=4,
        selector_gate_online_entropy_high_threshold=0.68,
        selector_gate_online_gini_high_threshold=0.75,
    )
    assert selector_decision_for_stage(
        gate_online,
        3,
        task=task_t1,
        current_snapshot={"task_visit_entropy": 0.80, "task_visit_gini": 0.40},
    )["selected_selector_label"] == "V"
    assert selector_decision_for_stage(
        gate_online,
        4,
        task=task_core,
        current_snapshot={"task_visit_entropy": 0.70, "task_visit_gini": 0.61},
    )["selected_selector_label"] == "VT-0.5"
    assert selector_decision_for_stage(
        gate_online,
        4,
        task=task_t1,
        current_snapshot={"task_visit_entropy": 0.56, "task_visit_gini": 0.72},
    )["selected_selector_label"] == "V"


def test_v61_bootstrap_exclusion_remains_intact():
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


def test_v61_selector_logging_and_predictiveness_fields_exist():
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
            "utility_grad_weight": 0.0,
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
    stage_summary = tracker.completed_stages[0]
    split_stats = stage_summary["split_stats"]
    assert "selected_parent_child_usefulness_mean" in split_stats
    assert "unselected_parent_child_usefulness_mean" in split_stats
    assert "utility_usefulness_correlation" in split_stats
    assert "selected_parent_scores" in split_stats
    assert "unselected_parent_scores" in split_stats


def test_v61_eval_completion_filter(tmp_path):
    module = load_module("run_v61_eval_sweep", "scripts/run_v61_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_text("", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 1260}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "s") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1260}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "s") is True


def test_v61_report_builder_ranks_pilot_and_best_gate():
    module = load_module("build_v61_report", "scripts/build_v61_report.py")
    pilot_records = [
        {"family": "visitonly", "arm": "visitonly", "arm_label": "V", "lr_multiplier": 0.8, "composite": 0.10, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "a", "regime": "core", "seed": 1, "tag": "lr08"},
        {"family": "visitonly", "arm": "visitonly", "arm_label": "V", "lr_multiplier": 1.0, "composite": 0.20, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "b", "regime": "t1", "seed": 1, "tag": "lr10"},
        {"family": "g_meta", "arm": "gate_meta_a", "arm_label": "G_meta(A)", "lr_multiplier": 1.0, "composite": 0.30, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "c", "regime": "core", "seed": 1, "tag": "lr10"},
        {"family": "g_meta", "arm": "gate_meta_b", "arm_label": "G_meta(B)", "lr_multiplier": 1.0, "composite": 0.25, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "d", "regime": "t1", "seed": 1, "tag": "lr10"},
    ]
    ranked = module.rank_candidates(pilot_records)
    assert ranked["visitonly"]["best"]["lr_multiplier"] == 1.0
    assert ranked["g_meta"]["best"]["arm"] == "gate_meta_a"

    gate_records = [
        {"family": "g_writers", "arm": "gate_writers_le2", "category": "gate", "composite": 0.12, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "e", "regime": "core", "seed": 1, "tag": "", "lr_multiplier": 1.0},
        {"family": "g_meta", "arm": "gate_meta_a", "category": "gate", "composite": 0.18, "best_val": 0.0, "last_val": 0.0, "last5_val_mean": 0.0, "best_to_last_drop": 0.0, "dense_mean": 0.0, "last_dense_mean": 0.0, "score": 0.0, "run": "f", "regime": "core", "seed": 1, "tag": "", "lr_multiplier": 1.0},
    ]
    best_gate = module.choose_best_gate(gate_records)
    assert best_gate["family"] == "g_meta"


def test_v61_build_config_is_reproducible():
    module = load_module("gen_v61_configs_again", "scripts/gen_v61_configs.py")
    assert module.build_config("gate_online_a", "core", "s") == module.build_config("gate_online_a", "core", "s")
