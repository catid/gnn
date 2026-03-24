from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import yaml

from apsgnn.config import GrowthConfig, TaskConfig, TrainConfig, selector_decision_for_stage
from apsgnn.growth import CoverageTracker, GrowthStage
from apsgnn.train import training_rollout_steps


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v63_generator_emits_contract_pairs_and_schedule_shapes():
    module = load_module("gen_v63_configs", "scripts/gen_v63_configs.py")
    assert set(module.REGIMES) == {"core", "t1", "t1r", "t2a", "t2b", "t2c", "hmid", "hmix"}
    assert set(module.SCHEDULES) == {"p", "s", "m", "l"}
    assert set(module.CONTRACTS) == {"b", "d", "ds", "dsg"}
    assert len(module.PAIRS) == 8

    for schedule, min_fraction in (("s", 0.50), ("m", 0.60), ("l", 0.65)):
        cfg = module.build_config("visit_taskgrad_half_dsg", "hmix", schedule)
        stage_steps = cfg["growth"]["stage_steps"]
        train_steps = cfg["train"]["train_steps"]
        assert sum(stage_steps) == train_steps
        assert stage_steps[-1] / train_steps >= min_fraction


def test_v63_contract_fields_match_expected_semantics():
    module = load_module("gen_v63_contracts", "scripts/gen_v63_configs.py")
    b_cfg = module.build_config("visitonly_b", "core", "m")
    d_cfg = module.build_config("visitonly_d", "core", "m")
    ds_cfg = module.build_config("visitonly_ds", "core", "m")
    dsg_cfg = module.build_config("visitonly_dsg", "core", "m")

    assert b_cfg["train"]["contract_detach_temporal_state"] is False
    assert d_cfg["train"]["contract_detach_temporal_state"] is True
    assert ds_cfg["train"]["contract_penultimate_keep_prob"] == 0.10
    assert dsg_cfg["train"]["contract_shallow_train_fraction"] == 0.80
    assert dsg_cfg["train"]["contract_shallow_rollout_steps"] == 8


def test_v63_static_selector_scores_are_unchanged():
    task = TaskConfig(writers_per_episode=3, start_node_pool_size=2, query_ttl_min=2, query_ttl_max=3)

    static_v = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.0)
    static_vt = GrowthConfig(utility_visit_weight=1.0, utility_grad_weight=0.5)

    assert selector_decision_for_stage(static_v, 3, task=task)["selected_selector"] == "V"
    assert selector_decision_for_stage(static_vt, 3, task=task)["selected_selector"] == "VT-0.5"


def test_v63_bootstrap_exclusion_remains_intact():
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


def test_v63_rollout_curriculum_only_shortens_early_training():
    config = type("Config", (), {})()
    config.task = TaskConfig(max_rollout_steps=12)
    config.train = TrainConfig(train_steps=1000, contract_shallow_train_fraction=0.8, contract_shallow_rollout_steps=8)

    assert training_rollout_steps(1, config) == 8
    assert training_rollout_steps(800, config) == 8
    assert training_rollout_steps(801, config) == 12
    assert training_rollout_steps(1000, config) == 12


def test_v63_selector_logging_and_predictiveness_fields_exist():
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


def test_v63_screening_promotion_keeps_multiple_contracts_when_close():
    module = load_module("build_v63_report", "scripts/build_v63_report.py")
    rankings = [
        {"pair": "visitonly_b", "contract": "b", "summary": {"composite": {"mean": 0.100, "std": 0.010}}},
        {"pair": "visit_taskgrad_half_b", "contract": "b", "summary": {"composite": {"mean": 0.099, "std": 0.010}}},
        {"pair": "visitonly_d", "contract": "d", "summary": {"composite": {"mean": 0.098, "std": 0.010}}},
        {"pair": "visit_taskgrad_half_d", "contract": "d", "summary": {"composite": {"mean": 0.097, "std": 0.010}}},
        {"pair": "visitonly_ds", "contract": "ds", "summary": {"composite": {"mean": 0.0965, "std": 0.010}}},
    ]
    promoted = module.promote_screening_pairs(rankings)
    assert len(promoted) == 4
    assert any(pair.endswith("_d") for pair in promoted) or any(pair.endswith("_ds") for pair in promoted)


def test_v63_eval_completion_filter_and_budget_matching(tmp_path):
    module = load_module("run_v63_eval_sweep", "scripts/run_v63_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_text("", encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1000}) + "\n", encoding="utf-8")
    (run_dir / "config.yaml").write_text(yaml.safe_dump({"train": {"train_steps": 1134}, "task": {"max_rollout_steps": 12}}), encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "s") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 1134}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "s") is True


def test_v63_extra_depth_metrics_map_to_settling_fields(tmp_path):
    module = load_module("build_v63_report_extra", "scripts/build_v63_report.py")
    run_dir = tmp_path / "v63-core-visitonly_b-32-m-s1234"
    run_dir.mkdir()
    payload = {
        "metrics": {
            "query_accuracy": 0.25,
            "query_delivery_rate": 0.75,
            "average_hops": 9.0,
        }
    }
    (run_dir / "eval_best_settle_k6.json").write_text(json.dumps(payload), encoding="utf-8")
    metrics = module.extract_eval_metrics(run_dir, "best", 6, depth_tag="settle")
    assert metrics["query_accuracy"] == 0.25
    assert metrics["query_delivery_rate"] == 0.75
    assert metrics["average_hops"] == 9.0


def test_v63_build_config_is_reproducible():
    module = load_module("gen_v63_configs_again", "scripts/gen_v63_configs.py")
    assert module.build_config("visitonly_b", "core", "m") == module.build_config("visitonly_b", "core", "m")
