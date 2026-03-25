from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import torch
import yaml

from apsgnn.config import TaskConfig, TrainConfig
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


def test_v64_generator_emits_raw_and_logical_contracts_with_schedule_shapes():
    module = load_module("gen_v64_configs", "scripts/gen_v64_configs.py")
    contracts = module.build_contracts(core_best="ds_p010", core_runner_up="ds_p025", aux_final_multiplier=0.25)
    assert set(module.RAW_CORE_CONTRACTS) == {
        "d",
        "ds_p005",
        "ds_p010",
        "ds_p025",
        "ds_p040",
        "ds_fixed1step",
        "ds_fixed2step",
    }
    for contract in ("ds_core_best", "ds_core_runner_up", "ds_auxanneal", "ds_randdepth"):
        assert contract in contracts
    for schedule, min_fraction in (("s", 0.50), ("m", 0.60), ("l", 0.65)):
        cfg = module.build_config(
            "visit_taskgrad_half_ds_randdepth",
            "hmix",
            schedule,
            core_best="ds_p010",
            core_runner_up="ds_p025",
            aux_final_multiplier=0.25,
        )
        stage_steps = cfg["growth"]["stage_steps"]
        train_steps = cfg["train"]["train_steps"]
        assert sum(stage_steps) == train_steps
        assert stage_steps[-1] / train_steps >= min_fraction


def test_v64_aux_anneal_and_randdepth_contract_fields_match_expected_semantics():
    module = load_module("gen_v64_configs_fields", "scripts/gen_v64_configs.py")
    anneal_cfg = module.build_config(
        "visitonly_ds_auxanneal",
        "core",
        "m",
        core_best="ds_p025",
        core_runner_up="ds_p010",
        aux_final_multiplier=0.25,
    )
    rand_cfg = module.build_config(
        "visitonly_ds_randdepth",
        "core",
        "m",
        core_best="ds_p025",
        core_runner_up="ds_p010",
        aux_final_multiplier=0.25,
    )
    assert anneal_cfg["train"]["contract_aux_anneal_start_fraction"] == 0.60
    assert anneal_cfg["train"]["contract_aux_anneal_final_multiplier"] == 0.25
    assert rand_cfg["train"]["contract_rand_depth_train_fraction"] == 0.80
    assert rand_cfg["train"]["contract_rand_depth_multipliers"] == [0.75, 1.0, 1.25]


def test_v64_randdepth_rollout_schedule_is_deterministic_and_restores_full_depth():
    config = type("Config", (), {})()
    config.task = TaskConfig(max_rollout_steps=12)
    config.train = TrainConfig(
        seed=1234,
        train_steps=1000,
        contract_rand_depth_train_fraction=0.8,
        contract_rand_depth_multipliers=[0.75, 1.0, 1.25],
    )
    first = training_rollout_steps(1, config)
    second = training_rollout_steps(1, config)
    assert first == second
    assert first in {9, 12, 15}
    assert training_rollout_steps(801, config) == 12
    assert training_rollout_steps(1000, config) == 12


def test_v64_bootstrap_exclusion_remains_intact():
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


def test_v64_report_builder_selects_shared_aux_anneal_setting():
    module = load_module("build_v64_report", "scripts/build_v64_report.py")
    pilot_records = [
        {"contract": "ds_auxanneal_050", "aux_anneal_final_multiplier": 0.50, "composite": 0.08, "best_val": 0.08, "last_val": 0.08, "last5_val_mean": 0.08, "best_to_last_drop": 0.0},
        {"contract": "ds_auxanneal_050", "aux_anneal_final_multiplier": 0.50, "composite": 0.07, "best_val": 0.07, "last_val": 0.07, "last5_val_mean": 0.07, "best_to_last_drop": 0.0},
        {"contract": "ds_auxanneal_025", "aux_anneal_final_multiplier": 0.25, "composite": 0.09, "best_val": 0.09, "last_val": 0.09, "last5_val_mean": 0.09, "best_to_last_drop": 0.0},
        {"contract": "ds_auxanneal_025", "aux_anneal_final_multiplier": 0.25, "composite": 0.10, "best_val": 0.10, "last_val": 0.10, "last5_val_mean": 0.10, "best_to_last_drop": 0.0},
    ]
    assert module.choose_aux_anneal_final_multiplier(pilot_records) == 0.25


def test_v64_logical_pair_key_maps_aux_variants_to_single_family():
    module = load_module("build_v64_report_pair_key", "scripts/build_v64_report.py")
    assert module.logical_pair_key("visitonly_ds_auxanneal_050") == "visitonly_ds_auxanneal"
    assert module.logical_pair_key("visit_taskgrad_half_ds_auxanneal_025") == "visit_taskgrad_half_ds_auxanneal"
    assert module.logical_pair_key("visitonly_ds_randdepth") == "visitonly_ds_randdepth"


def test_v64_eval_completion_filter_matches_budget(tmp_path):
    module = load_module("run_v64_eval_sweep", "scripts/run_v64_eval_sweep.py")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "last.pt").write_text("", encoding="utf-8")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"train": {"train_steps": 2268}, "task": {"max_rollout_steps": 12}}),
        encoding="utf-8",
    )
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 2200}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "m") is False
    (run_dir / "metrics.jsonl").write_text(json.dumps({"step": 2268}) + "\n", encoding="utf-8")
    assert module.is_complete_substantive_run(run_dir, "m") is True


def test_v64_build_config_is_reproducible():
    module = load_module("gen_v64_configs_again", "scripts/gen_v64_configs.py")
    cfg_a = module.build_config(
        "visitonly_ds_core_best",
        "core",
        "m",
        core_best="ds_p010",
        core_runner_up="ds_p025",
        aux_final_multiplier=0.50,
    )
    cfg_b = module.build_config(
        "visitonly_ds_core_best",
        "core",
        "m",
        core_best="ds_p010",
        core_runner_up="ds_p025",
        aux_final_multiplier=0.50,
    )
    assert cfg_a == cfg_b
