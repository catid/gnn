from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.probes import bucketed_accuracy, fit_linear_probe, hard_slice_summary
from apsgnn.tasks import GrowthMemoryRoutingTask, delay_targets_from_key


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v66_generator_emits_forensic_pack_definitions():
    module = load_module("gen_v66_configs", "scripts/gen_v66_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.DELAY_REGIMES) == {"d1", "d2", "rd1", "rd2"}
    packs = module.pack_definitions()
    assert "collision_pack" in packs
    assert "delay_pack" in packs
    assert "optional_followup_pack" in packs


def test_v66_collision_configs_toggle_cache_recent_topk_and_class_slice_cleanly():
    module = load_module("gen_v66_collision_cfg", "scripts/gen_v66_configs.py")
    cache_off = module.build_collision_config("visit_taskgrad_half_d", "c2", "m", condition="nocache")
    recent1 = module.build_collision_config("visit_taskgrad_half_d", "c2", "m", condition="recent1")
    topk1 = module.build_collision_config("visit_taskgrad_half_d", "c2", "m", condition="topk1")
    class_off = module.build_collision_config("visit_taskgrad_half_d", "c2", "m", condition="classoff")
    assert cache_off["model"]["enable_cache"] is False
    assert recent1["model"]["cache_visible_recent_limit"] == 1
    assert topk1["model"]["cache_retrieval_topk"] == 1
    assert class_off["model"]["use_reserved_class_slice"] is False


def test_v66_delay_targets_from_key_are_deterministic_and_key_conditioned():
    keys = torch.tensor(
        [
            [1.0, -1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
        ],
        dtype=torch.float32,
    )
    delays = delay_targets_from_key(keys, min_delay=2, max_delay=4, hash_bits=4)
    assert int(delays.min().item()) >= 2
    assert int(delays.max().item()) <= 4
    assert int(delays[0].item()) == int(delays[2].item())
    assert int(delays[0].item()) != int(delays[1].item())


def test_v66_redesigned_delay_batch_is_seed_reproducible_and_uses_exact_wait_mode():
    config = ExperimentConfig()
    config.task.name = "memory_growth"
    config.task.delay_mode = "key_hash_exact_wait"
    config.task.required_delay_min = 1
    config.task.required_delay_max = 3
    config.task.required_delay_hash_bits = 3
    config.task.writers_per_episode = 4
    config.task.home_node_pool_size = 8
    config.model.nodes_total = 33
    task = GrowthMemoryRoutingTask(config)
    batch_a = task.generate(batch_size=6, seed=1234, writers_per_episode=4, active_compute_nodes=32)
    batch_b = task.generate(batch_size=6, seed=1234, writers_per_episode=4, active_compute_nodes=32)
    assert batch_a.query_required_delay is not None
    assert torch.equal(batch_a.query_required_delay, batch_b.query_required_delay)
    expected = delay_targets_from_key(
        batch_a.query_keys,
        min_delay=1,
        max_delay=3,
        hash_bits=3,
    )
    assert torch.equal(batch_a.query_required_delay, expected)


def test_v66_delay_configs_cover_zero_random_fixed_and_required_controls():
    module = load_module("gen_v66_delay_cfg", "scripts/gen_v66_configs.py")
    zero = module.build_delay_config("visitonly_d", "rd2", "m", condition="zero")
    random_cfg = module.build_delay_config("visitonly_d", "rd2", "m", condition="random")
    fixed = module.build_delay_config("visitonly_d", "rd2", "m", condition="fixed")
    required = module.build_delay_config("visitonly_d", "rd2", "m", condition="required")
    assert zero["train"]["delay_override_mode"] == "zero"
    assert random_cfg["train"]["delay_override_mode"] == "random"
    assert fixed["train"]["delay_override_mode"] == "fixed"
    assert fixed["train"]["delay_override_value"] == module.DELAY_REGIMES["rd2"]["fixed_good_delay"]
    assert required["train"]["delay_override_mode"] == "required"


def test_v66_probe_pipeline_learns_simple_linear_signal():
    train_x = torch.eye(4, dtype=torch.float32)
    train_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    result = fit_linear_probe(
        train_x=train_x,
        train_y=train_y,
        valid_x=train_x,
        valid_y=train_y,
        test_x=train_x,
        test_y=train_y,
        num_classes=4,
        steps=150,
        lr=0.1,
    )
    assert result.valid_accuracy >= 0.99
    assert result.test_accuracy >= 0.99


def test_v66_hard_slice_helpers_bucket_and_split_rows():
    rows = [
        {"competing_entries": 0, "ambiguity": 0.1, "correct": 1.0},
        {"competing_entries": 1, "ambiguity": 0.2, "correct": 1.0},
        {"competing_entries": 3, "ambiguity": 0.8, "correct": 0.0},
        {"competing_entries": 4, "ambiguity": 0.9, "correct": 0.0},
    ]
    summary = hard_slice_summary(
        rows,
        difficulty_key="competing_entries",
        ambiguity_key="ambiguity",
        correct_key="correct",
        hard_difficulty_threshold=2,
        hard_ambiguity_threshold=0.5,
    )
    buckets = bucketed_accuracy(rows, bucket_key="competing_entries", correct_key="correct")
    assert summary["base_accuracy"] > summary["hard_accuracy"]
    assert any(entry["bucket"] == 3.0 for entry in buckets)


def test_v66_report_parser_accepts_v66_run_names():
    module = load_module("build_v66_report", "scripts/build_v66_report.py")
    parsed = module.parse_run_name(
        "20260326-220000-v66-collision-c2-recent1-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["pack"] == "collision"
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "recent1"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"


def test_v66_delay_semantic_audit_marks_old_task_broader_than_redesigned_task():
    module = load_module("build_v66_report_delay_audit", "scripts/build_v66_report.py")
    old_audit = module.delay_semantic_audit("d2", samples=256)
    redesigned_audit = module.delay_semantic_audit("rd2", samples=256)
    assert old_audit["acceptable_delay_count_mean"] > redesigned_audit["acceptable_delay_count_mean"]
    assert math.isclose(redesigned_audit["oracle_required_success_rate"], 1.0, rel_tol=0.0, abs_tol=1.0e-6)
