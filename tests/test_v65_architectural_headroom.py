from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.probes import bucketed_accuracy, fit_linear_probe, hard_slice_summary
from apsgnn.tasks import GrowthMemoryRoutingTask


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v65_generator_emits_collision_delay_and_pack_definitions():
    module = load_module("gen_v65_configs", "scripts/gen_v65_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c0", "c1", "c2"}
    assert set(module.DELAY_REGIMES) == {"d0", "d1", "d2"}
    assert module.COLLISION_REGIMES["c0"]["home_node_pool_size"] == 0
    assert module.COLLISION_REGIMES["c2"]["home_node_pool_size"] < module.COLLISION_REGIMES["c1"]["home_node_pool_size"]
    packs = module.pack_definitions()
    assert "collision_pack" in packs
    assert "class_slice_pack" in packs
    assert "delay_pack" in packs


def test_v65_collision_configs_toggle_cache_and_class_slice_cleanly():
    module = load_module("gen_v65_collision_cfg", "scripts/gen_v65_configs.py")
    cache_off = module.build_collision_config(
        "visit_taskgrad_half_d",
        "c2",
        "m",
        cache_enabled=False,
        class_slice_enabled=True,
    )
    class_off = module.build_collision_config(
        "visit_taskgrad_half_d",
        "c2",
        "m",
        cache_enabled=True,
        class_slice_enabled=False,
    )
    assert cache_off["model"]["enable_cache"] is False
    assert class_off["model"]["use_reserved_class_slice"] is False
    assert cache_off["task"]["home_node_pool_size"] == 2


def test_v65_delay_batch_generation_is_seed_reproducible_and_respects_required_delay_range():
    config = ExperimentConfig()
    config.task.name = "memory_growth"
    config.task.delay_mode = "required_wait"
    config.task.required_delay_min = 2
    config.task.required_delay_max = 4
    config.task.writers_per_episode = 4
    config.task.home_node_pool_size = 8
    config.model.nodes_total = 33
    task = GrowthMemoryRoutingTask(config)
    batch_a = task.generate(batch_size=3, seed=1234, writers_per_episode=4, active_compute_nodes=32)
    batch_b = task.generate(batch_size=3, seed=1234, writers_per_episode=4, active_compute_nodes=32)
    assert batch_a.query_required_delay is not None
    assert torch.equal(batch_a.query_required_delay, batch_b.query_required_delay)
    assert int(batch_a.query_required_delay.min().item()) >= 2
    assert int(batch_a.query_required_delay.max().item()) <= 4


def test_v65_delay_configs_cover_learned_and_control_conditions():
    module = load_module("gen_v65_delay_cfg", "scripts/gen_v65_configs.py")
    learned = module.build_delay_config("visitonly_d", "d1", "m", condition="learned")
    zero = module.build_delay_config("visitonly_d", "d1", "m", condition="zero")
    fixed = module.build_delay_config("visitonly_d", "d2", "m", condition="fixed")
    assert learned["train"]["delay_override_mode"] == "learned"
    assert zero["train"]["delay_override_mode"] == "zero"
    assert fixed["train"]["delay_override_mode"] == "fixed"
    assert fixed["train"]["delay_override_value"] == module.DELAY_REGIMES["d2"]["fixed_good_delay"]


def test_v65_probe_pipeline_learns_simple_linear_signal():
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


def test_v65_hard_slice_helpers_bucket_and_split_rows():
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


def test_v65_report_parser_accepts_runs_without_seed_suffix():
    module = load_module("build_v65_report", "scripts/build_v65_report.py")
    parsed = module.parse_run_name(
        "20260326-024230-v65-collision-c2-cacheon-classon-visit_taskgrad_half_d-32-m"
    )
    assert parsed is not None
    assert parsed["pack"] == "collision"
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "cacheon-classon"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == ""
