#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import json
import math
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from apsgnn.config import ExperimentConfig
from apsgnn.probes import bucketed_accuracy, fit_linear_probe, hard_slice_summary
from apsgnn.tasks import GrowthMemoryRoutingTask


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v66.json"
REPORT_PATH = REPORTS / "final_report_v66_forensic_headroom.md"
PACK_DEFS_PATH = REPORTS / "v66_pack_definitions.json"
V65_SUMMARY_PATH = REPORTS / "summary_metrics_v65.json"
BASE_LR = 2.0e-4
EXPECTED_TRAIN_STEPS = {"p": 300, "m": 1350, "l": 2160}
RUN_RE = re.compile(
    r"v66-(?P<pack>collision|delay|followup)-(?P<regime>[^-]+)-(?P<condition>[^-]+)-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|m|l)(?:-(?P<tag>(?!s\d+$)[^-]+))?(?:-s(?P<seed>\d+))?$"
)
PAIR_LABELS = {
    "visitonly_d": "V/D",
    "visit_taskgrad_half_d": "VT-0.5/D",
}


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = load_module("gen_v66_configs", "scripts/gen_v66_configs.py")


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": std(values)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def visible_gpu_count() -> int:
    result = subprocess.run(
        ["bash", "-lc", "nvidia-smi --list-gpus | wc -l"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(int(result.stdout.strip()), 0)
    except ValueError:
        return 0


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def latest_runs() -> list[tuple[dict[str, str], Path]]:
    pilot_runs: list[tuple[dict[str, str], Path]] = []
    latest: dict[tuple[str, str, str, str, str, str], tuple[dict[str, str], Path]] = {}
    for candidate in sorted(RUNS.glob("*-v66-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["schedule"] == "p":
            pilot_runs.append((meta, candidate))
            continue
        latest[(meta["pack"], meta["regime"], meta["condition"], meta["pair"], meta["schedule"], meta["seed"])] = (
            meta,
            candidate,
        )
    return pilot_runs + list(latest.values())


def is_complete_run(run_dir: Path, schedule: str) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    last_path = run_dir / "last.pt"
    if not metrics_path.exists() or not config_path.exists() or not last_path.exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != EXPECTED_TRAIN_STEPS[schedule]:
        return False
    max_step = 0
    for row in read_jsonl(metrics_path):
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= EXPECTED_TRAIN_STEPS[schedule]


def eval_metric(run_dir: Path, kind: str, writers: int, *, bypass_mode: str = "none") -> dict[str, float]:
    suffix = ""
    if bypass_mode != "none":
        suffix = f"_bypass-{bypass_mode}"
    path = run_dir / f"eval_{kind}_k{writers}{suffix}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def score_record(record: dict[str, Any]) -> float:
    return 0.45 * record["dense_mean"] + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    writers = list(config["task"].get("train_eval_writers", [config["task"]["writers_per_episode"]]))
    best_metrics = [eval_metric(run_dir, "best", writers_per_episode) for writers_per_episode in writers]
    last_metrics = [eval_metric(run_dir, "last", writers_per_episode) for writers_per_episode in writers]
    dense_mean = mean([item.get("query_accuracy", 0.0) for item in best_metrics])
    last_dense_mean = mean([item.get("query_accuracy", 0.0) for item in last_metrics])
    merged_metric = best_metrics[0] if best_metrics else {}
    return {
        "run": run_dir.name,
        "pack": meta["pack"],
        "regime": meta["regime"],
        "condition": meta["condition"],
        "pair": meta["pair"],
        "pair_label": PAIR_LABELS.get(meta["pair"], meta["pair"]),
        "schedule": meta["schedule"],
        "seed": int(meta["seed"] or config["train"]["seed"]),
        "tag": meta["tag"],
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "dense_mean": dense_mean,
        "last_dense_mean": last_dense_mean,
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "cache_enabled": bool(config["model"].get("enable_cache", True)),
        "class_slice_enabled": bool(config["model"].get("use_reserved_class_slice", True)),
        "cache_visible_recent_limit": int(config["model"].get("cache_visible_recent_limit", 0)),
        "cache_retrieval_topk": int(config["model"].get("cache_retrieval_topk", 0)),
        "cache_bypass_mode": str(config["model"].get("cache_read_bypass_mode", "none")),
        "delay_override_mode": str(config["train"].get("delay_override_mode", "learned")),
        "delay_mode": str(config["task"].get("delay_mode", "none")),
        "required_delay_min": int(config["task"].get("required_delay_min", 0)),
        "required_delay_max": int(config["task"].get("required_delay_max", 0)),
        "query_delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
        "retrieval_top_mass": float(merged_metric.get("retrieval_top_mass", 0.0)),
        "retrieval_entropy": float(merged_metric.get("retrieval_entropy", 0.0)),
        "retrieval_cache_entries": float(merged_metric.get("retrieval_cache_entries", 0.0)),
        "retrieval_target_entry_hit_rate": float(merged_metric.get("retrieval_target_entry_hit_rate", 0.0)),
        "query_first_delay_mean": float(merged_metric.get("query_first_delay_mean", 0.0)),
        "query_first_delay_nonzero_rate": float(merged_metric.get("query_first_delay_nonzero_rate", 0.0)),
        "query_first_delay_match_rate": float(merged_metric.get("query_first_delay_match_rate", 0.0)),
        "pilot_score": 0.0,
    } | {"pilot_score": 0.0}


def select_best_lr(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["schedule"] != "p" or not record["tag"]:
            continue
        family = f"{record['pack']}|{record['regime']}|{record['condition']}|{record['pair']}"
        if family not in best or record["pilot_score"] > best[family]["pilot_score"]:
            best[family] = {
                "lr_multiplier": record["lr_multiplier"],
                "pilot_score": record["pilot_score"],
            }
    return best


def group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "count": len(records),
        "dense_mean": mean_std([record["dense_mean"] for record in records]),
        "last_dense_mean": mean_std([record["last_dense_mean"] for record in records]),
        "last_val": mean_std([record["last_val"] for record in records]),
        "last5_val_mean": mean_std([record["last5_val_mean"] for record in records]),
        "pilot_score": mean_std([record["pilot_score"] for record in records]),
        "query_delivery_rate": mean_std([record["query_delivery_rate"] for record in records]),
        "query_first_hop_home_rate": mean_std([record["query_first_hop_home_rate"] for record in records]),
        "home_to_out_rate": mean_std([record["home_to_out_rate"] for record in records]),
        "retrieval_top_mass": mean_std([record["retrieval_top_mass"] for record in records]),
        "retrieval_entropy": mean_std([record["retrieval_entropy"] for record in records]),
        "retrieval_cache_entries": mean_std([record["retrieval_cache_entries"] for record in records]),
        "retrieval_target_entry_hit_rate": mean_std(
            [record["retrieval_target_entry_hit_rate"] for record in records]
        ),
        "query_first_delay_mean": mean_std([record["query_first_delay_mean"] for record in records]),
        "query_first_delay_nonzero_rate": mean_std(
            [record["query_first_delay_nonzero_rate"] for record in records]
        ),
        "query_first_delay_match_rate": mean_std(
            [record["query_first_delay_match_rate"] for record in records]
        ),
    }


def probe_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu")
    count = int(payload["labels"].size(0))
    rows: list[dict[str, float]] = []
    chosen_delay = payload.get("first_hop_delay", torch.zeros(count))
    required_delay = payload.get("required_delay", torch.zeros(count))
    for index in range(count):
        rows.append(
            {
                "correct": float(payload.get("correct", torch.zeros(count))[index].item()),
                "delivered": float(payload.get("delivered", torch.zeros(count, dtype=torch.bool))[index].item()),
                "competing_entries": float(payload.get("home_competing_entries", torch.zeros(count))[index].item()),
                "ambiguity": float(1.0 - payload.get("home_retrieval_top_mass", torch.zeros(count))[index].item()),
                "required_delay": float(required_delay[index].item()),
                "chosen_delay": float(chosen_delay[index].item()),
                "delay_gap": float(abs(chosen_delay[index].item() - required_delay[index].item())),
                "delay_ready": float(payload.get("home_delay_ready", torch.zeros(count, dtype=torch.bool))[index].item()),
                "target_entry_hit": float(payload.get("home_target_entry_hit", torch.zeros(count))[index].item()),
            }
        )
    return rows


def run_probe_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    labels = payload["labels"].to(torch.long)

    def split_and_probe(features: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        mask = mask.to(torch.bool)
        if int(mask.sum().item()) < 12:
            return {}
        x = features[mask].to(torch.float32)
        y = labels[mask]
        n = x.size(0)
        a = max(int(n * 0.6), 1)
        b = max(int(n * 0.8), a + 1)
        result = fit_linear_probe(
            train_x=x[:a],
            train_y=y[:a],
            valid_x=x[a:b],
            valid_y=y[a:b],
            test_x=x[b:],
            test_y=y[b:],
            num_classes=32,
            steps=150,
            lr=0.05,
        )
        return {
            "train_accuracy": result.train_accuracy,
            "valid_accuracy": result.valid_accuracy,
            "test_accuracy": result.test_accuracy,
        }

    delivered = payload.get("delivered", torch.zeros(labels.size(0), dtype=torch.bool))
    home_mask = payload.get("home_entry_count", torch.zeros(labels.size(0))) > 0
    return {
        "final_sink_state": split_and_probe(payload["final_sink_state"], delivered),
        "home_cache_mean_state": split_and_probe(payload["home_cache_mean_state"], home_mask),
        "home_hidden_state": split_and_probe(payload["home_hidden_state"], home_mask),
    }


def delay_semantic_audit(regime: str, *, samples: int = 4096) -> dict[str, Any]:
    regime_cfg = GEN.DELAY_REGIMES[regime]
    config = ExperimentConfig()
    config.model.nodes_total = GEN.MODEL_BASE["nodes_total"]
    config.model.delay_bins = GEN.MODEL_BASE["delay_bins"]
    config.task.name = "memory_growth"
    config.task.writers_per_episode = regime_cfg["writers_per_episode"]
    config.task.start_node_pool_size = regime_cfg["start_node_pool_size"]
    config.task.home_node_pool_size = regime_cfg["home_node_pool_size"]
    config.task.query_ttl_min = regime_cfg["query_ttl_min"]
    config.task.query_ttl_max = regime_cfg["query_ttl_max"]
    config.task.max_rollout_steps = regime_cfg["max_rollout_steps"]
    config.task.delay_mode = regime_cfg["mode"]
    config.task.required_delay_min = regime_cfg["required_delay_min"]
    config.task.required_delay_max = regime_cfg["required_delay_max"]
    config.task.required_delay_hash_bits = regime_cfg["required_delay_hash_bits"]
    task = GrowthMemoryRoutingTask(config)
    required_chunks = []
    remaining = samples
    seed = 9137
    while remaining > 0:
        batch_size = min(128, remaining)
        batch = task.generate(
            batch_size=batch_size,
            seed=seed,
            writers_per_episode=regime_cfg["writers_per_episode"],
            active_compute_nodes=32,
            bootstrap_mode=False,
            topology=None,
        )
        required_chunks.append(batch.query_required_delay.to(torch.long))
        remaining -= batch_size
        seed += 1
    required = torch.cat(required_chunks, dim=0)
    delay_bins = int(config.model.delay_bins)
    fixed_value = int(regime_cfg["fixed_good_delay"])
    if regime_cfg["mode"] == "required_wait":
        acceptable_count = (delay_bins - required).clamp(min=0)
        zero_success = (required <= 0).to(torch.float32)
        fixed_success = (required <= fixed_value).to(torch.float32)
        random_success = acceptable_count.to(torch.float32) / float(delay_bins)
    else:
        acceptable_count = torch.ones_like(required)
        zero_success = (required == 0).to(torch.float32)
        fixed_success = (required == fixed_value).to(torch.float32)
        random_success = torch.full_like(required, 1.0 / float(delay_bins), dtype=torch.float32)
    required_success = torch.ones_like(required, dtype=torch.float32)
    histogram: dict[str, int] = {}
    for value in required.tolist():
        histogram[str(int(value))] = histogram.get(str(int(value)), 0) + 1
    return {
        "mode": regime_cfg["mode"],
        "acceptable_delay_count_mean": float(acceptable_count.to(torch.float32).mean().item()),
        "acceptable_delay_count_min": float(acceptable_count.min().item()),
        "acceptable_delay_count_max": float(acceptable_count.max().item()),
        "zero_success_rate": float(zero_success.mean().item()),
        "fixed_success_rate": float(fixed_success.mean().item()),
        "random_expected_success_rate": float(random_success.mean().item()),
        "oracle_required_success_rate": float(required_success.mean().item()),
        "required_delay_histogram": histogram,
    }


def collision_bypass_summary(record: dict[str, Any], *, writers: int) -> dict[str, float]:
    run_dir = RUNS / record["run"]
    normal = eval_metric(run_dir, "best", writers)
    bypass = eval_metric(run_dir, "best", writers, bypass_mode="zero_update")
    return {
        "normal_dense": float(normal.get("query_accuracy", 0.0)),
        "bypass_dense": float(bypass.get("query_accuracy", 0.0)),
        "dense_delta": float(normal.get("query_accuracy", 0.0) - bypass.get("query_accuracy", 0.0)),
    }


def best_record(records: list[dict[str, Any]], *, pair: str, regime: str, condition: str, schedule: str = "m") -> dict[str, Any] | None:
    candidates = [
        record
        for record in records
        if record["pair"] == pair and record["regime"] == regime and record["condition"] == condition and record["schedule"] == schedule
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda record: (record["dense_mean"], record["last_val"]))


def substantive_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [record for record in records if record["schedule"] in {"m", "l"}]


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    for record in records:
        record["pilot_score"] = score_record(record)

    summary: dict[str, Any] = {
        "visible_gpu_count": visible_gpu_count(),
        "budgets": PACK_DEFS_PATH.exists() and read_json(PACK_DEFS_PATH).get("budgets", {}) or {},
        "completed_run_count": len(records),
        "pilot_choices": select_best_lr(records),
        "chosen_contract_family": "d",
        "incumbent_pair": "visit_taskgrad_half_d",
        "control_pair": "visitonly_d",
        "records": substantive_records(records),
    }

    inc = summary["incumbent_pair"]
    ctrl = summary["control_pair"]

    collision_records = [
        record for record in records if record["pack"] == "collision" and record["pair"] == inc and record["schedule"] in {"m", "l"}
    ]
    collision_summary: dict[str, Any] = {"baseline": {}, "interventions": {}, "runner_up_control": {}}
    for regime in ("c1", "c2"):
        cacheon_records = [record for record in collision_records if record["regime"] == regime and record["condition"] == "cacheon"]
        nocache_records = [record for record in collision_records if record["regime"] == regime and record["condition"] == "nocache"]
        collision_summary["baseline"][regime] = {
            "cache_on": group_summary(cacheon_records),
            "cache_off": group_summary(nocache_records),
        }
        collision_summary["baseline"][regime]["dense_gap"] = (
            collision_summary["baseline"][regime]["cache_on"].get("dense_mean", {}).get("mean", 0.0)
            - collision_summary["baseline"][regime]["cache_off"].get("dense_mean", {}).get("mean", 0.0)
        )
        deficit = (
            collision_summary["baseline"][regime]["cache_off"].get("dense_mean", {}).get("mean", 0.0)
            - collision_summary["baseline"][regime]["cache_on"].get("dense_mean", {}).get("mean", 0.0)
        )
        collision_summary["baseline"][regime]["cache_deficit"] = deficit

        for condition in ("recent1", "topk1"):
            key = f"{condition}:{regime}"
            condition_records = [
                record for record in collision_records if record["regime"] == regime and record["condition"] == condition
            ]
            condition_summary = group_summary(condition_records)
            recovery = condition_summary.get("dense_mean", {}).get("mean", 0.0) - collision_summary["baseline"][regime]["cache_on"].get("dense_mean", {}).get("mean", 0.0)
            recovery_fraction = recovery / deficit if deficit > 1.0e-6 else 0.0
            collision_summary["interventions"][key] = {
                "summary": condition_summary,
                "recovery_vs_cacheon": recovery,
                "recovery_fraction": recovery_fraction,
            }

    control_records = [
        record
        for record in records
        if record["pack"] == "collision"
        and record["pair"] == ctrl
        and record["regime"] == "c2"
        and record["schedule"] in {"m", "l"}
        and record["condition"] in {"cacheon", "nocache"}
    ]
    for condition in ("cacheon", "nocache"):
        collision_summary["runner_up_control"][condition] = group_summary(
            [record for record in control_records if record["condition"] == condition]
        )

    best_intervention_name = ""
    best_intervention_score = -1.0e9
    for condition in ("recent1", "topk1"):
        fraction = mean(
            [
                collision_summary["interventions"].get(f"{condition}:{regime}", {}).get("recovery_fraction", 0.0)
                for regime in ("c1", "c2")
            ]
        )
        if fraction > best_intervention_score:
            best_intervention_score = fraction
            best_intervention_name = condition
    collision_summary["best_intervention"] = best_intervention_name
    collision_summary["best_intervention_mean_recovery_fraction"] = best_intervention_score

    bypass_checks: dict[str, Any] = {}
    for regime in ("c1", "c2"):
        record = best_record(records, pair=inc, regime=regime, condition="cacheon")
        if record is None:
            continue
        writers = GEN.COLLISION_REGIMES[regime]["train_eval_writers"][0]
        bypass_checks[f"cacheon:{regime}"] = collision_bypass_summary(record, writers=writers)
        if best_intervention_name:
            best_int_record = best_record(records, pair=inc, regime=regime, condition=best_intervention_name)
            if best_int_record is not None:
                bypass_checks[f"{best_intervention_name}:{regime}"] = collision_bypass_summary(
                    best_int_record,
                    writers=writers,
                )
    collision_summary["bypass_checks"] = bypass_checks

    collision_decodability: dict[str, Any] = {}
    collision_hard_slice: dict[str, Any] = {}
    for regime, condition in (("c2", "cacheon"), ("c2", "nocache"), ("c2", best_intervention_name)):
        if not condition:
            continue
        record = best_record(records, pair=inc, regime=regime, condition=condition)
        if record is None:
            continue
        key = f"{condition}:{regime}"
        probe_path = RUNS / record["run"] / "probe_best.pt"
        collision_decodability[key] = run_probe_audit(probe_path)
        rows = probe_rows(probe_path)
        collision_hard_slice[key] = {
            "summary": hard_slice_summary(
                rows,
                difficulty_key="competing_entries",
                ambiguity_key="ambiguity",
                correct_key="correct",
                hard_difficulty_threshold=2.0,
                hard_ambiguity_threshold=0.25,
            ),
            "bucketed_accuracy": bucketed_accuracy(rows, bucket_key="competing_entries", correct_key="correct"),
        }
    collision_summary["decodability"] = collision_decodability
    collision_summary["hard_slice"] = collision_hard_slice

    decodable_cache_signal = collision_decodability.get("cacheon:c2", {}).get("home_cache_mean_state", {}).get("test_accuracy", 0.0)
    sink_signal = collision_decodability.get("cacheon:c2", {}).get("final_sink_state", {}).get("test_accuracy", 0.0)
    collision_positive = bool(best_intervention_score > 0.15 or decodable_cache_signal > sink_signal + 0.10)
    collision_summary["positive"] = collision_positive
    collision_summary["class_slice_followup_worth_testing"] = bool(best_intervention_score > 0.30 and decodable_cache_signal > 0.20)
    summary["collision_pack"] = collision_summary

    delay_records = [record for record in records if record["pack"] == "delay" and record["pair"] == inc and record["schedule"] in {"m", "l"}]
    delay_summary: dict[str, Any] = {
        "current": {},
        "redesigned": {},
        "timing_oracle_audit": {},
    }
    for regime in ("d1", "d2", "rd1", "rd2"):
        learned_records = [record for record in delay_records if record["regime"] == regime and record["condition"] == "learned"]
        zero_records = [record for record in delay_records if record["regime"] == regime and record["condition"] == "zero"]
        payload = {
            "learned": group_summary(learned_records),
            "forced_zero": group_summary(zero_records),
        }
        payload["dense_gap"] = (
            payload["learned"].get("dense_mean", {}).get("mean", 0.0)
            - payload["forced_zero"].get("dense_mean", {}).get("mean", 0.0)
        )
        if regime.startswith("rd"):
            delay_summary["redesigned"][regime] = payload
        else:
            delay_summary["current"][regime] = payload
        delay_summary["timing_oracle_audit"][regime] = delay_semantic_audit(regime)

    v65_reference = read_json(V65_SUMMARY_PATH) if V65_SUMMARY_PATH.exists() else {}
    delay_summary["v65_reference"] = {
        "delay_pack": v65_reference.get("delay_pack", {}),
        "delay_validation": v65_reference.get("delay_validation", {}),
    }

    delay_decodability: dict[str, Any] = {}
    delay_hard_slice: dict[str, Any] = {}
    for regime, condition in (("d2", "learned"), ("d2", "zero"), ("rd2", "learned"), ("rd2", "zero")):
        record = best_record(records, pair=inc, regime=regime, condition=condition)
        if record is None:
            continue
        key = f"{regime}:{condition}"
        probe_path = RUNS / record["run"] / "probe_best.pt"
        delay_decodability[key] = run_probe_audit(probe_path)
        rows = probe_rows(probe_path)
        delay_hard_slice[key] = {
            "summary": hard_slice_summary(
                rows,
                difficulty_key="required_delay",
                ambiguity_key="delay_gap",
                correct_key="correct",
                hard_difficulty_threshold=2.0,
                hard_ambiguity_threshold=1.0,
            ),
            "bucketed_accuracy": bucketed_accuracy(rows, bucket_key="required_delay", correct_key="correct"),
        }
    delay_summary["decodability"] = delay_decodability
    delay_summary["hard_slice"] = delay_hard_slice

    current_d2 = delay_summary["timing_oracle_audit"].get("d2", {})
    redesigned_rd2 = delay_summary["timing_oracle_audit"].get("rd2", {})
    current_benchmark_needs_redesign = current_d2.get("acceptable_delay_count_mean", 0.0) > 1.5
    redesigned_positive = any(
        delay_summary["redesigned"].get(regime, {}).get("dense_gap", 0.0) > 0.01
        and delay_summary["redesigned"].get(regime, {}).get("learned", {}).get("query_first_delay_nonzero_rate", {}).get("mean", 0.0) > 0.10
        for regime in ("rd1", "rd2")
    )
    delay_summary["current_benchmark_needs_redesign"] = current_benchmark_needs_redesign
    delay_summary["redesign_positive"] = redesigned_positive
    delay_summary["positive"] = bool(current_benchmark_needs_redesign or redesigned_positive)
    delay_summary["followup_worth_testing"] = bool(
        delay_summary["redesigned"].get("rd2", {}).get("dense_gap", 0.0) > 0.03
        and delay_hard_slice.get("rd2:learned", {}).get("summary", {}).get("hard_accuracy", 0.0)
        > delay_hard_slice.get("rd2:zero", {}).get("summary", {}).get("hard_accuracy", 0.0) + 0.02
    )
    summary["delay_pack"] = delay_summary

    summary["decodability_audit"] = {
        "collision": collision_decodability,
        "delay": delay_decodability,
    }

    optional_followup = {
        "triggered": False,
        "path": "",
        "reason": "",
    }
    if collision_summary["class_slice_followup_worth_testing"]:
        optional_followup = {
            "triggered": True,
            "path": "class_slice_removal",
            "reason": "collision intervention recovered enough of the cache-on deficit and cache state remained decodable",
        }
    elif delay_summary["followup_worth_testing"]:
        optional_followup = {
            "triggered": True,
            "path": "delay_extension",
            "reason": "redesigned delay benchmark showed a stable learned-delay advantage and hard-slice movement",
        }
    summary["optional_followup"] = optional_followup

    if collision_summary["positive"] and delay_summary["redesign_positive"]:
        headroom = "still_has_narrow_headroom"
        next_move = "redesigned_delay_work"
    elif collision_summary["positive"]:
        headroom = "forensic_collision_headroom_only"
        next_move = "de_scaffold_further"
    elif delay_summary["redesign_positive"]:
        headroom = "narrow_delay_headroom_only"
        next_move = "redesigned_delay_work"
    else:
        headroom = "low_priority_maintenance"
        next_move = "shift_main_research_effort_elsewhere"
    summary["headroom_conclusion"] = headroom
    summary["next_move"] = next_move
    return summary


def format_table(records: list[dict[str, Any]]) -> list[str]:
    if not records:
        return ["| Pack | Regime | Condition | Pair | Sched | Seed | Dense | Last |", "| --- | --- | --- | --- | --- | ---: | ---: | ---: |"]
    lines = [
        "| Pack | Regime | Condition | Pair | Sched | Seed | Dense | Last |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for record in sorted(
        records,
        key=lambda row: (row["pack"], row["regime"], row["condition"], row["pair"], row["schedule"], row["seed"]),
    ):
        lines.append(
            f"| {record['pack']} | {record['regime']} | {record['condition']} | {record['pair_label']} | {record['schedule']} | {record['seed']} | {record['dense_mean']:.4f} | {record['last_val']:.4f} |"
        )
    return lines


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# APSGNN v66: Forensic Architectural Headroom",
        "",
        "## What Changed From v65",
        "",
        "v66 treats `gnn` as a forensic diagnostic repo instead of the main frontier for broad mechanism search. The budget stays on the incumbent `VT-0.5 / D`, with `V / D` used only as a narrow control, and the main work goes into collision failure diagnosis plus a delay-benchmark audit/redesign.",
        "",
        "## Chosen Base",
        "",
        f"- contract family: `{summary.get('chosen_contract_family', 'd')}`",
        f"- incumbent: `{summary.get('incumbent_pair', 'visit_taskgrad_half_d')}`",
        f"- control: `{summary.get('control_pair', 'visitonly_d')}`",
        f"- visible GPUs used: `{summary.get('visible_gpu_count', 0)}`",
        f"- schedules: `P={summary.get('budgets', {}).get('p', 0)}`, `M={summary.get('budgets', {}).get('m', 0)}`, `L={summary.get('budgets', {}).get('l', 0)}`",
        "",
        "## Collision Pack Definitions",
        "",
    ]
    for regime, payload in GEN.COLLISION_REGIMES.items():
        lines.append(
            f"- `{regime}`: writers `{payload['writers_per_episode']}`, home-pool `{payload['home_node_pool_size']}`, start-pool `{payload['start_node_pool_size']}`, ttl `{payload['query_ttl_min']}-{payload['query_ttl_max']}`"
        )
    lines.extend(["", "## Delay Pack Definitions", ""])
    for regime, payload in GEN.DELAY_REGIMES.items():
        lines.append(
            f"- `{regime}`: mode `{payload['mode']}`, required-delay `{payload['required_delay_min']}-{payload['required_delay_max']}`, hash-bits `{payload['required_delay_hash_bits']}`"
        )

    lines.extend(["", "## Completed Runs", ""])
    lines.extend(format_table(summary.get("records", [])))

    lines.extend(["", "## Collision Pack Summary", ""])
    collision_pack = summary.get("collision_pack", {})
    for regime, payload in collision_pack.get("baseline", {}).items():
        lines.append(
            f"- `{regime}` baseline: cache-on `{payload['cache_on'].get('dense_mean', {}).get('mean', 0.0):.4f}`, cache-off `{payload['cache_off'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{payload.get('dense_gap', 0.0):.4f}`"
        )
    for key, payload in collision_pack.get("interventions", {}).items():
        lines.append(
            f"- `{key}`: dense `{payload['summary'].get('dense_mean', {}).get('mean', 0.0):.4f}`, recovery vs cache-on `{payload.get('recovery_vs_cacheon', 0.0):.4f}`, recovery fraction `{payload.get('recovery_fraction', 0.0):.4f}`"
        )
    for key, payload in collision_pack.get("bypass_checks", {}).items():
        lines.append(
            f"- bypass `{key}`: normal `{payload.get('normal_dense', 0.0):.4f}`, bypass `{payload.get('bypass_dense', 0.0):.4f}`, delta `{payload.get('dense_delta', 0.0):.4f}`"
        )
    lines.append(f"- best intervention: `{collision_pack.get('best_intervention', '')}`")
    lines.append(f"- collision bundle positive: `{collision_pack.get('positive', False)}`")

    lines.extend(["", "## Delay Pack Audit Summary", ""])
    delay_pack = summary.get("delay_pack", {})
    for regime, payload in delay_pack.get("current", {}).items():
        lines.append(
            f"- current `{regime}`: learned `{payload['learned'].get('dense_mean', {}).get('mean', 0.0):.4f}`, zero `{payload['forced_zero'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{payload.get('dense_gap', 0.0):.4f}`"
        )
    for regime, payload in delay_pack.get("redesigned", {}).items():
        lines.append(
            f"- redesigned `{regime}`: learned `{payload['learned'].get('dense_mean', {}).get('mean', 0.0):.4f}`, zero `{payload['forced_zero'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{payload.get('dense_gap', 0.0):.4f}`"
        )
    for regime, payload in delay_pack.get("timing_oracle_audit", {}).items():
        lines.append(
            f"- timing audit `{regime}`: acceptable-delay-count `{payload.get('acceptable_delay_count_mean', 0.0):.2f}`, zero `{payload.get('zero_success_rate', 0.0):.4f}`, fixed `{payload.get('fixed_success_rate', 0.0):.4f}`, random `{payload.get('random_expected_success_rate', 0.0):.4f}`, oracle `{payload.get('oracle_required_success_rate', 0.0):.4f}`"
        )
    lines.append(f"- current benchmark needs redesign: `{delay_pack.get('current_benchmark_needs_redesign', False)}`")
    lines.append(f"- redesigned benchmark positive: `{delay_pack.get('redesign_positive', False)}`")

    lines.extend(["", "## Decodability / Source-Quality Audit", ""])
    for domain, payload in summary.get("decodability_audit", {}).items():
        lines.append(f"- `{domain}` audited checkpoints: `{len(payload)}`")
        for key, probes in payload.items():
            lines.append(
                f"  - `{key}` sink `{probes.get('final_sink_state', {}).get('test_accuracy', 0.0):.3f}`, cache `{probes.get('home_cache_mean_state', {}).get('test_accuracy', 0.0):.3f}`, hidden `{probes.get('home_hidden_state', {}).get('test_accuracy', 0.0):.3f}`"
            )

    lines.extend(["", "## Optional Follow-up", ""])
    optional_followup = summary.get("optional_followup", {})
    if optional_followup.get("triggered", False):
        lines.append(
            f"- triggered: `{optional_followup.get('path', '')}` because `{optional_followup.get('reason', '')}`"
        )
    else:
        lines.append("- not triggered")

    lines.extend(
        [
            "",
            "## Final Diagnosis",
            "",
            f"- headroom conclusion: `{summary.get('headroom_conclusion', 'pending')}`",
            f"- next move: `{summary.get('next_move', 'pending')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    records = []
    for meta, run_dir in latest_runs():
        if not is_complete_run(run_dir, meta["schedule"]):
            continue
        records.append(summarize_run(run_dir, meta))
    summary = build_summary(records)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
