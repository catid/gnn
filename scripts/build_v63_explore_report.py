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

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v63_explore_exploit.json"
REPORT_PATH = REPORTS / "final_report_v63_explore_exploit_contracts.md"
BASE_LR = 2.0e-4
ROLLING_N = 5
EXPECTED_TRAIN_STEPS = {"p": 300, "s": 810, "m": 1620, "l": 2268}


def load_generator():
    path = ROOT / "scripts" / "gen_v63_explore_configs.py"
    spec = importlib.util.spec_from_file_location("gen_v63_explore_configs", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


GEN = load_generator()
PREFIX = GEN.PREFIX
REGIME_WRITERS = {name: spec["train_eval_writers"] for name, spec in GEN.REGIMES.items()}
REGIME_LABELS = {name: spec["display"] for name, spec in GEN.REGIMES.items()}
ALL_ARMS = GEN.ARMS
EXPLOITATION_ARMS = GEN.EXPLOITATION_ARMS
EXPLORATION_ARMS = GEN.EXPLORATION_ARMS
ANCHOR_REGIMES = ["core", "t1", "t2a"]
EXPLORATION_SCREEN_REGIMES = {
    "stage_late_vt": ["core", "t2a"],
    "stage_early_vt": ["core", "t2a"],
    "stable_v": ["core", "t2a"],
    "stable_vt": ["core", "t2a"],
    "gonline": ["core", "t2a"],
    "slowcommit": ["core", "hmix"],
}
EXPLORATION_CONFIRM_REGIMES = ["t1", "hmix"]
SHARED_VERIFICATION_REGIMES = ["t1r", "t2b", "t2c", "hmid"]
FINAL_RERUN_CANDIDATE_REGIMES = ["core", "t1", "t2a", "hmix"]
EXTRA_DEPTH_REGIMES = ["core", "t1", "t2a", "hmix"]

RUN_RE = re.compile(
    r"v63ee-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
)
SURFACE_TAG_RE = re.compile(r"surface_w(?P<writers>\d+)_p(?P<pool>\d+)_t(?P<ttl_min>\d)(?P<ttl_max>\d)")


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


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


def latest_runs() -> list[tuple[dict[str, str], Path]]:
    pilot_runs: list[tuple[dict[str, str], Path]] = []
    latest: dict[tuple[str, str, str, str, str], tuple[dict[str, str], Path]] = {}
    for candidate in sorted(RUNS.glob(f"*-{PREFIX}-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["schedule"] == "p":
            pilot_runs.append((meta, candidate))
            continue
        key = (meta["regime"], meta["arm"], meta["schedule"], meta["tag"], meta["seed"])
        latest[key] = (meta, candidate)
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


def extract_eval(run_dir: Path, kind: str, writers: int, *, depth_tag: str | None = None) -> float:
    suffix = f"_{depth_tag}" if depth_tag else ""
    path = run_dir / f"eval_{kind}{suffix}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def extract_eval_metrics(run_dir: Path, kind: str, writers: int, *, depth_tag: str | None = None) -> dict[str, float]:
    suffix = f"_{depth_tag}" if depth_tag else ""
    path = run_dir / f"eval_{kind}{suffix}_k{writers}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _history_value(history: list[dict[str, Any]], step: int, key: str) -> float:
    for row in history:
        if int(row.get("step", 0)) >= step:
            return float(row.get(key, 0.0))
    if history:
        return float(history[-1].get(key, 0.0))
    return 0.0


def _time_to_threshold(history: list[dict[str, Any]], key: str, threshold: float) -> float:
    bootstrap_end = None
    for row in history:
        if not bool(row.get("bootstrap_active", False)):
            bootstrap_end = int(row.get("step", 0))
            break
    if bootstrap_end is None:
        return 0.0
    for row in history:
        step = int(row.get("step", 0))
        if step < bootstrap_end:
            continue
        if float(row.get(key, 0.0)) >= threshold:
            return float(step - bootstrap_end)
    return 0.0


def extract_coverage_and_split_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "coverage_summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    history = payload.get("history", [])
    stages = payload.get("stages", [])
    out = {
        "task_visit_cov_10": _history_value(history, 10, "task_visit_coverage"),
        "task_visit_cov_50": _history_value(history, 50, "task_visit_coverage"),
        "task_visit_cov_100": _history_value(history, 100, "task_visit_coverage"),
        "task_visit_cov_200": _history_value(history, 200, "task_visit_coverage"),
        "task_grad_cov_10": _history_value(history, 10, "task_gradient_coverage"),
        "task_grad_cov_50": _history_value(history, 50, "task_gradient_coverage"),
        "task_grad_cov_100": _history_value(history, 100, "task_gradient_coverage"),
        "task_grad_cov_200": _history_value(history, 200, "task_gradient_coverage"),
        "time_to_visit_50": _time_to_threshold(history, "task_visit_coverage", 0.50),
        "time_to_visit_75": _time_to_threshold(history, "task_visit_coverage", 0.75),
        "time_to_visit_100": _time_to_threshold(history, "task_visit_coverage", 1.00),
        "time_to_grad_50": _time_to_threshold(history, "task_gradient_coverage", 0.50),
        "time_to_grad_75": _time_to_threshold(history, "task_gradient_coverage", 0.75),
        "time_to_grad_100": _time_to_threshold(history, "task_gradient_coverage", 1.00),
    }
    split_blocks = [stage.get("split_stats", {}) for stage in stages if stage.get("split_stats")]
    if split_blocks:
        def split_scalar(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                return mean([float(item) for item in value.values()]) if value else 0.0
            return 0.0

        for key in (
            "selected_parent_child_usefulness_mean",
            "unselected_parent_child_usefulness_mean",
            "child_visit_share",
            "child_grad_share",
            "utility_usefulness_correlation",
        ):
            values = [split_scalar(block.get(key, 0.0)) for block in split_blocks if key in block]
            out[key] = mean(values)
    return out


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(ROLLING_N, len(vals)) :]
    arm = meta["arm"]
    arm_info = ALL_ARMS[arm]
    writers = REGIME_WRITERS[meta["regime"]]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    record: dict[str, Any] = {
        "run": run_dir.name,
        "arm": arm,
        "arm_label": arm_info["display"],
        "category": arm_info["category"],
        "selector": arm_info["selector"],
        "selector_label": arm_info["selector_label"],
        "contract": arm_info["contract"],
        "contract_label": arm_info["contract_label"],
        "regime": meta["regime"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"]),
        "tag": meta["tag"],
        "config_name": f"configs/{PREFIX}_{meta['regime']}_{arm}_32_{meta['schedule']}.yaml",
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "p_keep_prev": float(config["train"].get("contract_penultimate_keep_prob", 0.0)),
        "stability_weight": float(config["train"].get("late_stage_stability_weight", 0.0)),
        "slow_commit_interval": int(config["train"].get("slow_commit_interval", 0)),
        "gate_kind": str(config["growth"].get("selector_gate_kind", "none")),
        "gate_online_stage_index_min": int(config["growth"].get("selector_gate_online_stage_index_min", -1)),
        "gate_online_entropy_high_threshold": float(config["growth"].get("selector_gate_online_entropy_high_threshold", 0.0)),
        "gate_online_gini_high_threshold": float(config["growth"].get("selector_gate_online_gini_high_threshold", 0.0)),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in dense_writers])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in dense_writers])
    record["composite"] = (
        0.40 * record["dense_mean"]
        + 0.30 * record["last_val"]
        + 0.15 * record["last5_val_mean"]
        - 0.15 * record["best_to_last_drop"]
    )
    record.update(extract_coverage_and_split_metrics(run_dir))
    return record


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    out = {
        "count": len(records),
        "best_val": mean_std([record["best_val"] for record in records]),
        "last_val": mean_std([record["last_val"] for record in records]),
        "last5_val_mean": mean_std([record["last5_val_mean"] for record in records]),
        "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in records]),
        "dense_mean": mean_std([record["dense_mean"] for record in records]),
        "last_dense_mean": mean_std([record["last_dense_mean"] for record in records]),
        "composite": mean_std([record["composite"] for record in records]),
        "lr_multiplier": mean_std([record["lr_multiplier"] for record in records]),
        "p_keep_prev": mean_std([record["p_keep_prev"] for record in records]),
        "stability_weight": mean_std([record["stability_weight"] for record in records]),
        "slow_commit_interval": mean_std([float(record["slow_commit_interval"]) for record in records]),
        "query_first_hop_home_rate": mean_std([record.get("query_first_hop_home_rate", 0.0) for record in records]),
        "delivery_rate": mean_std([record.get("delivery_rate", 0.0) for record in records]),
        "home_to_out_rate": mean_std([record.get("home_to_out_rate", 0.0) for record in records]),
        "task_visit_cov_10": mean_std([record.get("task_visit_cov_10", 0.0) for record in records]),
        "task_visit_cov_50": mean_std([record.get("task_visit_cov_50", 0.0) for record in records]),
        "task_visit_cov_100": mean_std([record.get("task_visit_cov_100", 0.0) for record in records]),
        "task_visit_cov_200": mean_std([record.get("task_visit_cov_200", 0.0) for record in records]),
        "task_grad_cov_10": mean_std([record.get("task_grad_cov_10", 0.0) for record in records]),
        "task_grad_cov_50": mean_std([record.get("task_grad_cov_50", 0.0) for record in records]),
        "task_grad_cov_100": mean_std([record.get("task_grad_cov_100", 0.0) for record in records]),
        "task_grad_cov_200": mean_std([record.get("task_grad_cov_200", 0.0) for record in records]),
        "selected_parent_child_usefulness_mean": mean_std(
            [record.get("selected_parent_child_usefulness_mean", 0.0) for record in records]
        ),
        "unselected_parent_child_usefulness_mean": mean_std(
            [record.get("unselected_parent_child_usefulness_mean", 0.0) for record in records]
        ),
        "child_visit_share": mean_std([record.get("child_visit_share", 0.0) for record in records]),
        "child_grad_share": mean_std([record.get("child_grad_share", 0.0) for record in records]),
        "utility_usefulness_correlation": mean_std(
            [record.get("utility_usefulness_correlation", 0.0) for record in records]
        ),
        "runs": [record["run"] for record in records],
    }
    return out


def pilot_candidate_key(record: dict[str, Any]) -> str:
    fields = [f"lr{record['lr_multiplier']:.1f}"]
    if record["contract"] in {"ds", "dsg"}:
        fields.append(f"p{record['p_keep_prev']:.2f}")
    if record["stability_weight"] > 0.0:
        fields.append(f"lam{record['stability_weight']:.3f}")
    if record["slow_commit_interval"] > 1:
        fields.append(f"c{record['slow_commit_interval']}")
    if record["gate_kind"] == "online":
        fields.append(f"s{record['gate_online_stage_index_min']}")
        fields.append(f"e{record['gate_online_entropy_high_threshold']:.2f}")
        fields.append(f"g{record['gate_online_gini_high_threshold']:.2f}")
    return f"{record['arm']}@{'@'.join(fields)}"


def select_pilot_settings(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(pilot_candidate_key(record), []).append(record)
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for key, candidate_records in grouped.items():
        row = {
            "candidate": key,
            "arm": candidate_records[0]["arm"],
            "arm_label": candidate_records[0]["arm_label"],
            "summary": summarize_group(candidate_records),
            "lr_multiplier": candidate_records[0]["lr_multiplier"],
            "p_keep_prev": candidate_records[0]["p_keep_prev"],
            "stability_weight": candidate_records[0]["stability_weight"],
            "slow_commit_interval": candidate_records[0]["slow_commit_interval"],
            "gate_online_stage_index_min": candidate_records[0]["gate_online_stage_index_min"],
            "gate_online_entropy_high_threshold": candidate_records[0]["gate_online_entropy_high_threshold"],
            "gate_online_gini_high_threshold": candidate_records[0]["gate_online_gini_high_threshold"],
        }
        per_arm.setdefault(row["arm"], []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for arm, rows in per_arm.items():
        rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
        selected[arm] = rows[0]
    return selected


def rank_arms(records: list[dict[str, Any]], arms: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(arms) if arms is not None else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if allowed is not None and record["arm"] not in allowed:
            continue
        grouped.setdefault(record["arm"], []).append(record)
    rows = []
    for arm, arm_records in grouped.items():
        row = {
            "arm": arm,
            "arm_label": ALL_ARMS[arm]["display"],
            "category": ALL_ARMS[arm]["category"],
            "selector_label": ALL_ARMS[arm]["selector_label"],
            "contract_label": ALL_ARMS[arm]["contract_label"],
            "contract": ALL_ARMS[arm]["contract"],
            "summary": summarize_group(arm_records),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    return rows


def promote_exploitation_pairs(rankings: list[dict[str, Any]]) -> list[str]:
    if not rankings:
        return []
    promoted = rankings[:4]
    if len(promoted) < 4:
        return [row["arm"] for row in promoted]
    contracts = {row["contract"] for row in promoted}
    if len(contracts) >= 2:
        return [row["arm"] for row in promoted]
    floor_mean = promoted[-1]["summary"]["composite"]["mean"]
    floor_std = promoted[-1]["summary"]["composite"]["std"]
    for candidate in rankings[4:]:
        if candidate["contract"] in contracts:
            continue
        candidate_mean = candidate["summary"]["composite"]["mean"]
        candidate_std = candidate["summary"]["composite"]["std"]
        if candidate_mean + max(candidate_std, floor_std) >= floor_mean:
            promoted[-1] = candidate
            break
    return [row["arm"] for row in promoted]


def promote_exploration_arms(rankings: list[dict[str, Any]]) -> list[str]:
    return [row["arm"] for row in rankings[:3]]


def summarize_by_regime_and_arm(records: list[dict[str, Any]], arms: list[str] | None = None) -> dict[str, dict[str, Any]]:
    allowed = set(arms) if arms is not None else None
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted({record["regime"] for record in records}):
        regime_records = [record for record in records if record["regime"] == regime and (allowed is None or record["arm"] in allowed)]
        if not regime_records:
            continue
        out[regime] = {}
        for arm in sorted({record["arm"] for record in regime_records}):
            out[regime][arm] = summarize_group([record for record in regime_records if record["arm"] == arm])
    return out


def arm_total(summary: dict[str, dict[str, Any]], arm: str, regimes: list[str]) -> float:
    return sum(summary.get(regime, {}).get(arm, {}).get("composite", {}).get("mean", 0.0) for regime in regimes)


def choose_top2_exploration(screen_records: list[dict[str, Any]], confirm_records: list[dict[str, Any]], promoted: list[str]) -> list[str]:
    if not promoted:
        return []
    combined = rank_arms(screen_records + confirm_records, promoted)
    return [row["arm"] for row in combined[:2]]


def choose_top2_overall(
    finalists: list[str],
    track_records: list[dict[str, Any]],
    shared_summary: dict[str, dict[str, Any]],
) -> list[str]:
    if len(finalists) <= 2:
        return finalists
    track_rank = {row["arm"]: row for row in rank_arms(track_records, finalists)}
    def score(arm: str) -> float:
        return track_rank[arm]["summary"]["composite"]["mean"] + arm_total(shared_summary, arm, SHARED_VERIFICATION_REGIMES)
    return sorted(finalists, key=score, reverse=True)[:2]


def choose_final_rerun_regimes(finalists: list[str], records: list[dict[str, Any]]) -> list[str]:
    if len(finalists) < 2:
        return []
    summaries = summarize_by_regime_and_arm(records, finalists)
    margins: list[tuple[float, str]] = []
    for regime in FINAL_RERUN_CANDIDATE_REGIMES:
        if regime not in summaries:
            continue
        first = summaries[regime].get(finalists[0], {}).get("composite", {}).get("mean", 0.0)
        second = summaries[regime].get(finalists[1], {}).get("composite", {}).get("mean", 0.0)
        margins.append((abs(first - second), regime))
    margins.sort(key=lambda item: item[0])
    return [regime for _, regime in margins[:2]]


def _source_run_priority(meta: dict[str, str]) -> tuple[int, int]:
    tag_rank = 2 if meta.get("tag") == "rerun" else 1 if meta.get("tag") == "" else 0
    schedule_rank = {"l": 3, "m": 2, "s": 1, "p": 0}.get(meta.get("schedule", ""), 0)
    return (tag_rank, schedule_rank)


def pick_source_run(runs: list[tuple[dict[str, str], Path]], arm: str, regime: str) -> Path | None:
    candidates = [(meta, path) for meta, path in runs if meta.get("arm") == arm and meta.get("regime") == regime]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (_source_run_priority(item[0]), item[1].name))
    return candidates[-1][1]


def collect_extra_depth(finalists: list[str], runs: list[tuple[dict[str, str], Path]]) -> dict[str, dict[str, Any]]:
    depth_tags = ["standard", "plus50", "plus100", "settle"]
    out: dict[str, dict[str, Any]] = {}
    for arm in finalists:
        arm_out: dict[str, Any] = {}
        for regime in EXTRA_DEPTH_REGIMES:
            run_dir = pick_source_run(runs, arm, regime)
            if run_dir is None:
                continue
            dense_writers = REGIME_WRITERS[regime][1:] if len(REGIME_WRITERS[regime]) > 1 else REGIME_WRITERS[regime]
            regime_out: dict[str, Any] = {}
            for kind in ("best", "last"):
                kind_out: dict[str, Any] = {}
                for depth_tag in depth_tags:
                    dense_metrics = [extract_eval_metrics(run_dir, kind, writer, depth_tag=depth_tag) for writer in dense_writers]
                    dense_metrics = [metrics for metrics in dense_metrics if metrics]
                    if not dense_metrics:
                        continue
                    kind_out[depth_tag] = {
                        "dense_mean": mean([metrics.get("query_accuracy", 0.0) for metrics in dense_metrics]),
                        "settle_rate": mean([metrics.get("settle_rate", metrics.get("query_delivery_rate", 0.0)) for metrics in dense_metrics]),
                        "steps_to_settle": mean([metrics.get("steps_to_settle", metrics.get("average_hops", 0.0)) for metrics in dense_metrics]),
                        "settled_accuracy": mean([metrics.get("settled_accuracy", metrics.get("query_accuracy", 0.0)) for metrics in dense_metrics]),
                        "accept_on_settle_accuracy": mean(
                            [metrics.get("accept_on_settle_accuracy", metrics.get("query_accuracy", 0.0)) for metrics in dense_metrics]
                        ),
                        "accept_on_settle_coverage": mean(
                            [metrics.get("accept_on_settle_coverage", metrics.get("query_delivery_rate", 0.0)) for metrics in dense_metrics]
                        ),
                        "non_settling_rate": mean([metrics.get("non_settling_rate", 0.0) for metrics in dense_metrics]),
                    }
                if kind_out:
                    regime_out[kind] = kind_out
            if regime_out:
                arm_out[regime] = regime_out
        out[arm] = arm_out
    return out


def collect_surface_maps(finalists: list[str], runs: list[tuple[dict[str, str], Path]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm in finalists:
        arm_out: dict[str, Any] = {}
        for regime in EXTRA_DEPTH_REGIMES:
            run_dir = pick_source_run(runs, arm, regime)
            if run_dir is None:
                continue
            entries: dict[str, Any] = {}
            for path in sorted(run_dir.glob("eval_best_surface_w*_p*_t*.json")):
                payload = read_json(path)
                metrics = payload.get("metrics", payload)
                match = SURFACE_TAG_RE.search(path.stem.replace("eval_best_", ""))
                if match is None:
                    continue
                key = f"w{match.group('writers')}_p{match.group('pool')}_t{match.group('ttl_min')}{match.group('ttl_max')}"
                entries[key] = {
                    "query_accuracy": float(metrics.get("query_accuracy", 0.0)),
                    "query_delivery_rate": float(metrics.get("query_delivery_rate", 0.0)),
                    "query_first_hop_home_rate": float(metrics.get("query_first_hop_home_rate", 0.0)),
                    "query_home_to_output_rate": float(metrics.get("query_home_to_output_rate", 0.0)),
                }
            if entries:
                arm_out[regime] = entries
        out[arm] = arm_out
    return out


def overall_rows(
    finalists: list[str],
    track_records: list[dict[str, Any]],
    shared_summary: dict[str, dict[str, Any]],
    rerun_summary: dict[str, dict[str, Any]],
    ambiguity_summary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    track_rank = {row["arm"]: row for row in rank_arms(track_records, finalists)}
    rows = []
    for arm in finalists:
        score = (
            track_rank[arm]["summary"]["composite"]["mean"]
            + arm_total(shared_summary, arm, SHARED_VERIFICATION_REGIMES)
            + arm_total(rerun_summary, arm, list(rerun_summary))
            + arm_total(ambiguity_summary, arm, list(ambiguity_summary))
        )
        rows.append(
            {
                "arm": arm,
                "arm_label": ALL_ARMS[arm]["display"],
                "track_mean": track_rank[arm]["summary"]["composite"]["mean"],
                "shared_total": arm_total(shared_summary, arm, SHARED_VERIFICATION_REGIMES),
                "rerun_total": arm_total(rerun_summary, arm, list(rerun_summary)),
                "ambiguity_total": arm_total(ambiguity_summary, arm, list(ambiguity_summary)),
                "score": score,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def classify_outcome(
    final_rows: list[dict[str, Any]],
    rerun_summary: dict[str, dict[str, Any]],
    ambiguity_summary: dict[str, dict[str, Any]],
) -> dict[str, str]:
    if len(final_rows) < 2:
        return {"outcome": "unresolved", "winner": final_rows[0]["arm"] if final_rows else "none"}
    top, runner_up = final_rows[:2]
    gap = top["score"] - runner_up["score"]
    rerun_gap = arm_total(rerun_summary, top["arm"], list(rerun_summary)) - arm_total(
        rerun_summary,
        runner_up["arm"],
        list(rerun_summary),
    )
    ambiguity_gap = arm_total(ambiguity_summary, top["arm"], list(ambiguity_summary)) - arm_total(
        ambiguity_summary,
        runner_up["arm"],
        list(ambiguity_summary),
    )
    if gap <= 0.01 or abs(rerun_gap) < 0.01 or (ambiguity_summary and abs(ambiguity_gap) < 0.01):
        return {"outcome": "unresolved", "winner": "unresolved"}
    if top["arm"] in EXPLORATION_ARMS:
        return {"outcome": "dynamic_rule", "winner": top["arm"]}
    return {"outcome": "selector_contract", "winner": top["arm"]}


def rows_table(records: list[dict[str, Any]]) -> str:
    header = "| Phase | Regime | Arm | Seed | LR x | Composite |\n| --- | --- | --- | --- | --- | --- |"
    body = [
        f"| {record['schedule'].upper()}{('-' + record['tag']) if record['tag'] else ''} | {record['regime']} | {record['arm_label']} | {record['seed']} | {record['lr_multiplier']:.2f} | {record['composite']:.4f} |"
        for record in records
    ]
    return "\n".join([header, *body]) if body else header


def summary_table(rows: list[dict[str, Any]]) -> str:
    header = "| Arm | Dense | Last | Last5 | Drop | Composite |\n| --- | --- | --- | --- | --- | --- |"
    body = [
        f"| {row['arm_label']} | {row['summary']['dense_mean']['mean']:.4f} | {row['summary']['last_val']['mean']:.4f} | {row['summary']['last5_val_mean']['mean']:.4f} | {row['summary']['best_to_last_drop']['mean']:.4f} | {row['summary']['composite']['mean']:.4f} |"
        for row in rows
    ]
    return "\n".join([header, *body]) if body else header


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    runs = [(meta, path) for meta, path in latest_runs() if meta and is_complete_run(path, meta["schedule"])]
    records = [summarize_run(path, meta) for meta, path in runs]

    pilot_records = [record for record in records if record["schedule"] == "p"]
    selected_settings = select_pilot_settings(pilot_records)

    exploit_records = [
        record for record in records
        if record["schedule"] == "s" and record["arm"] in EXPLOITATION_ARMS and record["tag"] == ""
    ]
    exploit_rankings = rank_arms(exploit_records, list(EXPLOITATION_ARMS))
    promoted_exploitation = promote_exploitation_pairs(exploit_rankings)

    explore_screen_records = [
        record for record in records
        if record["schedule"] == "s" and record["arm"] in EXPLORATION_ARMS and record["tag"] == ""
    ]
    explore_screen_rankings = rank_arms(explore_screen_records, list(EXPLORATION_ARMS))
    promoted_exploration = promote_exploration_arms(explore_screen_rankings)

    explore_confirm_records = [
        record for record in records
        if record["schedule"] == "m" and record["arm"] in promoted_exploration and record["regime"] in EXPLORATION_CONFIRM_REGIMES
    ]
    exploration_combined_rankings = rank_arms(explore_screen_records + explore_confirm_records, promoted_exploration)
    top2_exploration = choose_top2_exploration(explore_screen_records, explore_confirm_records, promoted_exploration)

    finalists = promoted_exploitation[:2] + top2_exploration
    shared_records = [
        record for record in records
        if record["schedule"] == "m" and record["arm"] in finalists and record["regime"] in SHARED_VERIFICATION_REGIMES
    ]
    shared_summary = summarize_by_regime_and_arm(shared_records, finalists)

    track_records = exploit_records + explore_screen_records + explore_confirm_records
    final_candidates = choose_top2_overall(finalists, track_records, shared_summary)
    final_rerun_regimes = choose_final_rerun_regimes(final_candidates, track_records + shared_records)

    rerun_records = [
        record for record in records
        if record["schedule"] == "l" and record["arm"] in final_candidates and record["tag"] == "rerun"
    ]
    rerun_summary = summarize_by_regime_and_arm(rerun_records, final_candidates)
    ambiguity_records = [
        record for record in records
        if record["schedule"] == "l" and record["arm"] in final_candidates and record["tag"] == "amb"
    ]
    ambiguity_summary = summarize_by_regime_and_arm(ambiguity_records, final_candidates)

    extra_depth = collect_extra_depth(final_candidates, runs)
    surface_maps = collect_surface_maps(final_candidates, runs)
    overall = overall_rows(final_candidates, track_records, shared_summary, rerun_summary, ambiguity_summary)
    outcome = classify_outcome(overall, rerun_summary, ambiguity_summary)

    summary = {
        "budgets": {name: spec["train_steps"] for name, spec in GEN.SCHEDULES.items()},
        "visible_gpus": visible_gpu_count(),
        "counts": {
            "p": len(pilot_records),
            "s_exploit": len(exploit_records),
            "s_explore": len(explore_screen_records),
            "m_explore": len(explore_confirm_records),
            "m_shared": len(shared_records),
            "l_rerun": len(rerun_records),
            "l_ambiguity": len(ambiguity_records),
        },
        "pilot_choices": selected_settings,
        "selected_settings": selected_settings,
        "exploit_rankings": exploit_rankings,
        "promoted_exploitation": promoted_exploitation,
        "explore_screen_rankings": explore_screen_rankings,
        "promoted_exploration": promoted_exploration,
        "explore_confirmation_rankings": exploration_combined_rankings,
        "top2_exploration": top2_exploration,
        "finalists": finalists,
        "finalists_shared_verification": finalists,
        "shared_verification": shared_summary,
        "shared_verification_summary": shared_summary,
        "top2_overall": final_candidates,
        "final_candidates": final_candidates,
        "rerun_regimes": final_rerun_regimes,
        "final_rerun_regimes": final_rerun_regimes,
        "rerun_summary": rerun_summary,
        "ambiguity_breaker": ambiguity_summary,
        "ambiguity_summary": ambiguity_summary,
        "extra_depth": extra_depth,
        "surface_maps": surface_maps,
        "overall": overall,
        "outcome": outcome,
        "winner": outcome.get("winner"),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    completed_records = exploit_records + explore_screen_records + explore_confirm_records + shared_records + rerun_records + ambiguity_records

    report_lines = [
        "# APSGNN v63: Explore / Exploit Contracts",
        "",
        "## What Changed From v62",
        "",
        "This sibling v63 pass keeps the strong static baselines (`V`, `VT-0.5`) but broadens the campaign into a 50/50 explore/exploit split around contract dynamics, stage-conditioned schedules, late-stage stability regularization, an online non-oracle gate, and a minimal slow-commit pilot.",
        "",
        "The existing narrow v63 contract tie-break remains intact. This pass uses a separate `v63ee` artifact prefix to preserve that line while extending the question.",
        "",
        "## Budgets",
        "",
        *(f"- `{name.upper()} = {spec['train_steps']}`" for name, spec in GEN.SCHEDULES.items()),
        f"- visible GPUs used: `{visible_gpu_count()}`",
        f"- rolling late-stage window: `{ROLLING_N}` evals",
        "",
        "## Exact Regimes",
        "",
        *(f"- `{name}`: writers={spec['writers_per_episode']}, start_pool={spec['start_node_pool_size']}, ttl={spec['query_ttl_min']}..{spec['query_ttl_max']}, rollout={spec['max_rollout_steps']}, eval densities={','.join(str(x) for x in spec['train_eval_writers'])}" for name, spec in GEN.REGIMES.items()),
        "",
        "## Calibration Settings",
        "",
        "| Arm | LR x | p_keep_prev | lambda | slow_commit | Composite |",
        "| --- | --- | --- | --- | --- | --- |",
        *(
            f"| {ALL_ARMS[arm]['display']} | {row['lr_multiplier']:.2f} | {row['p_keep_prev']:.2f} | {row['stability_weight']:.3f} | {row['slow_commit_interval']} | {row['summary']['composite']['mean']:.4f} |"
            for arm, row in sorted(selected_settings.items())
        ),
        "",
        "## Completed Runs",
        "",
        rows_table(completed_records),
        "",
        "## Exploitation Summary",
        "",
        summary_table(exploit_rankings),
        "",
        f"Promoted exploitation pairs: `{', '.join(promoted_exploitation)}`",
        "",
        "## Exploration Summary",
        "",
        "### E1 Screening",
        "",
        summary_table(explore_screen_rankings),
        "",
        "### E2 Promoted Arms",
        "",
        summary_table(exploration_combined_rankings),
        "",
        f"Promoted exploration arms: `{', '.join(promoted_exploration)}`",
        f"Top 2 exploration arms: `{', '.join(top2_exploration)}`",
        "",
        "## Shared Verification",
        "",
        f"Finalists: `{', '.join(finalists)}`",
        "",
        "| Regime | Arm | Composite |",
        "| --- | --- | --- |",
    ]
    for regime in SHARED_VERIFICATION_REGIMES:
        for arm in finalists:
            composite = shared_summary.get(regime, {}).get(arm, {}).get("composite", {}).get("mean", 0.0)
            report_lines.append(f"| {regime} | {ALL_ARMS[arm]['display']} | {composite:.4f} |")

    report_lines.extend(
        [
            "",
            "## Fresh Reruns",
            "",
            f"Chosen rerun regimes: `{', '.join(final_rerun_regimes)}`",
            "",
            "| Regime | Arm | Composite |",
            "| --- | --- | --- |",
        ]
    )
    for regime in final_rerun_regimes:
        for arm in final_candidates:
            composite = rerun_summary.get(regime, {}).get(arm, {}).get("composite", {}).get("mean", 0.0)
            report_lines.append(f"| {regime} | {ALL_ARMS[arm]['display']} | {composite:.4f} |")

    if ambiguity_records:
        report_lines.extend(
            [
                "",
                "## Ambiguity Breaker",
                "",
                "| Regime | Arm | Composite |",
                "| --- | --- | --- |",
            ]
        )
        for regime in sorted(ambiguity_summary):
            for arm in final_candidates:
                composite = ambiguity_summary.get(regime, {}).get(arm, {}).get("composite", {}).get("mean", 0.0)
                report_lines.append(f"| {regime} | {ALL_ARMS[arm]['display']} | {composite:.4f} |")

    report_lines.extend(
        [
            "",
            "## Extra Compute / Settling",
            "",
            "| Arm | Regime | Best settle dense | Settle rate | Steps | Accept-on-settle acc |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for arm in final_candidates:
        for regime in EXTRA_DEPTH_REGIMES:
            settle = extra_depth.get(arm, {}).get(regime, {}).get("best", {}).get("settle")
            if not settle:
                continue
            report_lines.append(
                f"| {ALL_ARMS[arm]['display']} | {regime} | {settle['dense_mean']:.4f} | {settle['settle_rate']:.4f} | {settle['steps_to_settle']:.2f} | {settle['accept_on_settle_accuracy']:.4f} |"
            )

    report_lines.extend(
        [
            "",
            "## Final Diagnosis",
            "",
            f"- outcome: `{outcome['outcome']}`",
            f"- winner: `{outcome['winner']}`",
            "",
            "## Final Overall Ranking",
            "",
            "| Arm | Track mean | Shared total | Rerun total | Ambiguity total | Score |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in overall:
        report_lines.append(
            f"| {ALL_ARMS[row['arm']]['display']} | {row['track_mean']:.4f} | {row['shared_total']:.4f} | {row['rerun_total']:.4f} | {row['ambiguity_total']:.4f} | {row['score']:.4f} |"
        )

    report_lines.extend(
        [
            "",
            "- Summary JSON: [summary_metrics_v63_explore_exploit.json](/home/catid/gnn/reports/summary_metrics_v63_explore_exploit.json)",
            "- Report: [final_report_v63_explore_exploit_contracts.md](/home/catid/gnn/reports/final_report_v63_explore_exploit_contracts.md)",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
