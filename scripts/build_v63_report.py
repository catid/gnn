#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
import subprocess
import re
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v63.json"
REPORT_PATH = REPORTS / "final_report_v63_rsm_lite_contract_tiebreak.md"

BASE_LR = 2.0e-4
ROLLING_N = 5
EXPECTED_TRAIN_STEPS = {"p": 420, "s": 1134, "m": 2268, "l": 3024}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t1r": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
    "t2b": [4, 8, 12, 14],
    "t2c": [6, 10, 14, 16],
    "hmid": [3, 7, 11],
    "hmix": [3, 7, 11],
}

ANCHOR_REGIMES = ["core", "t1", "t2a"]
VERIFICATION_REGIMES = ["t1r", "t2b", "t2c", "hmid", "hmix"]
EXTRA_DEPTH_REGIMES = ["core", "t1", "t2a", "hmix"]
ALL_REGIMES = ANCHOR_REGIMES + VERIFICATION_REGIMES

PAIR_INFO = {
    "visitonly_b": {"selector": "visitonly", "selector_label": "V", "contract": "b", "contract_label": "B", "label": "V/B"},
    "visitonly_d": {"selector": "visitonly", "selector_label": "V", "contract": "d", "contract_label": "D", "label": "V/D"},
    "visitonly_ds": {"selector": "visitonly", "selector_label": "V", "contract": "ds", "contract_label": "DS", "label": "V/DS"},
    "visitonly_dsg": {"selector": "visitonly", "selector_label": "V", "contract": "dsg", "contract_label": "DSG", "label": "V/DSG"},
    "visit_taskgrad_half_b": {
        "selector": "visit_taskgrad_half",
        "selector_label": "VT-0.5",
        "contract": "b",
        "contract_label": "B",
        "label": "VT-0.5/B",
    },
    "visit_taskgrad_half_d": {
        "selector": "visit_taskgrad_half",
        "selector_label": "VT-0.5",
        "contract": "d",
        "contract_label": "D",
        "label": "VT-0.5/D",
    },
    "visit_taskgrad_half_ds": {
        "selector": "visit_taskgrad_half",
        "selector_label": "VT-0.5",
        "contract": "ds",
        "contract_label": "DS",
        "label": "VT-0.5/DS",
    },
    "visit_taskgrad_half_dsg": {
        "selector": "visit_taskgrad_half",
        "selector_label": "VT-0.5",
        "contract": "dsg",
        "contract_label": "DSG",
        "label": "VT-0.5/DSG",
    },
}

CONTRACT_SETTINGS = {
    "b": "Baseline v62 contract",
    "d": "DetachedWarmup",
    "ds": "DetachedWarmup + stochastic penultimate credit",
    "dsg": "DS + grow-short / test-deep",
}

RUN_RE = re.compile(
    r"v63-(?P<regime>[^-]+)-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
)


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


def latest_v63_runs() -> list[tuple[dict[str, str], Path]]:
    latest: dict[tuple[str, str, str, str, str], Path] = {}
    for candidate in sorted(RUNS.glob("*-v63-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        key = (meta["regime"], meta["pair"], meta["schedule"], meta["tag"], meta["seed"])
        latest[key] = candidate
    return [(parse_run_name(path.name) or {}, path) for path in latest.values()]


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
    pair = meta["pair"]
    pair_info = PAIR_INFO[pair]
    writers = REGIME_WRITERS[meta["regime"]]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    record: dict[str, Any] = {
        "run": run_dir.name,
        "regime": meta["regime"],
        "pair": pair,
        "pair_label": pair_info["label"],
        "selector": pair_info["selector"],
        "selector_label": pair_info["selector_label"],
        "contract": pair_info["contract"],
        "contract_label": pair_info["contract_label"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"]),
        "tag": meta["tag"],
        "config_name": f"configs/v63_{meta['regime']}_{pair}_32_{meta['schedule']}.yaml",
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "p_keep_prev": float(config["train"].get("contract_penultimate_keep_prob", 0.0)),
        "contract_kind": str(config["train"].get("contract_kind", "baseline")),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
        "effective_rollout_steps": float(last.get("train/effective_rollout_steps", config["task"]["max_rollout_steps"])),
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
    record["score"] = record["last_val"] + record["dense_mean"] - record["best_to_last_drop"]
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
        "score": mean_std([record["score"] for record in records]),
        "lr_multiplier": mean_std([record["lr_multiplier"] for record in records]),
        "p_keep_prev": mean_std([record["p_keep_prev"] for record in records]),
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
        "time_to_visit_50": mean_std([record.get("time_to_visit_50", 0.0) for record in records]),
        "time_to_visit_75": mean_std([record.get("time_to_visit_75", 0.0) for record in records]),
        "time_to_visit_100": mean_std([record.get("time_to_visit_100", 0.0) for record in records]),
        "time_to_grad_50": mean_std([record.get("time_to_grad_50", 0.0) for record in records]),
        "time_to_grad_75": mean_std([record.get("time_to_grad_75", 0.0) for record in records]),
        "time_to_grad_100": mean_std([record.get("time_to_grad_100", 0.0) for record in records]),
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
    all_writers = sorted({writer for record in records for writer in REGIME_WRITERS[record["regime"]]})
    for writer in all_writers:
        if all(f"k{writer}" in record for record in records):
            out[f"k{writer}"] = mean_std([record[f"k{writer}"] for record in records])
        if all(f"last_k{writer}" in record for record in records):
            out[f"last_k{writer}"] = mean_std([record[f"last_k{writer}"] for record in records])
    return out


def pilot_candidate_key(record: dict[str, Any]) -> str:
    p_keep = float(record["p_keep_prev"])
    if record["contract"] in {"ds", "dsg"}:
        return f"{record['pair']}@lr{record['lr_multiplier']:.1f}@p{p_keep:.2f}"
    return f"{record['pair']}@lr{record['lr_multiplier']:.1f}"


def select_pilot_settings(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(pilot_candidate_key(record), []).append(record)
    per_pair: dict[str, list[dict[str, Any]]] = {}
    for key, candidate_records in grouped.items():
        row = {
            "candidate": key,
            "pair": candidate_records[0]["pair"],
            "pair_label": candidate_records[0]["pair_label"],
            "selector": candidate_records[0]["selector"],
            "contract": candidate_records[0]["contract"],
            "lr_multiplier": candidate_records[0]["lr_multiplier"],
            "p_keep_prev": candidate_records[0]["p_keep_prev"],
            "summary": summarize_group(candidate_records),
        }
        per_pair.setdefault(row["pair"], []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for pair, rows in per_pair.items():
        rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
        selected[pair] = rows[0]
    return selected


def rank_pairs(records: list[dict[str, Any]], pairs: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(pairs) if pairs is not None else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if allowed is not None and record["pair"] not in allowed:
            continue
        grouped.setdefault(record["pair"], []).append(record)
    rows = []
    for pair, pair_records in grouped.items():
        row = {
            "pair": pair,
            "pair_label": PAIR_INFO[pair]["label"],
            "selector": PAIR_INFO[pair]["selector"],
            "selector_label": PAIR_INFO[pair]["selector_label"],
            "contract": PAIR_INFO[pair]["contract"],
            "contract_label": PAIR_INFO[pair]["contract_label"],
            "summary": summarize_group(pair_records),
        }
        rows.append(row)
    rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    return rows


def promote_screening_pairs(rankings: list[dict[str, Any]]) -> list[str]:
    promoted = rankings[:4]
    contracts = {row["contract"] for row in promoted}
    if len(contracts) >= 2:
        return [row["pair"] for row in promoted]
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
    return [row["pair"] for row in promoted]


def summarize_by_regime_and_pair(records: list[dict[str, Any]], pairs: list[str] | None = None) -> dict[str, dict[str, Any]]:
    allowed = set(pairs) if pairs is not None else None
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted({record["regime"] for record in records}):
        regime_records = [record for record in records if record["regime"] == regime and (allowed is None or record["pair"] in allowed)]
        if not regime_records:
            continue
        out[regime] = {}
        for pair in sorted({record["pair"] for record in regime_records}):
            out[regime][pair] = summarize_group([record for record in regime_records if record["pair"] == pair])
    return out


def pair_total(summary: dict[str, dict[str, Any]], pair: str, regimes: list[str]) -> float:
    return sum(summary.get(regime, {}).get(pair, {}).get("composite", {}).get("mean", 0.0) for regime in regimes)


def pair_total_std(summary: dict[str, dict[str, Any]], pair: str, regimes: list[str]) -> float:
    return math.sqrt(
        sum(summary.get(regime, {}).get(pair, {}).get("composite", {}).get("std", 0.0) ** 2 for regime in regimes)
    )


def pick_top_pairs(confirmation_summary: dict[str, dict[str, Any]], holdout_summary: dict[str, dict[str, Any]], candidates: list[str]) -> list[str]:
    def score(pair: str) -> float:
        return pair_total(confirmation_summary, pair, ANCHOR_REGIMES) + pair_total(holdout_summary, pair, VERIFICATION_REGIMES)

    ranked = sorted(candidates, key=score, reverse=True)
    return ranked[:2]


def choose_final_rerun_regimes(top_pairs: list[str], confirmation_summary: dict[str, dict[str, Any]], holdout_summary: dict[str, dict[str, Any]]) -> list[str]:
    candidate_regimes = ["core", "t1", "t2a", "hmix"]
    margins: list[tuple[float, str]] = []
    for regime in candidate_regimes:
        source = holdout_summary if regime in holdout_summary else confirmation_summary
        if regime not in source:
            continue
        first = source[regime].get(top_pairs[0], {}).get("composite", {}).get("mean", 0.0)
        second = source[regime].get(top_pairs[1], {}).get("composite", {}).get("mean", 0.0)
        margins.append((abs(first - second), regime))
    margins.sort(key=lambda item: item[0])
    return [regime for _, regime in margins[:2]]


def collect_extra_depth(top_pairs: list[str]) -> dict[str, dict[str, Any]]:
    depth_tags = ["standard", "plus50", "plus100", "settle"]
    out: dict[str, dict[str, Any]] = {}
    runs = latest_v63_runs()
    for pair in top_pairs:
        pair_out: dict[str, Any] = {}
        for regime in EXTRA_DEPTH_REGIMES:
            candidates = [
                path for meta, path in runs
                if meta.get("pair") == pair and meta.get("regime") == regime and meta.get("schedule") in {"m", "l"}
            ]
            if not candidates:
                continue
            run_dir = sorted(candidates)[-1]
            writers = REGIME_WRITERS[regime]
            dense_writers = writers[1:] if len(writers) > 1 else writers
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
                        "settle_rate": mean([metrics.get("query_delivery_rate", 0.0) for metrics in dense_metrics]),
                        "steps_to_settle": mean([metrics.get("average_hops", 0.0) for metrics in dense_metrics]),
                        "settled_accuracy": mean([metrics.get("query_accuracy", 0.0) for metrics in dense_metrics]),
                        "non_settling_rate": mean([1.0 - metrics.get("query_delivery_rate", 0.0) for metrics in dense_metrics]),
                    }
                if kind_out:
                    regime_out[kind] = kind_out
            if regime_out:
                pair_out[regime] = regime_out
        out[pair] = pair_out
    return out


def overall_ranking(
    final_pairs: list[str],
    confirmation_summary: dict[str, dict[str, Any]],
    holdout_summary: dict[str, dict[str, Any]],
    rerun_summary: dict[str, dict[str, Any]],
    ambiguity_summary: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    final_regimes = ANCHOR_REGIMES + VERIFICATION_REGIMES + sorted(rerun_summary) + sorted(ambiguity_summary)
    for pair in final_pairs:
        score = (
            pair_total(confirmation_summary, pair, ANCHOR_REGIMES)
            + pair_total(holdout_summary, pair, VERIFICATION_REGIMES)
            + pair_total(rerun_summary, pair, sorted(rerun_summary))
            + pair_total(ambiguity_summary, pair, sorted(ambiguity_summary))
        )
        std_total = math.sqrt(
            pair_total_std(confirmation_summary, pair, ANCHOR_REGIMES) ** 2
            + pair_total_std(holdout_summary, pair, VERIFICATION_REGIMES) ** 2
            + pair_total_std(rerun_summary, pair, sorted(rerun_summary)) ** 2
            + pair_total_std(ambiguity_summary, pair, sorted(ambiguity_summary)) ** 2
        )
        rows.append(
            {
                "pair": pair,
                "pair_label": PAIR_INFO[pair]["label"],
                "selector": PAIR_INFO[pair]["selector"],
                "contract": PAIR_INFO[pair]["contract"],
                "score": score,
                "std": std_total,
                "regimes": final_regimes,
            }
        )
    rows.sort(key=lambda row: row["score"], reverse=True)
    return rows


def outcome_label(
    rankings: list[dict[str, Any]],
    rerun_summary: dict[str, dict[str, Any]],
    ambiguity_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(rankings) < 2:
        return {"outcome": "insufficient_data", "winner": ""}
    first, second = rankings[0], rankings[1]
    rerun_regimes = sorted(rerun_summary) + sorted(ambiguity_summary)
    rerun_wins = 0
    for regime in rerun_regimes:
        source = ambiguity_summary if regime in ambiguity_summary else rerun_summary
        a = source.get(regime, {}).get(first["pair"], {}).get("composite", {}).get("mean", 0.0)
        b = source.get(regime, {}).get(second["pair"], {}).get("composite", {}).get("mean", 0.0)
        if a >= b:
            rerun_wins += 1
    resolved = bool(rerun_regimes) and first["score"] > second["score"] + max(first["std"], second["std"]) and rerun_wins == len(rerun_regimes)
    if resolved:
        return {"outcome": "resolved", "winner": first["pair"], "score_gap": first["score"] - second["score"]}
    return {"outcome": "unresolved", "winner": "", "score_gap": first["score"] - second["score"]}


def markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    complete_records: list[dict[str, Any]] = []
    for meta, run_dir in latest_v63_runs():
        if not meta or not is_complete_run(run_dir, meta["schedule"]):
            continue
        complete_records.append(summarize_run(run_dir, meta))

    pilot_records = [record for record in complete_records if record["schedule"] == "p"]
    screening_records = [record for record in complete_records if record["schedule"] == "s" and record["tag"] == ""]
    confirmation_records = [record for record in complete_records if record["schedule"] == "m" and record["tag"] == ""]
    holdout_records = [
        record
        for record in complete_records
        if record["schedule"] == "l" and record["tag"] == "" and record["seed"] == 4234 and record["regime"] in VERIFICATION_REGIMES
    ]
    rerun_records = [record for record in complete_records if record["schedule"] == "l" and record["tag"] == "rerun"]
    ambiguity_records = [record for record in complete_records if record["schedule"] == "l" and record["tag"] == "amb"]

    selected_settings = select_pilot_settings(pilot_records)
    screening_rankings = rank_pairs(screening_records)
    promoted_pairs = promote_screening_pairs(screening_rankings) if screening_rankings else []
    confirmation_rankings = rank_pairs(confirmation_records, promoted_pairs) if promoted_pairs else []
    top2_pairs = [row["pair"] for row in confirmation_rankings[:2]]
    confirmation_summary = summarize_by_regime_and_pair(confirmation_records, promoted_pairs)
    holdout_summary = summarize_by_regime_and_pair(holdout_records, top2_pairs) if top2_pairs else {}
    final_top2_pairs = pick_top_pairs(confirmation_summary, holdout_summary, top2_pairs) if top2_pairs else []
    rerun_regimes = choose_final_rerun_regimes(final_top2_pairs, confirmation_summary, holdout_summary) if final_top2_pairs else []
    rerun_summary = summarize_by_regime_and_pair(rerun_records, final_top2_pairs) if final_top2_pairs else {}
    ambiguity_summary = summarize_by_regime_and_pair(ambiguity_records, final_top2_pairs) if final_top2_pairs else {}
    extra_depth_summary = collect_extra_depth(final_top2_pairs) if final_top2_pairs else {}
    overall = (
        overall_ranking(final_top2_pairs, confirmation_summary, holdout_summary, rerun_summary, ambiguity_summary)
        if final_top2_pairs
        else []
    )
    outcome = outcome_label(overall, rerun_summary, ambiguity_summary) if overall else {"outcome": "insufficient_data", "winner": ""}

    summary_payload = {
        "budgets": {"p": 420, "s": 1134, "m": 2268, "l": 3024},
        "visible_gpu_count": visible_gpu_count(),
        "selected_settings": {
            pair: {"lr_multiplier": row["lr_multiplier"], "p_keep_prev": row["p_keep_prev"]}
            for pair, row in selected_settings.items()
        },
        "screening_rankings": screening_rankings,
        "promoted_pairs": promoted_pairs,
        "confirmation_rankings": confirmation_rankings,
        "top2_pairs_after_confirmation": top2_pairs,
        "top2_pairs_after_holdout": final_top2_pairs,
        "final_rerun_regimes": rerun_regimes,
        "extra_depth_summary": extra_depth_summary,
        "overall_ranking": overall,
        "outcome": outcome,
        "counts": {
            "p": len(pilot_records),
            "s": len(screening_records),
            "m": len(confirmation_records),
            "l_holdout": len(holdout_records),
            "l_rerun": len(rerun_records),
            "l_ambiguity": len(ambiguity_records),
        },
        "per_regime_confirmation": confirmation_summary,
        "per_regime_holdout": holdout_summary,
        "per_regime_rerun": rerun_summary,
        "per_regime_ambiguity": ambiguity_summary,
    }
    SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2, sort_keys=False), encoding="utf-8")

    lines: list[str] = []
    lines.append("# APSGNN v63: RSM-lite Contract Tie-Break")
    lines.append("")
    lines.append("## What Changed From v62")
    lines.append("")
    lines.append(
        "v63 stops slicing selector weights and instead tests whether temporal-credit and train-shallow/test-deep "
        "contracts can stabilize the unresolved `V` vs `VT-0.5` tie from v62."
    )
    lines.append("")
    lines.append("## Budgets")
    lines.append("")
    lines.extend([
        "- `P = 420`",
        "- `S = 1134`",
        "- `M = 2268`",
        "- `L = 3024`",
        f"- visible GPUs used: `{visible_gpu_count()}`",
        f"- rolling late-stage window: `{ROLLING_N}` evals",
    ])
    lines.append("")
    lines.append("## Calibration Settings")
    lines.append("")
    if selected_settings:
        rows = []
        for pair in sorted(selected_settings):
            info = selected_settings[pair]
            rows.append(
                [
                    PAIR_INFO[pair]["label"],
                    f"{info['lr_multiplier']:.2f}",
                    f"{info['p_keep_prev']:.2f}",
                    f"{info['summary']['composite']['mean']:.4f}",
                ]
            )
        lines.extend(markdown_table(["Pair", "LR x", "p_keep_prev", "Pilot Composite"], rows))
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Screening Summary")
    lines.append("")
    if screening_rankings:
        rows = []
        for row in screening_rankings:
            rows.append(
                [
                    row["pair_label"],
                    f"{row['summary']['dense_mean']['mean']:.4f}",
                    f"{row['summary']['last_val']['mean']:.4f}",
                    f"{row['summary']['last5_val_mean']['mean']:.4f}",
                    f"{row['summary']['best_to_last_drop']['mean']:.4f}",
                    f"{row['summary']['composite']['mean']:.4f}",
                ]
            )
        lines.extend(markdown_table(["Pair", "Dense", "Last", "Last5", "Drop", "Composite"], rows))
        lines.append("")
        lines.append(f"Promoted pairs: `{', '.join(promoted_pairs)}`")
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Confirmation Summary")
    lines.append("")
    if confirmation_rankings:
        rows = []
        for row in confirmation_rankings:
            rows.append(
                [
                    row["pair_label"],
                    f"{pair_total(confirmation_summary, row['pair'], ANCHOR_REGIMES):.4f}",
                    f"{row['summary']['dense_mean']['mean']:.4f}",
                    f"{row['summary']['last_val']['mean']:.4f}",
                    f"{row['summary']['best_to_last_drop']['mean']:.4f}",
                ]
            )
        lines.extend(markdown_table(["Pair", "Anchor Total", "Dense", "Last", "Drop"], rows))
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Holdout Verification")
    lines.append("")
    if final_top2_pairs and holdout_summary:
        rows = []
        for pair in final_top2_pairs:
            rows.append([PAIR_INFO[pair]["label"], f"{pair_total(holdout_summary, pair, VERIFICATION_REGIMES):.4f}"])
        lines.extend(markdown_table(["Pair", "Holdout Total"], rows))
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Extra Compute / Settling")
    lines.append("")
    if extra_depth_summary:
        rows = []
        for pair, pair_payload in extra_depth_summary.items():
            if "hmix" in pair_payload and "best" in pair_payload["hmix"] and "settle" in pair_payload["hmix"]["best"]:
                settle = pair_payload["hmix"]["best"]["settle"]
                rows.append(
                    [
                        PAIR_INFO[pair]["label"],
                        f"{settle['dense_mean']:.4f}",
                        f"{settle['settle_rate']:.4f}",
                        f"{settle['steps_to_settle']:.2f}",
                    ]
                )
        if rows:
            lines.extend(markdown_table(["Pair", "Hmix settle dense", "Settle Rate", "Steps"], rows))
        else:
            lines.append("- pending")
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Fresh Reruns")
    lines.append("")
    if rerun_summary:
        rows = []
        for regime in sorted(rerun_summary):
            for pair in final_top2_pairs:
                if pair not in rerun_summary[regime]:
                    continue
                summary = rerun_summary[regime][pair]
                rows.append(
                    [
                        regime,
                        PAIR_INFO[pair]["label"],
                        f"{summary['dense_mean']['mean']:.4f}",
                        f"{summary['last_val']['mean']:.4f}",
                        f"{summary['composite']['mean']:.4f}",
                    ]
                )
        lines.extend(markdown_table(["Regime", "Pair", "Dense", "Last", "Composite"], rows))
    else:
        lines.append("- pending")
    lines.append("")
    lines.append("## Ambiguity Breaker")
    lines.append("")
    if ambiguity_summary:
        rows = []
        for regime in sorted(ambiguity_summary):
            for pair in final_top2_pairs:
                if pair not in ambiguity_summary[regime]:
                    continue
                summary = ambiguity_summary[regime][pair]
                rows.append(
                    [
                        regime,
                        PAIR_INFO[pair]["label"],
                        f"{summary['dense_mean']['mean']:.4f}",
                        f"{summary['last_val']['mean']:.4f}",
                        f"{summary['composite']['mean']:.4f}",
                    ]
                )
        lines.extend(markdown_table(["Regime", "Pair", "Dense", "Last", "Composite"], rows))
    else:
        lines.append("- not triggered")
    lines.append("")
    lines.append("## Final Diagnosis")
    lines.append("")
    lines.append(f"- Outcome: `{outcome['outcome']}`")
    if overall:
        lines.append(f"- Top pair: `{overall[0]['pair']}` score `{overall[0]['score']:.4f}`")
        if len(overall) > 1:
            lines.append(f"- Runner-up: `{overall[1]['pair']}` score `{overall[1]['score']:.4f}`")
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
