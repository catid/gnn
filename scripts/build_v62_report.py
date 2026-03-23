#!/usr/bin/env python3
from __future__ import annotations

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
SUMMARY_PATH = REPORTS / "summary_metrics_v62.json"
REPORT_PATH = REPORTS / "final_report_v62_static_selector_tiebreak.md"

BASE_LR = 2.0e-4
ROLLING_N = 5
EXPECTED_TRAIN_STEPS = {"p": 420, "m": 2268, "l": 3024, "xl": 3780}

ARM_INFO = {
    "visitonly": {"family": "visitonly", "label": "V"},
    "visit_taskgrad_half": {"family": "visit_taskgrad_half", "label": "VT-0.5"},
}

ARM_SETTINGS = {
    "visitonly": "Static `z(task_visits)`",
    "visit_taskgrad_half": "Static `z(task_visits) + 0.5*z(task_grad)`",
}

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

ALL_REGIMES = ["core", "t1", "t1r", "t2a", "t2b", "t2c", "hmid", "hmix"]
PILOT_REGIMES = ["core", "t1", "t2a", "hmix"]
ANCHOR_REGIMES = ["core", "t1", "t2a", "hmix"]

RUN_RE = re.compile(
    r"v62-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|m|l|xl)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
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


def latest_v62_runs() -> list[tuple[dict[str, str], Path]]:
    latest: dict[tuple[str, str, str, str, str], Path] = {}
    for candidate in sorted(RUNS.glob("*-v62-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        key = (meta["regime"], meta["arm"], meta["schedule"], meta["tag"], meta["seed"])
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


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


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
    writers = REGIME_WRITERS[meta["regime"]]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    record: dict[str, Any] = {
        "run": run_dir.name,
        "regime": meta["regime"],
        "arm": meta["arm"],
        "arm_label": ARM_INFO[meta["arm"]]["label"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"]),
        "tag": meta["tag"],
        "config_name": f"configs/v62_{meta['regime']}_{meta['arm']}_32_{meta['schedule']}.yaml",
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "train_steps": int(config["train"]["train_steps"]),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "delivery_rate": float(last.get("val/delivery_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/home_to_out_rate", 0.0)),
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in dense_writers])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in dense_writers])
    record["composite"] = 0.45 * record["dense_mean"] + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]
    record["score"] = record["last_val"] + record["dense_mean"]
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


def candidate_key(record: dict[str, Any]) -> str:
    return f"{record['arm']}@lr{record['lr_multiplier']:.1f}"


def rank_candidates(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(candidate_key(record), []).append(record)
    per_arm: dict[str, list[dict[str, Any]]] = {}
    for key, candidate_records in grouped.items():
        row = {
            "candidate": key,
            "arm": candidate_records[0]["arm"],
            "label": candidate_records[0]["arm_label"],
            "lr_multiplier": candidate_records[0]["lr_multiplier"],
            "summary": summarize_group(candidate_records),
        }
        per_arm.setdefault(row["arm"], []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for arm, rows in per_arm.items():
        rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
        selected[arm] = rows[0]
    return selected


def summarize_by_regime_and_arm(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted({record["regime"] for record in records}):
        out[regime] = {}
        for arm in sorted({record["arm"] for record in records if record["regime"] == regime}):
            regime_arm_records = [record for record in records if record["regime"] == regime and record["arm"] == arm]
            out[regime][arm] = summarize_group(regime_arm_records)
    return out


def regime_margin(summary: dict[str, dict[str, Any]], regime: str) -> float:
    visitonly = summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
    vt_half = summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
    return abs(visitonly - vt_half)


def regime_winner(summary: dict[str, dict[str, Any]], regime: str) -> str:
    visitonly = summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
    vt_half = summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
    return "visitonly" if visitonly >= vt_half else "visit_taskgrad_half"


def pooled_scores(summary: dict[str, dict[str, Any]], regimes: list[str]) -> dict[str, float]:
    totals = {"visitonly": 0.0, "visit_taskgrad_half": 0.0}
    for regime in regimes:
        for arm in totals:
            totals[arm] += summary.get(regime, {}).get(arm, {}).get("composite", {}).get("mean", 0.0)
    return totals


def pooled_uncertainty(summary: dict[str, dict[str, Any]], regimes: list[str]) -> float:
    variance = 0.0
    for regime in regimes:
        visitonly_std = summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("std", 0.0)
        vt_half_std = summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("std", 0.0)
        variance += max(visitonly_std, vt_half_std) ** 2
    return math.sqrt(variance)


def winner_rows(summary: dict[str, dict[str, Any]], regimes: list[str]) -> list[list[str]]:
    rows: list[list[str]] = []
    for regime in regimes:
        visitonly = summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
        vt_half = summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
        winner = "V" if visitonly >= vt_half else "VT-0.5"
        rows.append([regime, winner, f"{visitonly:.4f}", f"{vt_half:.4f}"])
    return rows


def should_trigger_xl(main_summary: dict[str, dict[str, Any]], anchor_summary: dict[str, dict[str, Any]]) -> bool:
    main_scores = pooled_scores(main_summary, ALL_REGIMES)
    anchor_scores = pooled_scores(anchor_summary, ANCHOR_REGIMES)
    main_diff = abs(main_scores["visitonly"] - main_scores["visit_taskgrad_half"])
    anchor_diff = abs(anchor_scores["visitonly"] - anchor_scores["visit_taskgrad_half"])
    unclear = main_diff <= pooled_uncertainty(main_summary, ALL_REGIMES)
    if anchor_summary:
        unclear = unclear or anchor_diff <= pooled_uncertainty(anchor_summary, ANCHOR_REGIMES)
    contradictory = False
    for regime in ("core", "t2a", "hmix"):
        if regime in anchor_summary and regime_winner(main_summary, regime) != regime_winner(anchor_summary, regime):
            contradictory = True
    universal_flip = False
    if anchor_summary:
        universal_flip = (
            ("visitonly" if main_scores["visitonly"] >= main_scores["visit_taskgrad_half"] else "visit_taskgrad_half")
            != (
                "visitonly"
                if anchor_scores["visitonly"] >= anchor_scores["visit_taskgrad_half"]
                else "visit_taskgrad_half"
            )
        )
    return unclear or contradictory or universal_flip


def choose_ambiguity_regimes(main_summary: dict[str, dict[str, Any]], anchor_summary: dict[str, dict[str, Any]]) -> list[str]:
    candidates: list[tuple[int, float, str]] = []
    for regime in ALL_REGIMES:
        contradictory = 0
        if regime in anchor_summary and regime_winner(main_summary, regime) != regime_winner(anchor_summary, regime):
            contradictory = -1
        margin = regime_margin(anchor_summary if regime in anchor_summary else main_summary, regime)
        if regime in anchor_summary:
            margin = min(margin, regime_margin(main_summary, regime))
        candidates.append((contradictory, margin, regime))
    candidates.sort(key=lambda row: (row[0], row[1], row[2]))
    return [regime for _, _, regime in candidates[:2]]


def classify_outcome(
    main_summary: dict[str, dict[str, Any]],
    anchor_summary: dict[str, dict[str, Any]],
    xl_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    main_scores = pooled_scores(main_summary, ALL_REGIMES)
    anchor_scores = pooled_scores(anchor_summary, ANCHOR_REGIMES)
    xl_regimes = sorted(xl_summary)
    xl_scores = pooled_scores(xl_summary, xl_regimes) if xl_regimes else {"visitonly": 0.0, "visit_taskgrad_half": 0.0}
    total_scores = {
        "visitonly": main_scores["visitonly"] + anchor_scores["visitonly"] + xl_scores["visitonly"],
        "visit_taskgrad_half": main_scores["visit_taskgrad_half"] + anchor_scores["visit_taskgrad_half"] + xl_scores["visit_taskgrad_half"],
    }
    winner = "visitonly" if total_scores["visitonly"] >= total_scores["visit_taskgrad_half"] else "visit_taskgrad_half"
    winner_label = ARM_INFO[winner]["label"]
    loser = "visit_taskgrad_half" if winner == "visitonly" else "visitonly"
    holdout_ok = True
    for regime in ("hmid", "hmix"):
        winner_score = main_summary.get(regime, {}).get(winner, {}).get("composite", {}).get("mean", 0.0)
        loser_score = main_summary.get(regime, {}).get(loser, {}).get("composite", {}).get("mean", 0.0)
        winner_std = main_summary.get(regime, {}).get(winner, {}).get("composite", {}).get("std", 0.0)
        if winner_score + winner_std < loser_score:
            holdout_ok = False
    anchor_ok = not anchor_summary or (
        anchor_scores[winner] >= anchor_scores[loser] or abs(anchor_scores[winner] - anchor_scores[loser]) <= pooled_uncertainty(anchor_summary, ANCHOR_REGIMES)
    )
    consistent_split = True
    for regime in ANCHOR_REGIMES:
        if regime in anchor_summary and regime_winner(main_summary, regime) != regime_winner(anchor_summary, regime):
            consistent_split = False
    if winner == "visitonly" and main_scores["visitonly"] >= main_scores["visit_taskgrad_half"] and anchor_ok and holdout_ok:
        return {"outcome": "A", "winner": winner_label, "scores": total_scores}
    if winner == "visit_taskgrad_half" and main_scores["visit_taskgrad_half"] >= main_scores["visitonly"] and anchor_ok and holdout_ok:
        return {"outcome": "B", "winner": winner_label, "scores": total_scores}
    if consistent_split:
        return {"outcome": "C", "winner": "regime-dependent", "scores": total_scores}
    return {"outcome": "D", "winner": "unresolved", "scores": total_scores}


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return ""
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join([header, sep, body])


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    records = []
    for meta, run_dir in latest_v62_runs():
        if not meta or not is_complete_run(run_dir, meta["schedule"]):
            continue
        records.append(summarize_run(run_dir, meta))

    pilot_records = [record for record in records if record["schedule"] == "p" and record["regime"] in PILOT_REGIMES]
    main_records = [record for record in records if record["schedule"] == "m" and record["regime"] in ALL_REGIMES]
    anchor_records = [record for record in records if record["schedule"] == "l" and record["regime"] in ANCHOR_REGIMES]
    xl_records = [record for record in records if record["schedule"] == "xl"]

    pilot_rankings = rank_candidates(pilot_records)
    main_summary = summarize_by_regime_and_arm(main_records)
    anchor_summary = summarize_by_regime_and_arm(anchor_records)
    xl_summary = summarize_by_regime_and_arm(xl_records)

    main_scores = pooled_scores(main_summary, ALL_REGIMES)
    anchor_scores = pooled_scores(anchor_summary, ANCHOR_REGIMES)
    ambiguity = {
        "triggered": should_trigger_xl(main_summary, anchor_summary),
        "candidate_regimes": choose_ambiguity_regimes(main_summary, anchor_summary),
    }
    outcome = classify_outcome(main_summary, anchor_summary, xl_summary)

    selected_lrs = {
        arm: payload["lr_multiplier"]
        for arm, payload in pilot_rankings.items()
    }
    payload = {
        "rollup_window": ROLLING_N,
        "visible_gpu_count": visible_gpu_count(),
        "pilot_rankings": pilot_rankings,
        "selected_lrs": selected_lrs,
        "main_summary": main_summary,
        "anchor_summary": anchor_summary,
        "xl_summary": xl_summary,
        "main_scores": main_scores,
        "anchor_scores": anchor_scores,
        "ambiguity": ambiguity,
        "outcome": outcome,
        "config_names": sorted({record["config_name"] for record in records}),
        "records": records,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    pilot_rows = []
    for arm in ("visitonly", "visit_taskgrad_half"):
        if arm not in pilot_rankings:
            continue
        choice = pilot_rankings[arm]
        summary = choice["summary"]
        pilot_rows.append(
            [
                choice["label"],
                f"{choice['lr_multiplier']:.1f}",
                f"{summary['composite']['mean']:.4f}",
                f"{summary['dense_mean']['mean']:.4f}",
                f"{summary['last_val']['mean']:.4f}",
            ]
        )

    main_rows = winner_rows(main_summary, ALL_REGIMES)
    anchor_rows = winner_rows(anchor_summary, ANCHOR_REGIMES)
    xl_rows = winner_rows(xl_summary, sorted(xl_summary)) if xl_summary else []

    report_lines = [
        "# APSGNN v62: Static Selector Tie-Break",
        "",
        "## What Changed From v61",
        "",
        "v62 removes all gate logic and all adaptive-selector branches. This round exists solely to settle the static `V` vs `VT-0.5` tie-break on the same 32-leaf APSGNN family, with the full 8-regime matrix plus long-rerun anchors and an XL breaker only if the result stays ambiguous.",
        "",
        "## Budgets",
        "",
        "- `P = 420`",
        "- `M = 2268`",
        "- `L = 3024`",
        "- `XL = 3780`",
        f"- visible GPUs used: `{payload['visible_gpu_count']}`",
        f"- rolling late-stage window: `{ROLLING_N}` evals",
        "",
        "## Exact Regimes",
        "",
        "- `Core`: writers=2, start_pool=2, query_ttl=2..3, rollout=12, eval densities=2/6/10",
        "- `T1`: writers=4, start_pool=2, query_ttl=2..3, rollout=12, eval densities=4/8/12/14",
        "- `T1r`: writers=4, start_pool=2, query_ttl=2..3, rollout=12, eval densities=4/8/12/14",
        "- `T2a`: writers=4, start_pool=1, query_ttl=2..3, rollout=12, eval densities=4/8/12/14",
        "- `T2b`: writers=4, start_pool=2, query_ttl=2..2, rollout=12, eval densities=4/8/12/14",
        "- `T2c`: writers=6, start_pool=2, query_ttl=2..3, rollout=12, eval densities=6/10/14/16",
        "- `Hmid`: writers=3, start_pool=2, query_ttl=2..3, rollout=12, eval densities=3/7/11",
        "- `Hmix`: writers=3, start_pool=1, query_ttl=2..3, rollout=12, eval densities=3/7/11",
        "",
        "## Calibration Summary",
        "",
        format_table(
            ["Selector", "LR x", "Composite", "Dense", "Last"],
            pilot_rows,
        ),
        "",
        "## Completed Runs",
        "",
        format_table(
            ["Schedule", "Runs"],
            [
                ["P", str(len(pilot_records))],
                ["M", str(len(main_records))],
                ["L", str(len(anchor_records))],
                ["XL", str(len(xl_records))],
            ],
        ),
        "",
        "## Pooled All-8-Regime Summary",
        "",
        format_table(
            ["Selector", "Main Total"],
            [
                ["V", f"{main_scores['visitonly']:.6f}"],
                ["VT-0.5", f"{main_scores['visit_taskgrad_half']:.6f}"],
            ],
        ),
        "",
        "## Per-Regime Winners On M",
        "",
        format_table(["Regime", "Winner", "V Composite", "VT-0.5 Composite"], main_rows),
        "",
        "## Long-Rerun Anchor Winners On L",
        "",
        format_table(["Regime", "Winner", "V Composite", "VT-0.5 Composite"], anchor_rows),
        "",
        "## Ambiguity Breaker",
        "",
        f"- Triggered: `{ambiguity['triggered']}`",
        f"- Candidate regimes: `{', '.join(ambiguity['candidate_regimes'])}`",
    ]
    if xl_rows:
        report_lines.extend(
            [
                "",
                format_table(["Regime", "Winner", "V Composite", "VT-0.5 Composite"], xl_rows),
            ]
        )
    report_lines.extend(
        [
            "",
            "## Coverage And Split Diagnostics",
            "",
            "- Coverage deltas and split predictiveness were aggregated directly from `coverage_summary.json` for each run.",
            "- The main diagnostic question is whether the selector gap is already visible by steps `10/50/100/200` or only emerges through later split usefulness and stability.",
            "",
            "## Final Diagnosis",
            "",
            f"- Outcome: `{outcome['outcome']}`",
            f"- Winner: `{outcome['winner']}`",
            f"- Main totals: `V={main_scores['visitonly']:.6f}`, `VT-0.5={main_scores['visit_taskgrad_half']:.6f}`",
            f"- Anchor totals: `V={anchor_scores['visitonly']:.6f}`, `VT-0.5={anchor_scores['visit_taskgrad_half']:.6f}`",
            "",
            "## Exact Configs Used",
            "",
        ]
    )
    for config_name in payload["config_names"]:
        report_lines.append(f"- `{config_name}`")
    report_lines.extend(
        [
            "",
            f"- Summary JSON: [summary_metrics_v62.json]({SUMMARY_PATH})",
            f"- Report: [final_report_v62_static_selector_tiebreak.md]({REPORT_PATH})",
        ]
    )
    REPORT_PATH.write_text("\n".join(line for line in report_lines if line is not None) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
