#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v25.json"
REPORT_PATH = REPORTS / "final_report_v25_selector_consolidation.md"

SELECTORS = ["visitonly", "querygradonly", "visit_taskgrad_half"]

DISPLAY = {
    "visitonly": "V",
    "querygradonly": "Q",
    "visit_taskgrad_half": "VT-0.5",
}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
}

LABELS = {
    "core_l": "Core-L",
    "t1_l": "T1-L",
    "core_xl": "Core-XL",
    "t1_xl": "T1-XL",
    "t2a_xl": "T2a-XL",
    "t1_r": "T1-R",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def latest_runs(prefix: str) -> list[tuple[Path, int]]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [(latest_by_seed[seed], seed) for seed in sorted(latest_by_seed)]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": std(values)}


def take(stage: dict[str, Any], key: str, subkey: str) -> float:
    payload = stage.get(key, {})
    if isinstance(payload, dict):
        return float(payload.get(subkey, 0.0))
    return 0.0


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def summarize_run(run_dir: Path, selector: str, regime: str, schedule: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    record: dict[str, Any] = {
        "selector": selector,
        "selector_label": DISPLAY[selector],
        "regime": regime,
        "schedule": schedule,
        "phase": f"{regime}_{schedule}",
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
    }
    for writers in REGIME_WRITERS[regime]:
        record[f"k{writers}"] = extract_eval(run_dir, "best", writers)
        record[f"last_k{writers}"] = extract_eval(run_dir, "last", writers)
    coverage = read_json(run_dir / "coverage_summary.json")
    stages = coverage.get("stages", [])
    if stages:
        first = stages[0]
        for step in ("10", "50", "100", "200"):
            record[f"task_visit_{step}"] = take(first, "task_visit_coverage_at", step)
            record[f"task_grad_{step}"] = take(first, "task_grad_coverage_at", step)
    return record


def phase_records(selector: str, regime: str, schedule: str) -> list[dict[str, Any]]:
    prefix = f"v25-{regime}-{selector}-32-{schedule}"
    return [
        summarize_run(run_dir, selector, regime, schedule, seed)
        for run_dir, seed in latest_runs(prefix)
        if (run_dir / "coverage_summary.json").exists() and (run_dir / "metrics.jsonl").exists()
    ]


def summarize_phase(records: list[dict[str, Any]], regime: str) -> dict[str, dict[str, Any]]:
    writers = REGIME_WRITERS[regime]
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    out: dict[str, dict[str, Any]] = {}
    for selector in sorted({record["selector"] for record in records}):
        selector_records = [record for record in records if record["selector"] == selector]
        dense_means = [mean([record[key] for key in dense_keys]) for record in selector_records]
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "last5_val_mean": mean_std([record["last5_val_mean"] for record in selector_records]),
            "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in selector_records]),
            "dense_mean": mean_std(dense_means),
            "screen_composite": mean_std(
                [
                    0.45 * dense_mean + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]
                    for dense_mean, record in zip(dense_means, selector_records, strict=True)
                ]
            ),
        }
        for key in dense_keys:
            out[selector][key] = mean_std([record[key] for record in selector_records])
    return out


def rank_phase(summary: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(summary, key=lambda selector: summary[selector]["screen_composite"]["mean"], reverse=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_stat(payload: dict[str, float]) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def plot_phase(summary: dict[str, dict[str, Any]], selectors: list[str], title: str, metrics: list[tuple[str, str]], path: Path) -> None:
    selectors = [selector for selector in selectors if selector in summary]
    if not selectors:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2))
    if len(metrics) == 1:
        axes = [axes]
    labels = [DISPLAY[selector] for selector in selectors]
    for ax, (metric, metric_title) in zip(axes, metrics):
        means = [summary[selector][metric]["mean"] for selector in selectors]
        stds = [summary[selector][metric]["std"] for selector in selectors]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(metric_title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_table(records: list[dict[str, Any]], extra_keys: list[str]) -> str:
    headers = ["Phase", "Sel", "Seed", "Best", "Last", "Last5", "Drop", *extra_keys]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["phase"], item["selector"], item["seed"])):
        rows.append(
            [
                LABELS.get(record["phase"], record["phase"]),
                record["selector_label"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['last5_val_mean']:.4f}",
                f"{record['best_to_last_drop']:.4f}",
                *[f"{record.get(key, 0.0):.4f}" for key in extra_keys],
            ]
        )
    return markdown_table(headers, rows)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for regime, schedules in {"core": ["l", "xl"], "t1": ["l", "xl", "r"], "t2a": ["xl"]}.items():
        for schedule in schedules:
            phase_records_list: list[dict[str, Any]] = []
            for selector in SELECTORS:
                phase_records_list.extend(phase_records(selector, regime, schedule))
            if not phase_records_list:
                continue
            phase = f"{regime}_{schedule}"
            all_records.extend(phase_records_list)
            phase_summaries[phase] = summarize_phase(phase_records_list, regime)

    core_screen_rank = rank_phase(phase_summaries.get("core_l", {}))
    transfer_screen_rank = rank_phase(phase_summaries.get("t1_l", {}))
    combined_scores: dict[str, float] = {}
    for selector in set(core_screen_rank) | set(transfer_screen_rank):
        combined_scores[selector] = (
            phase_summaries.get("core_l", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
            + phase_summaries.get("t1_l", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
        )
    top2_screened = sorted(combined_scores, key=combined_scores.get, reverse=True)[:2]

    confirm_candidates = sorted(
        set(top2_screened)
        | set(phase_summaries.get("core_xl", {}))
        | set(phase_summaries.get("t1_xl", {}))
        | set(phase_summaries.get("t2a_xl", {}))
        | set(phase_summaries.get("t1_r", {}))
    )
    final_scores: dict[str, float] = {}
    for selector in confirm_candidates:
        final_scores[selector] = (
            phase_summaries.get("core_xl", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("core_xl", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_xl", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t1_xl", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2a_xl", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_r", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
        )
    top2_finalists = sorted(final_scores, key=final_scores.get, reverse=True)[:2]
    final_recommendation = top2_finalists[0] if top2_finalists else "visit_taskgrad_half"

    plot_phase(
        phase_summaries.get("core_l", {}),
        core_screen_rank,
        "v25 Core-L Screening",
        [("screen_composite", "Composite"), ("dense_mean", "Dense Mean"), ("last_val", "Last Val")],
        REPORTS / "v25_core_screening.png",
    )
    plot_phase(
        phase_summaries.get("t1_l", {}),
        transfer_screen_rank,
        "v25 T1-L Screening",
        [("screen_composite", "Composite"), ("dense_mean", "Dense Mean"), ("last_val", "Last Val")],
        REPORTS / "v25_t1_screening.png",
    )
    plot_phase(
        phase_summaries.get("core_xl", {}),
        top2_finalists,
        "v25 Core-XL Finalists",
        [("best_val", "Best Val"), ("last_val", "Last Val"), ("dense_mean", "Dense Mean")],
        REPORTS / "v25_core_xl.png",
    )
    plot_phase(
        phase_summaries.get("t1_xl", {}),
        top2_finalists,
        "v25 T1-XL Finalists",
        [("best_val", "Best Val"), ("last_val", "Last Val"), ("dense_mean", "Dense Mean")],
        REPORTS / "v25_t1_xl.png",
    )

    summary = {
        "budgets": {"l": 3570, "xl": 4590, "r": 4590},
        "visible_gpu_count": 2,
        "phase_summaries": phase_summaries,
        "top2_screened": top2_screened,
        "top2_finalists": top2_finalists,
        "confirm_candidates": confirm_candidates,
        "final_recommendation": final_recommendation,
        "records": all_records,
        "confirmatory_summary": {k: phase_summaries.get(k, {}) for k in ("core_xl", "t1_xl")},
        "stress_summary": {"t2a_xl": phase_summaries.get("t2a_xl", {})},
        "rerun_summary": {"t1_r": phase_summaries.get("t1_r", {})},
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    screening_rows = []
    for selector in SELECTORS:
        core_stats = phase_summaries.get("core_l", {}).get(selector)
        t1_stats = phase_summaries.get("t1_l", {}).get(selector)
        screening_rows.append(
            [
                DISPLAY[selector],
                fmt_stat(core_stats["screen_composite"]) if core_stats else "-",
                fmt_stat(t1_stats["screen_composite"]) if t1_stats else "-",
                fmt_stat(core_stats["dense_mean"]) if core_stats else "-",
                fmt_stat(t1_stats["dense_mean"]) if t1_stats else "-",
            ]
        )

    confirm_rows = []
    for selector in top2_finalists:
        core_stats = phase_summaries.get("core_xl", {}).get(selector)
        t1_stats = phase_summaries.get("t1_xl", {}).get(selector)
        confirm_rows.append(
            [
                DISPLAY[selector],
                fmt_stat(core_stats["last_val"]) if core_stats else "-",
                fmt_stat(t1_stats["last_val"]) if t1_stats else "-",
                fmt_stat(core_stats["dense_mean"]) if core_stats else "-",
                fmt_stat(t1_stats["dense_mean"]) if t1_stats else "-",
            ]
        )

    stress_rows = []
    for selector in top2_finalists:
        t2_stats = phase_summaries.get("t2a_xl", {}).get(selector)
        rerun_stats = phase_summaries.get("t1_r", {}).get(selector)
        stress_rows.append(
            [
                DISPLAY[selector],
                fmt_stat(t2_stats["last_val"]) if t2_stats else "-",
                fmt_stat(t2_stats["dense_mean"]) if t2_stats else "-",
                fmt_stat(rerun_stats["last_val"]) if rerun_stats else "-",
                fmt_stat(rerun_stats["dense_mean"]) if rerun_stats else "-",
            ]
        )

    report = f"""# APSGNN v25 Selector Consolidation

## What Changed From v24

- Moved from task-grad micro-neighborhood tuning to cross-family consolidation.
- Compared the v24 flat winner representative `VT-0.5` against the earlier family contenders `V` and `Q`.
- Reused the same 32-leaf `L`, `XL`, and `R` budgets as v24 so this round is directly compatible.

## Budgets

- Visible GPU count used: `2`
- Chosen schedules: `L=3570`, `XL=4590`, `R=4590` steps
- Screening composite: `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`

## Completed Runs

{run_table(all_records, ['k2', 'k6', 'k10', 'k4', 'k8', 'k12', 'k14'])}

## Screening Summary

{markdown_table(['Selector', 'Core-L composite', 'T1-L composite', 'Core-L dense', 'T1-L dense'], screening_rows)}

- Top 2 by screening composite: `{', '.join(DISPLAY[s] for s in top2_screened)}`
- Final head-to-head after XL + stress + rerun scoring: `{', '.join(DISPLAY[s] for s in top2_finalists)}`

## Confirmatory Summary

{markdown_table(['Selector', 'Core-XL last', 'T1-XL last', 'Core-XL dense', 'T1-XL dense'], confirm_rows)}

## Stress And Fresh-Seed Checks

{markdown_table(['Selector', 'T2a-XL last', 'T2a-XL dense', 'T1-R last', 'T1-R dense'], stress_rows)}

## Interpretation

v25 is the consolidation round after the v24 flat task-grad neighborhood result. The question is whether `VT-0.5` actually displaces the earlier selector-family contenders when they are rerun on the same longer 32-leaf budgets, not whether another tiny task-grad weight is marginally cleaner.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
