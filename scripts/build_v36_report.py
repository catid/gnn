#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v36.json"
REPORT_PATH = REPORTS / "final_report_v36_sparse_f025_package_confirmation.md"

SELECTORS = {
    "visit_taskgrad_half": {"label": "VT-0.5"},
    "visit_taskgrad_half_agree_mutate_z00_m075_f025": {"label": "VT-0.5 CAG-z0.0-m0.75-f0.25"},
}

PHASES = {
    "core_xl": {"regime": "core", "writers": [2, 6, 10], "title": "Core-XL Fresh Confirmation"},
    "t1_xl": {"regime": "t1", "writers": [4, 8, 12, 14], "title": "T1-XL Fresh Confirmation"},
    "t2a_xl": {"regime": "t2a", "writers": [4, 8, 12, 14], "title": "T2a-XL Stress"},
    "t1r_xl": {"regime": "t1r", "writers": [4, 8, 12, 14], "title": "T1-XL Final Rerun"},
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": std(values)}


def latest_runs(prefix: str) -> list[tuple[Path, int]]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [(latest_by_seed[seed], seed) for seed in sorted(latest_by_seed)]


def is_complete_run(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists() or not (run_dir / "last.pt").exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != 4590:
        return False
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= 4590


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def summarize_run(run_dir: Path, selector: str, phase: str, seed: int) -> dict[str, Any]:
    spec = PHASES[phase]
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    record: dict[str, Any] = {
        "phase": phase,
        "selector": selector,
        "selector_label": SELECTORS[selector]["label"],
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
    }
    for writer in spec["writers"]:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in spec["writers"][1:]])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in spec["writers"][1:]])
    record["screen_composite"] = (
        0.45 * record["dense_mean"] + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]
    )
    return record


def phase_records(phase: str) -> list[dict[str, Any]]:
    regime = PHASES[phase]["regime"]
    records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        prefix = f"v36-{regime}-{selector}-32-xl"
        for run_dir, seed in latest_runs(prefix):
            if not is_complete_run(run_dir):
                continue
            records.append(summarize_run(run_dir, selector, phase, seed))
    return records


def summarize_phase(records: list[dict[str, Any]], phase: str) -> dict[str, dict[str, Any]]:
    writers = PHASES[phase]["writers"]
    out: dict[str, dict[str, Any]] = {}
    for selector in SELECTORS:
        selector_records = [record for record in records if record["selector"] == selector]
        if not selector_records:
            continue
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "last5_val_mean": mean_std([record["last5_val_mean"] for record in selector_records]),
            "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in selector_records]),
            "dense_mean": mean_std([record["dense_mean"] for record in selector_records]),
            "last_dense_mean": mean_std([record["last_dense_mean"] for record in selector_records]),
            "screen_composite": mean_std([record["screen_composite"] for record in selector_records]),
        }
        for writer in writers[1:]:
            out[selector][f"k{writer}"] = mean_std([record[f"k{writer}"] for record in selector_records])
            out[selector][f"last_k{writer}"] = mean_std([record[f"last_k{writer}"] for record in selector_records])
    return out


def fmt_stat(payload: dict[str, float]) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_table(records: list[dict[str, Any]]) -> str:
    headers = ["Phase", "Sel", "Seed", "Best", "Last", "Dense", "LastDense"]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["phase"], item["selector"], item["seed"])):
        rows.append(
            [
                PHASES[record["phase"]]["title"],
                record["selector_label"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['dense_mean']:.4f}",
                f"{record['last_dense_mean']:.4f}",
            ]
        )
    return markdown_table(headers, rows)


def phase_table(summary: dict[str, dict[str, Any]], phase: str) -> str:
    writers = PHASES[phase]["writers"]
    headers = ["Selector", "Best", "Last", "Dense", "LastDense", *[f"K{w}" for w in writers[1:]]]
    rows: list[list[str]] = []
    for selector in SELECTORS:
        payload = summary.get(selector)
        if not payload:
            continue
        row = [
            SELECTORS[selector]["label"],
            fmt_stat(payload["best_val"]),
            fmt_stat(payload["last_val"]),
            fmt_stat(payload["dense_mean"]),
            fmt_stat(payload["last_dense_mean"]),
        ]
        for writer in writers[1:]:
            row.append(fmt_stat(payload[f"k{writer}"]))
        rows.append(row)
    return markdown_table(headers, rows)


def plot_phase(summary: dict[str, dict[str, Any]], phase: str, path: Path) -> None:
    selectors = [selector for selector in SELECTORS if selector in summary]
    if not selectors:
        return
    labels = [SELECTORS[selector]["label"] for selector in selectors]
    metrics = [("best_val", "Best"), ("last_val", "Last"), ("dense_mean", "Dense")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.6 * len(metrics), 4.2))
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        means = [summary[selector][metric]["mean"] for selector in selectors]
        stds = [summary[selector][metric]["std"] for selector in selectors]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(PHASES[phase]["title"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        records = phase_records(phase)
        if not records:
            continue
        all_records.extend(records)
        phase_summaries[phase] = summarize_phase(records, phase)

    final_score: dict[str, float] = {}
    for selector in SELECTORS:
        total = 0.0
        for phase in ("core_xl", "t1_xl", "t2a_xl", "t1r_xl"):
            payload = phase_summaries.get(phase, {}).get(selector)
            if not payload:
                continue
            total += payload["last_val"]["mean"] + payload["dense_mean"]["mean"]
        final_score[selector] = total
    final_recommendation = max(final_score, key=final_score.get) if final_score else "visit_taskgrad_half"

    for phase, summary in phase_summaries.items():
        plot_phase(summary, phase, REPORTS / f"v36_{phase}_summary.png")

    payload = {
        "phase_summaries": phase_summaries,
        "records": all_records,
        "final_score": final_score,
        "final_recommendation": final_recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# APSGNN v36: Sparse F0.25 Package Confirmation",
        "",
        "## What Changed",
        "",
        "v36 is the fresh-seed confirmation round for the more aggressive sparse mutation package that tied the v34 screen but did not get the v35 confirmation slot.",
        "It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75-f0.25` on fresh Core-XL, T1-XL, T2a-XL, plus a final T1 rerun block with both best and last checkpoint evals.",
        "",
        "## Completed Runs",
        "",
        run_table(all_records),
        "",
    ]
    for phase in ("core_xl", "t1_xl", "t2a_xl", "t1r_xl"):
        summary = phase_summaries.get(phase)
        if not summary:
            continue
        lines.extend([f"## {PHASES[phase]['title']}", "", phase_table(summary, phase), ""])
    lines.extend(
        [
            "## Current Recommendation",
            "",
            f"Current best supported default: `{SELECTORS[final_recommendation]['label']}`.",
            "",
            "## Outputs",
            "",
            f"- Summary JSON: [summary_metrics_v36.json]({SUMMARY_PATH})",
            f"- Report: [final_report_v36_sparse_f025_package_confirmation.md]({REPORT_PATH})",
            f"- Plots: [reports]({REPORTS})",
            "",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
