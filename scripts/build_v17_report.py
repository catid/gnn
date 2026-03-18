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
REPORTS.mkdir(exist_ok=True)


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


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def extract_eval(run_dir: Path, writers: int) -> dict[str, float]:
    payload = read_json(run_dir / f"eval_best_k{writers}.json")
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def summarize_core_run(run_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    return {
        "arm": arm,
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "k2": extract_eval(run_dir, 2)["query_accuracy"],
        "k6": extract_eval(run_dir, 6)["query_accuracy"],
        "k10": extract_eval(run_dir, 10)["query_accuracy"],
        "best_query_first_hop": float(best["val/query_first_hop_home_rate"]),
    }


def summarize_transfer_run(run_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    return {
        "arm": arm,
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "k4": extract_eval(run_dir, 4)["query_accuracy"],
        "k8": extract_eval(run_dir, 8)["query_accuracy"],
        "k12": extract_eval(run_dir, 12)["query_accuracy"],
        "k14": extract_eval(run_dir, 14)["query_accuracy"],
    }


def summarize(records: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({record["arm"] for record in records}):
        arm_records = [record for record in records if record["arm"] == arm]
        out[arm] = {"count": len(arm_records)}
        for metric in metrics:
            out[arm][metric] = mean_std([record[metric] for record in arm_records])
    return out


def plot(summary: dict[str, Any], metric_names: list[tuple[str, str]], order: list[str], labels: list[str], path: Path) -> None:
    present = [(arm, label) for arm, label in zip(order, labels) if arm in summary]
    if not present:
        return
    arms = [arm for arm, _ in present]
    axis_labels = [label for _, label in present]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.5 * len(metric_names), 4.5), constrained_layout=True)
    if len(metric_names) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metric_names):
        means = [summary[arm][metric]["mean"] for arm in arms]
        stds = [summary[arm][metric]["std"] for arm in arms]
        ax.bar(axis_labels, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    core_records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("querygrad", "v10-utility-querygrad-longplus"),
        ("visitonly", "v17-utility-visitonly-longplus"),
        ("querygradonly", "v17-utility-querygradonly-longplus"),
    ):
        for run_dir, seed in latest_runs(prefix):
            core_records.append(summarize_core_run(run_dir, arm, seed))

    transfer_records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("querygrad_h1", "v10-transfer-h1-utility-querygrad-longplus"),
        ("visitonly_h1", "v17-transfer-h1-utility-visitonly-longplus"),
        ("querygradonly_h1", "v17-transfer-h1-utility-querygradonly-longplus"),
    ):
        for run_dir, seed in latest_runs(prefix):
            transfer_records.append(summarize_transfer_run(run_dir, arm, seed))

    core_summary = summarize(core_records, ["best_val", "last_val", "k2", "k6", "k10", "best_query_first_hop"])
    transfer_summary = summarize(transfer_records, ["best_val", "last_val", "k4", "k8", "k12", "k14"])

    plot(
        core_summary,
        [("best_val", "Best Val"), ("k6", "K6"), ("k10", "K10")],
        ["querygrad", "visitonly", "querygradonly"],
        ["Querygrad", "Visit Only", "Querygrad Only"],
        REPORTS / "v17_core_component_ablation.png",
    )
    plot(
        transfer_summary,
        [("k4", "K4"), ("k8", "K8"), ("k12", "K12"), ("k14", "K14")],
        ["querygrad_h1", "visitonly_h1", "querygradonly_h1"],
        ["Querygrad", "Visit Only", "Querygrad Only"],
        REPORTS / "v17_h1_component_ablation.png",
    )

    payload = {
        "core_summary": core_summary,
        "core_runs": core_records,
        "transfer_summary": transfer_summary,
        "transfer_runs": transfer_records,
    }
    (REPORTS / "summary_metrics_v17.json").write_text(json.dumps(payload, indent=2))

    core_rows = [
        [
            arm,
            str(core_summary[arm]["count"]),
            fmt_stat(core_summary[arm]["best_val"]),
            fmt_stat(core_summary[arm]["last_val"]),
            fmt_stat(core_summary[arm]["k2"]),
            fmt_stat(core_summary[arm]["k6"]),
            fmt_stat(core_summary[arm]["k10"]),
        ]
        for arm in ("querygrad", "visitonly", "querygradonly")
    ]
    transfer_rows = [
        [
            arm,
            str(transfer_summary[arm]["count"]),
            fmt_stat(transfer_summary[arm]["best_val"]),
            fmt_stat(transfer_summary[arm]["k4"]),
            fmt_stat(transfer_summary[arm]["k8"]),
            fmt_stat(transfer_summary[arm]["k12"]),
            fmt_stat(transfer_summary[arm]["k14"]),
        ]
        for arm in ("querygrad_h1", "visitonly_h1", "querygradonly_h1")
        if arm in transfer_summary
    ]
    transfer_table = (
        markdown_table(['Arm', 'Count', 'Best Val', 'K4', 'K8', 'K12', 'K14'], transfer_rows)
        if transfer_rows
        else "No new H1 ablation runs were completed in this stopping-point report. The existing `querygrad_h1` reference remains the transfer anchor."
    )
    report = f"""# APSGNN v17 Querygrad Component Ablation

## What Changed

This round keeps the selective-growth architecture fixed and tests only the utility selector components behind the current `querygrad` default.

New additive mechanism:

- `utility_visit_weight`: explicit weight on the base task-visit term, so `visit-only` and `querygrad-only` are both honest score variants.

Arms:

- `querygrad`: existing v10 reference, `visit + query_grad`
- `visitonly`: `visit` only
- `querygradonly`: `query_grad` only

Core regime stays on the long selective benchmark with `writers_per_episode=2` and eval at `2/6/10`. Transfer regime stays on H1 with training at `4` writers and eval at `4/8/12/14`.

## Core Summary

{markdown_table(['Arm', 'Count', 'Best Val', 'Last Val', 'K2', 'K6', 'K10'], core_rows)}

## H1 Transfer Summary

{transfer_table}

## Interpretation

`querygrad` still looks like the best default once the score is decomposed cleanly. `visitonly` tests whether simple traffic concentration is already enough; `querygradonly` tests whether the query-side signal alone is enough. If both fall below `querygrad`, the selector really is using a useful combination rather than one redundant term.

In practice this round is aimed at validating the current score, not replacing the model family. The right next step after this should depend on whether the better challenger is `visitonly` or `querygradonly`.

## Outputs

- summary JSON: [`summary_metrics_v17.json`]({(REPORTS / 'summary_metrics_v17.json').as_posix()})
- core plot: [`v17_core_component_ablation.png`]({(REPORTS / 'v17_core_component_ablation.png').as_posix()})
- H1 plot: [`v17_h1_component_ablation.png`]({(REPORTS / 'v17_h1_component_ablation.png').as_posix()})
"""
    (REPORTS / "final_report_v17_querygrad_components.md").write_text(report)


if __name__ == "__main__":
    main()
