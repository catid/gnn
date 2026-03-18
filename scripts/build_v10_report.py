#!/usr/bin/env python3
from __future__ import annotations

import json
import re
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


def latest_runs(prefix: str) -> list[tuple[Path, int]]:
    latest_by_seed: dict[int, Path] = {}
    pattern = re.compile(r"-s(\d+)$")
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        match = pattern.search(candidate.name)
        if not match:
            continue
        seed = int(match.group(1))
        latest_by_seed[seed] = candidate
    return [(latest_by_seed[seed], seed) for seed in sorted(latest_by_seed)]


def extract_evals(run_dir: Path) -> dict[str, float]:
    out = {}
    for tag in ("best_k2", "best_k6", "best_k10", "best_k4", "best_k8", "best_k12", "best_k14"):
        path = run_dir / f"eval_{tag}.json"
        if path.exists():
            payload = read_json(path)
            metrics = payload.get("metrics", payload)
            out[tag] = float(metrics["query_accuracy"])
    return out


def summarize_run(run_dir: Path, arm: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    coverage = read_json(run_dir / "coverage_summary.json")
    split_events = [stage.get("split_stats", {}) for stage in coverage["stages"][:-1]]
    corr = [
        float(event.get("utility_usefulness_correlation", 0.0))
        for event in split_events
        if event.get("parent_components")
    ]
    selected = [
        float(event.get("selected_parent_child_usefulness_mean", 0.0))
        for event in split_events
        if event.get("parent_components")
    ]
    unselected = [
        float(event.get("unselected_parent_child_usefulness_mean", 0.0))
        for event in split_events
        if event.get("parent_components")
    ]
    mutation_win = [
        float(event.get("mutated_child_usefulness_win_rate", 0.0))
        for event in split_events
        if event.get("mutated_children")
    ]
    return {
        "arm": arm,
        "seed": seed,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "best_query_first_hop": float(best["val/query_first_hop_home_rate"]),
        "best_writer_first_hop": float(best["val/writer_first_hop_home_rate"]),
        "best_delivery": float(best["val/query_delivery_rate"]),
        "best_home_to_output": float(best["val/query_home_to_output_rate"]),
        "task_visit_10": float(coverage["history"][9]["task_visit_coverage"]),
        "task_grad_10": float(coverage["history"][9]["task_gradient_coverage"]),
        "evals": extract_evals(run_dir),
        "utility_usefulness_correlation": mean(corr),
        "selected_child_usefulness": mean(selected),
        "unselected_child_usefulness": mean(unselected),
        "mutation_win_rate": mean(mutation_win),
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({record["arm"] for record in records}):
        arm_records = [record for record in records if record["arm"] == arm]
        out[arm] = {
            "count": len(arm_records),
            "best_val": mean_std([record["best_val"] for record in arm_records]),
            "last_val": mean_std([record["last_val"] for record in arm_records]),
            "best_k2": mean_std([record["evals"]["best_k2"] for record in arm_records if "best_k2" in record["evals"]]),
            "best_k6": mean_std([record["evals"]["best_k6"] for record in arm_records if "best_k6" in record["evals"]]),
            "best_k10": mean_std([record["evals"]["best_k10"] for record in arm_records if "best_k10" in record["evals"]]),
            "best_k4": mean_std([record["evals"]["best_k4"] for record in arm_records if "best_k4" in record["evals"]]),
            "best_k8": mean_std([record["evals"]["best_k8"] for record in arm_records if "best_k8" in record["evals"]]),
            "best_k12": mean_std([record["evals"]["best_k12"] for record in arm_records if "best_k12" in record["evals"]]),
            "best_k14": mean_std([record["evals"]["best_k14"] for record in arm_records if "best_k14" in record["evals"]]),
            "utility_usefulness_correlation": mean_std([record["utility_usefulness_correlation"] for record in arm_records]),
            "selected_child_usefulness": mean_std([record["selected_child_usefulness"] for record in arm_records]),
            "unselected_child_usefulness": mean_std([record["unselected_child_usefulness"] for record in arm_records]),
            "mutation_win_rate": mean_std([record["mutation_win_rate"] for record in arm_records]),
        }
    return out


def plot_comparison(summary: dict[str, Any]) -> None:
    arms = ["v9_U", "v9_US", "querygrad", "querymix", "querygrad_mutate", "querymix_mutate"]
    labels = ["v9 U", "v9 US", "v10 QG", "v10 QMix", "v10 QG+M", "v10 QMix+M"]
    metrics = ["last_val", "best_k6", "best_k10"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, metric in zip(axes, metrics):
        means = [summary.get(arm, {}).get(metric, {}).get("mean", 0.0) for arm in arms]
        stds = [summary.get(arm, {}).get(metric, {}).get("std", 0.0) for arm in arms]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(metric.replace("_", " ").title())
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(REPORTS / "v10_query_utility_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    v9 = read_json(REPORTS / "summary_metrics_v9.json")
    records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("querygrad", "v10-utility-querygrad-longplus"),
        ("querymix", "v10-utility-querymix-longplus"),
        ("querygrad_mutate", "v10-utility-querygrad-mutate-longplus"),
        ("querymix_mutate", "v10-utility-querymix-mutate-longplus"),
        ("transfer_querygrad", "v10-transfer-h1-utility-querygrad-longplus"),
        ("transfer_querygrad_mutate", "v10-transfer-h1-utility-querygrad-mutate-longplus"),
        ("transfer_querymix", "v10-transfer-h1-utility-querymix-longplus"),
        ("transfer_querymix_mutate", "v10-transfer-h1-utility-querymix-mutate-longplus"),
    ):
        for run_dir, seed in latest_runs(prefix):
            records.append(summarize_run(run_dir, arm, seed))

    summary = summarize(records)
    summary["v9_U"] = {
        "count": v9["arm_summary"]["U"]["count"],
        "best_val": v9["arm_summary"]["U"]["best_val"],
        "last_val": v9["arm_summary"]["U"]["last_val"],
        "best_k2": v9["arm_summary"]["U"]["eval_best_k2"],
        "best_k6": v9["arm_summary"]["U"]["eval_best_k6"],
        "best_k10": v9["arm_summary"]["U"]["eval_best_k10"],
    }
    summary["v9_US"] = {
        "count": v9["arm_summary"]["US"]["count"],
        "best_val": v9["arm_summary"]["US"]["best_val"],
        "last_val": v9["arm_summary"]["US"]["last_val"],
        "best_k2": v9["arm_summary"]["US"]["eval_best_k2"],
        "best_k6": v9["arm_summary"]["US"]["eval_best_k6"],
        "best_k10": v9["arm_summary"]["US"]["eval_best_k10"],
    }

    plot_comparison(summary)
    payload = {"summary": summary, "runs": records}
    (REPORTS / "summary_metrics_v10.json").write_text(json.dumps(payload, indent=2))

    core_rows = []
    for arm in ("v9_U", "v9_US", "querygrad", "querymix", "querygrad_mutate", "querymix_mutate"):
        if arm not in summary:
            continue
        stat = summary[arm]
        core_rows.append(
            [
                arm,
                str(stat["count"]),
                fmt_stat(stat["best_val"]),
                fmt_stat(stat["last_val"]),
                fmt_stat(stat["best_k6"]),
                fmt_stat(stat["best_k10"]),
            ]
        )
    transfer_rows = []
    for arm in ("transfer_querygrad", "transfer_querygrad_mutate", "transfer_querymix", "transfer_querymix_mutate"):
        if arm not in summary:
            continue
        stat = summary[arm]
        transfer_rows.append(
            [arm, str(stat["count"]), fmt_stat(stat["best_k4"]), fmt_stat(stat["best_k8"]), fmt_stat(stat["best_k12"]), fmt_stat(stat["best_k14"])]
        )
    text = f"""# APSGNN v10: Query-Aware Utility Refinement

## Purpose

V10 follows directly from v9. The goal is to keep the hard selective-growth benchmark fixed and test a narrower utility score that uses tail query-specific traffic/gradient signal instead of the weak success-conditioned traffic term. The core comparison is against the pushed v9 utility baselines `U` and `US`.

## Core Summary

{markdown_table(["Arm", "Seeds", "Best Val", "Last Val", "Best K6", "Best K10"], core_rows)}

## Transfer Summary

{markdown_table(["Arm", "Seeds", "Best K4", "Best K8", "Best K12", "Best K14"], transfer_rows) if transfer_rows else "Transfer runs not completed."}

## Key Findings

- `querygrad` materially improved over the pushed v9 `U` and `US` baselines on the core regime.
- `querymix` was retained only as a redundancy check. On the two-seed score-selection round it produced the same split parents and the same `K2/K6/K10` checkpoint accuracies as `querygrad`.
- `querygrad_mutate` did not improve mean best validation over `querygrad`, but it did improve mean last-checkpoint stability and mean `K10` on the core regime.
- The H1 transfer round favored `querygrad` over `querygrad_mutate`, so mutation is not yet a reliable default upgrade.

## Notes

- `querygrad`: `z(visits) + z(grad) + z(query_grad)`
- `querymix`: `z(visits) + z(grad) + 0.5*z(query_visit) + z(query_grad)`
- `querygrad_mutate`: same score as `querygrad` with the existing modest local mutation policy
- The v9 baselines are read from [summary_metrics_v9.json](/home/catid/gnn/reports/summary_metrics_v9.json)

Artifacts:
- [summary_metrics_v10.json](/home/catid/gnn/reports/summary_metrics_v10.json)
- [v10_query_utility_comparison.png](/home/catid/gnn/reports/v10_query_utility_comparison.png)
"""
    (REPORTS / "final_report_v10_query_utility.md").write_text(text)


if __name__ == "__main__":
    main()
