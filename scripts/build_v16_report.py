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


RUN_GROUPS: dict[str, list[str]] = {
    "querygrad_h1": [
        "20260318-075401-v10-transfer-h1-utility-querygrad-longplus-s1234",
        "20260318-080752-v10-transfer-h1-utility-querygrad-longplus-s2234",
        "20260318-111409-v10-transfer-h1-utility-querygrad-longplus-s3234",
    ],
    "staged_static_h1": [
        "20260318-131453-v16-transfer-h1-staged-static-longplus-s1234",
        "20260318-132150-v16-transfer-h1-staged-static-longplus-s2234",
    ],
    "clone_h1": [
        "20260318-132900-v16-transfer-h1-clone-selective-longplus-s1234",
        "20260318-133557-v16-transfer-h1-clone-selective-longplus-s2234",
    ],
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


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def load_eval(run_dir: Path, writers: int) -> dict[str, float]:
    payload = read_json(run_dir / f"eval_best_k{writers}.json")
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def summarize_run(run_dir: Path, arm: str) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    val_rows = [row for row in metrics if "val/query_accuracy" in row]
    best = max(val_rows, key=lambda row: float(row["val/query_accuracy"]))
    last = val_rows[-1]
    coverage = read_json(run_dir / "coverage_summary.json")
    step10 = next(row for row in coverage["history"] if int(row["step"]) == 10)
    env = read_json(run_dir / "environment.json")
    seed = int(run_dir.name.rsplit("-s", 1)[1])
    return {
        "arm": arm,
        "seed": seed,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "visible_gpus": int(env.get("cuda_device_count", 0)),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "best_query_first_hop": float(best["val/query_first_hop_home_rate"]),
        "best_writer_first_hop": float(best["val/writer_first_hop_home_rate"]),
        "best_delivery": float(best["val/query_delivery_rate"]),
        "best_home_to_output": float(best["val/query_home_to_output_rate"]),
        "step10_task_visit": float(step10["task_visit_coverage"]),
        "step10_task_grad": float(step10["task_gradient_coverage"]),
        "step10_task_ge5": float(step10["task_visit_ge5_fraction"]),
        "k4": load_eval(run_dir, 4),
        "k8": load_eval(run_dir, 8),
        "k12": load_eval(run_dir, 12),
        "k14": load_eval(run_dir, 14),
    }


def summarize_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(records),
        "best_val": mean_std([record["best_val"] for record in records]),
        "last_val": mean_std([record["last_val"] for record in records]),
        "best_query_first_hop": mean_std([record["best_query_first_hop"] for record in records]),
        "best_writer_first_hop": mean_std([record["best_writer_first_hop"] for record in records]),
        "best_delivery": mean_std([record["best_delivery"] for record in records]),
        "best_home_to_output": mean_std([record["best_home_to_output"] for record in records]),
        "k4": mean_std([record["k4"]["query_accuracy"] for record in records]),
        "k8": mean_std([record["k8"]["query_accuracy"] for record in records]),
        "k12": mean_std([record["k12"]["query_accuracy"] for record in records]),
        "k14": mean_std([record["k14"]["query_accuracy"] for record in records]),
        "step10_task_visit": mean_std([record["step10_task_visit"] for record in records]),
        "step10_task_grad": mean_std([record["step10_task_grad"] for record in records]),
    }


def plot_summary(summary: dict[str, Any]) -> None:
    order = ["querygrad_h1", "staged_static_h1", "clone_h1"]
    labels = ["Querygrad", "Staged Static", "Clone"]
    metrics = [("k4", "K4"), ("k8", "K8"), ("k12", "K12"), ("k14", "K14")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes, metrics):
        means = [summary[arm][metric]["mean"] for arm in order]
        stds = [summary[arm][metric]["std"] for arm in order]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(REPORTS / "v16_h1_transfer_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    records: list[dict[str, Any]] = []
    for arm, runs in RUN_GROUPS.items():
        for run_name in runs:
            records.append(summarize_run(RUNS / run_name, arm))

    summary = {arm: summarize_arm([record for record in records if record["arm"] == arm]) for arm in RUN_GROUPS}
    plot_summary(summary)

    payload = {"summary": summary, "runs": records}
    (REPORTS / "summary_metrics_v16.json").write_text(json.dumps(payload, indent=2))

    per_seed_rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["arm"], item["seed"])):
        per_seed_rows.append(
            [
                record["arm"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['k4']['query_accuracy']:.4f}",
                f"{record['k8']['query_accuracy']:.4f}",
                f"{record['k12']['query_accuracy']:.4f}",
                f"{record['k14']['query_accuracy']:.4f}",
            ]
        )

    mean_rows: list[list[str]] = []
    for arm in ("querygrad_h1", "staged_static_h1", "clone_h1"):
        stats = summary[arm]
        mean_rows.append(
            [
                arm,
                str(stats["count"]),
                fmt_stat(stats["best_val"]),
                fmt_stat(stats["last_val"]),
                fmt_stat(stats["k4"]),
                fmt_stat(stats["k8"]),
                fmt_stat(stats["k12"]),
                fmt_stat(stats["k14"]),
            ]
        )

    visible_gpus = max(record["visible_gpus"] for record in records)
    report = f"""# APSGNN v16 H1 Transfer Validation

## What Changed

This follow-up does not change the model family. It validates the current H1 transfer regime by adding the two missing longplus controls under the same settings:

- `staged_static_h1`: staged curriculum with activation but no inheritance
- `clone_h1`: deterministic clone growth with inheritance
- `querygrad_h1`: existing utility-only `querygrad` selective growth from v10

The goal is to check whether the v10 `querygrad` winner still looks like the safest default once the transfer comparison includes both a curriculum-only control and an inheritance-only control.

Visible GPU count used: `{visible_gpus}`

## Exact Configs

- [`v16_transfer_h1_staged_static_longplus.yaml`]({(ROOT / 'configs/v16_transfer_h1_staged_static_longplus.yaml').as_posix()})
- [`v16_transfer_h1_clone_selective_longplus.yaml`]({(ROOT / 'configs/v16_transfer_h1_clone_selective_longplus.yaml').as_posix()})
- Existing reference arm: `v10_transfer_h1_utility_querygrad_longplus.yaml`

Shared regime:

- train writers per episode: `4`
- eval writers per episode: `4, 8, 12, 14`
- total train steps: `8000`
- selective stage schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`
- stage steps: `[250, 250, 300, 300, 400, 600, 5900]`

## Per-Seed Results

{markdown_table(['Arm', 'Seed', 'Best Val', 'Last Val', 'K4', 'K8', 'K12', 'K14'], per_seed_rows)}

## Mean / Std Summary

{markdown_table(['Arm', 'Count', 'Best Val', 'Last Val', 'K4', 'K8', 'K12', 'K14'], mean_rows)}

## Interpretation

`querygrad_h1` remains the safest transfer default. It stays ahead of `staged_static_h1` on mean `K4`, `K8`, and `K12`, and it also stays ahead of `clone_h1` on mean `K4` and `K14`. `clone_h1` does recover some of the gap versus `querygrad_h1` in the middle of the grid and clearly improves over `staged_static_h1`, which is evidence that inheritance still matters under H1.

The cleaner conclusion is therefore:

- curriculum alone helps, but not enough
- inheritance helps beyond staged activation
- utility-selected growth is still the strongest transfer default overall

This validates the earlier v9-v15 direction rather than overturning it. The main open question is still not whether clone beats staged static under transfer; it does. The open question is whether a mutation policy can beat utility-only `querygrad` without hurting robustness, and the accumulated v11-v15 results still say no.

## Outputs

- summary JSON: [`summary_metrics_v16.json`]({(REPORTS / 'summary_metrics_v16.json').as_posix()})
- plot: [`v16_h1_transfer_comparison.png`]({(REPORTS / 'v16_h1_transfer_comparison.png').as_posix()})
"""

    (REPORTS / "final_report_v16_h1_transfer_validation.md").write_text(report)


if __name__ == "__main__":
    main()
