#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


RUN_SPECS: list[dict[str, Any]] = [
    {"label": "A s1234", "arm": "A", "variant": "static_bootstrap", "seed": 1234, "run": "20260317-210440-v7-static-bootstrap-hard-s1234"},
    {"label": "A s2234", "arm": "A", "variant": "static_bootstrap", "seed": 2234, "run": "20260317-210702-v7-static-bootstrap-hard-s2234"},
    {"label": "A s3234", "arm": "A", "variant": "static_bootstrap", "seed": 3234, "run": "20260317-210924-v7-static-bootstrap-hard-s3234"},
    {"label": "B s1234", "arm": "B", "variant": "staged_static", "seed": 1234, "run": "20260317-211145-v7-staged-static-hard-s1234"},
    {"label": "B s2234", "arm": "B", "variant": "staged_static", "seed": 2234, "run": "20260317-211442-v7-staged-static-hard-s2234"},
    {"label": "B s3234", "arm": "B", "variant": "staged_static", "seed": 3234, "run": "20260317-211739-v7-staged-static-hard-s3234"},
    {"label": "C s1234", "arm": "C", "variant": "growth_clone", "seed": 1234, "run": "20260317-212036-v7-growth-clone-hard-s1234"},
    {"label": "C s2234", "arm": "C", "variant": "growth_clone", "seed": 2234, "run": "20260317-212333-v7-growth-clone-hard-s2234"},
    {"label": "C s3234", "arm": "C", "variant": "growth_clone", "seed": 3234, "run": "20260317-212630-v7-growth-clone-hard-s3234"},
    {"label": "B long s4234", "arm": "B_long", "variant": "staged_static_long", "seed": 4234, "run": "20260317-213303-v7-staged-static-hard-long-s4234"},
    {"label": "C long s4234", "arm": "C_long", "variant": "growth_clone_long", "seed": 4234, "run": "20260317-213629-v7-growth-clone-hard-long-s4234"},
    {"label": "D long s4234", "arm": "D", "variant": "growth_mutate_long", "seed": 4234, "run": "20260317-214058-v7-growth-mutate-hard-long-s4234"},
    {"label": "D long s5234", "arm": "D", "variant": "growth_mutate_long", "seed": 5234, "run": "20260317-214422-v7-growth-mutate-hard-long-s5234"},
]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return statistics.stdev(values)


def mean_and_std(values: list[float]) -> dict[str, float]:
    return {"mean": float(statistics.mean(values)), "std": float(sample_std(values))}


def metrics_at_step(history: list[dict[str, Any]], step: int) -> dict[str, float]:
    row = history[min(step - 1, len(history) - 1)]
    return {
        "task_visit_coverage": float(row["task_visit_coverage"]),
        "task_gradient_coverage": float(row["task_gradient_coverage"]),
        "task_visit_ge5_fraction": float(row["task_visit_ge5_fraction"]),
        "task_visit_entropy": float(row["task_visit_entropy"]),
        "task_visit_gini": float(row["task_visit_gini"]),
    }


def series_mean(records: list[list[float]]) -> list[float]:
    length = min(len(record) for record in records)
    return [float(statistics.mean(record[i] for record in records)) for i in range(length)]


def load_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS / spec["run"]
    rows = read_jsonl(run_dir / "metrics.jsonl")
    val_rows = [row for row in rows if "val/query_accuracy" in row]
    best = max(val_rows, key=lambda row: row["val/query_accuracy"])
    last = val_rows[-1]
    coverage = read_json(run_dir / "coverage_summary.json")
    history = coverage["history"]
    final_stage = coverage["stages"][-1]

    evals: dict[str, Any] = {}
    for name in ("k2", "k6", "k10"):
        path = run_dir / f"eval_{name}.json"
        if path.exists():
            evals[name] = read_json(path)["metrics"]

    record = {
        **spec,
        "run_dir": str(run_dir),
        "best_val": {
            "step": int(best["step"]),
            "query_accuracy": float(best["val/query_accuracy"]),
            "query_first_hop_home_rate": float(best["val/query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(best["val/writer_first_hop_home_rate"]),
            "query_delivery_rate": float(best["val/query_delivery_rate"]),
            "query_home_to_output_rate": float(best["val/query_home_to_output_rate"]),
        },
        "last_val": {
            "step": int(last["step"]),
            "query_accuracy": float(last["val/query_accuracy"]),
            "query_first_hop_home_rate": float(last["val/query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(last["val/writer_first_hop_home_rate"]),
            "query_delivery_rate": float(last["val/query_delivery_rate"]),
            "query_home_to_output_rate": float(last["val/query_home_to_output_rate"]),
        },
        "coverage_checkpoints": {
            str(step): metrics_at_step(history, step) for step in (10, 50, 100, 200)
        },
        "final_stage": {
            "task_visit_coverage_at": {k: float(v) for k, v in final_stage["task_visit_coverage_at"].items()},
            "task_grad_coverage_at": {k: float(v) for k, v in final_stage["task_grad_coverage_at"].items()},
            "task_visit_ge5_at": {k: float(v) for k, v in final_stage["task_visit_ge5_at"].items()},
            "task_visit_entropy_at": {k: float(v) for k, v in final_stage["task_visit_entropy_at"].items()},
            "task_visit_gini_at": {k: float(v) for k, v in final_stage["task_visit_gini_at"].items()},
            "task_time_to_visit": {k: (None if v is None else int(v)) for k, v in final_stage["task_time_to_visit"].items()},
            "task_time_to_grad": {k: (None if v is None else int(v)) for k, v in final_stage["task_time_to_grad"].items()},
            "task_visit_ema": [float(v) for v in final_stage["task_visit_ema"]],
            "task_grad_ema": [float(v) for v in final_stage["task_grad_ema"]],
            "split_stats": final_stage["split_stats"],
        },
        "history_task_visit": [float(row["task_visit_coverage"]) for row in history],
        "history_task_grad": [float(row["task_gradient_coverage"]) for row in history],
        "history_val_steps": [int(row["step"]) for row in val_rows],
        "history_val_accuracy": [float(row["val/query_accuracy"]) for row in val_rows],
        "evals": evals,
    }
    return record


def plot_accuracy_bars(summary: dict[str, Any]) -> None:
    labels = ["A\nstatic+boot", "B\nstaged-static", "C\ngrowth-clone"]
    metrics = ["best_val", "last_val", "eval_k2", "eval_k6"]
    titles = ["Best Val", "Last Val", "Eval K2", "Eval K6"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics, titles):
        means = [summary["arm_summary"][arm][metric]["mean"] for arm in ("A", "B", "C")]
        stds = [summary["arm_summary"][arm][metric]["std"] for arm in ("A", "B", "C")]
        ax.bar(labels, means, yerr=stds, capsize=4, color=["#8da0cb", "#66c2a5", "#fc8d62"])
        ax.set_ylim(0.0, max(means) + max(stds) + 0.05)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V7 Initial A/B/C Mean±Std Across Seeds")
    fig.savefig(REPORTS / "v7_mean_std_accuracy_bars.png", dpi=180)
    plt.close(fig)


def plot_coverage_curves(records: list[dict[str, Any]], key: str, title: str, filename: str) -> None:
    colors = {"A": "#8da0cb", "B": "#66c2a5", "C": "#fc8d62"}
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm in ("A", "B", "C"):
        arm_records = [record for record in records if record["arm"] == arm]
        series = [record[key] for record in arm_records]
        mean = series_mean(series)
        ax.plot(range(1, len(mean) + 1), mean, label=arm, color=colors[arm])
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Coverage Fraction")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / filename, dpi=180)
    plt.close(fig)


def plot_stagewise_training(records: list[dict[str, Any]]) -> None:
    colors = {"A": "#8da0cb", "B": "#66c2a5", "C": "#fc8d62"}
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm in ("A", "B", "C"):
        arm_records = [record for record in records if record["arm"] == arm]
        x = arm_records[0]["history_val_steps"]
        ys = [record["history_val_accuracy"] for record in arm_records]
        y = series_mean(ys)
        ax.plot(x, y, label=f"{arm} mean", color=colors[arm])
    for label, path, color in (
        ("B long", "20260317-213303-v7-staged-static-hard-long-s4234", "#1b9e77"),
        ("C long", "20260317-213629-v7-growth-clone-hard-long-s4234", "#d95f02"),
    ):
        record = next(record for record in records if record["run"].endswith(path))
        ax.plot(record["history_val_steps"], record["history_val_accuracy"], linestyle="--", color=color, label=label)
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Validation Query Accuracy")
    ax.set_title("V7 Stage-Wise Validation Accuracy")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / "v7_stagewise_training_curves.png", dpi=180)
    plt.close(fig)


def plot_final_stage_coverage(records: list[dict[str, Any]]) -> None:
    targets = [
        ("B long", "B_long", "#1b9e77"),
        ("C long", "C_long", "#d95f02"),
        ("D long mean", "D", "#7570b3"),
    ]
    xs = [10, 50, 100, 200]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, arm, color in targets:
        arm_records = [record for record in records if record["arm"] == arm]
        visit = [
            statistics.mean(float(record["final_stage"]["task_visit_coverage_at"][str(x)]) for record in arm_records)
            for x in xs
        ]
        ge5 = [
            statistics.mean(float(record["final_stage"]["task_visit_ge5_at"][str(x)]) for record in arm_records)
            for x in xs
        ]
        grad = [
            statistics.mean(float(record["final_stage"]["task_grad_coverage_at"][str(x)]) for record in arm_records)
            for x in xs
        ]
        axes[0].plot(xs, visit, marker="o", label=label, color=color)
        axes[0].plot(xs, ge5, marker="x", linestyle="--", color=color, alpha=0.8)
        axes[1].plot(xs, grad, marker="o", label=label, color=color)
    axes[0].set_title("Final-Stage Task Visit Coverage")
    axes[0].set_xlabel("Final-Stage Local Step")
    axes[0].set_ylabel("Fraction")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].set_title("Final-Stage Task Gradient Coverage")
    axes[1].set_xlabel("Final-Stage Local Step")
    axes[1].set_ylabel("Fraction")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(REPORTS / "v7_final_stage_coverage.png", dpi=180)
    plt.close(fig)


def plot_final_stage_histograms(records: list[dict[str, Any]]) -> None:
    targets = [
        ("B long", "B_long", "#1b9e77"),
        ("C long", "C_long", "#d95f02"),
        ("D long", "D", "#7570b3"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, arm, color in targets:
        arm_records = [record for record in records if record["arm"] == arm]
        visit_values = [value for record in arm_records for value in record["final_stage"]["task_visit_ema"]]
        grad_values = [value for record in arm_records for value in record["final_stage"]["task_grad_ema"]]
        axes[0].hist(visit_values, bins=16, alpha=0.45, color=color, label=label)
        axes[1].hist([max(value, 1e-8) for value in grad_values], bins=16, alpha=0.45, color=color, label=label)
    axes[0].set_title("Final-Stage Task Visit EMA")
    axes[0].set_xlabel("EMA Visit Weight")
    axes[0].set_ylabel("Node Count")
    axes[0].legend()
    axes[1].set_title("Final-Stage Task Gradient EMA")
    axes[1].set_xlabel("EMA Gradient Norm")
    axes[1].set_ylabel("Node Count")
    axes[1].set_xscale("log")
    axes[1].legend()
    fig.savefig(REPORTS / "v7_final_stage_utility_histograms.png", dpi=180)
    plt.close(fig)


def plot_inheritance_comparison(summary: dict[str, Any], records: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    labels = ["B staged-static", "C growth-clone"]
    best_means = [summary["arm_summary"]["B"]["best_val"]["mean"], summary["arm_summary"]["C"]["best_val"]["mean"]]
    best_stds = [summary["arm_summary"]["B"]["best_val"]["std"], summary["arm_summary"]["C"]["best_val"]["std"]]
    k6_means = [summary["arm_summary"]["B"]["eval_k6"]["mean"], summary["arm_summary"]["C"]["eval_k6"]["mean"]]
    k6_stds = [summary["arm_summary"]["B"]["eval_k6"]["std"], summary["arm_summary"]["C"]["eval_k6"]["std"]]
    axes[0].bar(labels, best_means, yerr=best_stds, capsize=4, color=["#66c2a5", "#fc8d62"])
    axes[0].set_title("Initial Seeds: Best Val Mean±Std")
    axes[0].set_ylim(0.0, max(best_means) + max(best_stds) + 0.05)
    axes[0].grid(axis="y", alpha=0.2)
    axes[1].bar(labels, k6_means, yerr=k6_stds, capsize=4, color=["#66c2a5", "#fc8d62"])
    axes[1].set_title("Initial Seeds: Eval K6 Mean±Std")
    axes[1].set_ylim(0.0, max(k6_means) + max(k6_stds) + 0.05)
    axes[1].grid(axis="y", alpha=0.2)
    fig.savefig(REPORTS / "v7_inheritance_comparison.png", dpi=180)
    plt.close(fig)


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    arm_summary: dict[str, Any] = {}
    for arm in ("A", "B", "C"):
        arm_records = [record for record in records if record["arm"] == arm]
        arm_summary[arm] = {
            "best_val": mean_and_std([record["best_val"]["query_accuracy"] for record in arm_records]),
            "last_val": mean_and_std([record["last_val"]["query_accuracy"] for record in arm_records]),
            "eval_k2": mean_and_std([record["evals"]["k2"]["query_accuracy"] for record in arm_records]),
            "eval_k6": mean_and_std([record["evals"]["k6"]["query_accuracy"] for record in arm_records]),
            "query_first_hop_best": mean_and_std([record["best_val"]["query_first_hop_home_rate"] for record in arm_records]),
            "writer_first_hop_best": mean_and_std([record["best_val"]["writer_first_hop_home_rate"] for record in arm_records]),
        }
    followups = [record for record in records if record["arm"] not in ("A", "B", "C")]
    return {
        "visible_gpu_count": 2,
        "benchmark": {
            "final_leaf_homes": 32,
            "stage_schedule": [4, 8, 16, 32],
            "initial_stage_steps": [250, 350, 500, 1900],
            "long_stage_steps": [250, 350, 500, 2900],
            "writers_per_episode_train": 2,
            "query_ttl_range": [2, 3],
            "start_node_pool_size": 2,
            "max_rollout_steps": 12,
        },
        "runs": records,
        "arm_summary": arm_summary,
        "followups": followups,
    }


def main() -> None:
    records = [load_run(spec) for spec in RUN_SPECS]
    summary = build_summary(records)

    (REPORTS / "summary_metrics_v7.json").write_text(json.dumps(summary, indent=2))

    plot_accuracy_bars(summary)
    plot_coverage_curves([record for record in records if record["arm"] in ("A", "B", "C")], "history_task_visit", "Task-Only Visit Coverage (Initial A/B/C Seeds)", "v7_task_visit_coverage_curves.png")
    plot_coverage_curves([record for record in records if record["arm"] in ("A", "B", "C")], "history_task_grad", "Task-Only Gradient Coverage (Initial A/B/C Seeds)", "v7_task_gradient_coverage_curves.png")
    plot_stagewise_training(records)
    plot_final_stage_coverage(records)
    plot_final_stage_histograms(records)
    plot_inheritance_comparison(summary, records)


if __name__ == "__main__":
    main()
