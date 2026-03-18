#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


ARM_PATTERNS: dict[str, list[str]] = {
    "A": ["v8-static-bootstrap-hard-s"],
    "B": ["v8-staged-static-selective-hard-s"],
    "C": ["v8-clone-selective-hard-s"],
    "R": ["v8-random-selective-hard-s"],
    "U": ["v8-utility-selective-hard-s"],
    "C_followup": ["v8-clone-selective-hard-long-s"],
    "U_followup": ["v8-utility-selective-hard-long-s"],
    "R_followup": ["v8-random-selective-hard-long-s"],
    "UM_followup": ["v8-utility-mutate-hard-long-s"],
}


SEED_RE = re.compile(r"-s(?P<seed>\d+)$")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def sample_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return float(statistics.stdev(values))


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": sample_std(values)}


def latest_matching_run(prefix: str) -> Path | None:
    matches = sorted(RUNS.glob(f"*-{prefix}*"))
    non_smoke = [path for path in matches if "-smoke-" not in path.name]
    if non_smoke:
        return non_smoke[-1]
    return matches[-1] if matches else None


def discover_runs() -> list[dict[str, Any]]:
    latest_specs: dict[tuple[str, int], dict[str, Any]] = {}
    for arm, prefixes in ARM_PATTERNS.items():
        for prefix in prefixes:
            for path in sorted(RUNS.glob(f"*-{prefix}*")):
                if "-smoke-" in path.name:
                    continue
                run_name = path.name.split("-", 1)[1]
                match = SEED_RE.search(run_name)
                if not match:
                    continue
                spec = {
                    "arm": arm,
                    "run": path.name,
                    "run_name": run_name,
                    "seed": int(match.group("seed")),
                }
                key = (arm, spec["seed"])
                previous = latest_specs.get(key)
                if previous is None or spec["run"] > previous["run"]:
                    latest_specs[key] = spec
    specs = list(latest_specs.values())
    specs.sort(key=lambda spec: (spec["arm"], spec["seed"], spec["run"]))
    return specs


def extract_eval_metrics(run_dir: Path) -> dict[str, dict[str, float]]:
    evals: dict[str, dict[str, float]] = {}
    for path in sorted(run_dir.glob("eval_*.json")):
        payload = read_json(path)
        tag = path.stem.removeprefix("eval_")
        metrics = payload.get("metrics", payload)
        evals[tag] = {
            "query_accuracy": float(metrics["query_accuracy"]),
            "query_first_hop_home_rate": float(metrics["query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(metrics["writer_first_hop_home_rate"]),
            "query_delivery_rate": float(metrics["query_delivery_rate"]),
            "query_home_to_output_rate": float(metrics["query_home_to_output_rate"]),
        }
    return evals


def metrics_at_step(history: list[dict[str, Any]], step: int) -> dict[str, float]:
    row = history[min(step - 1, len(history) - 1)]
    return {
        "task_visit_coverage": float(row["task_visit_coverage"]),
        "task_gradient_coverage": float(row["task_gradient_coverage"]),
        "task_nodes_ge5_visit_fraction": float(
            row.get("task_nodes_ge5_visit_fraction", row["task_visit_ge5_fraction"]),
        ),
        "task_visit_entropy": float(row["task_visit_entropy"]),
        "task_visit_gini": float(row["task_visit_gini"]),
    }


def series_mean(records: list[list[float]]) -> list[float]:
    if not records:
        return []
    length = min(len(record) for record in records)
    return [float(statistics.mean(record[i] for record in records)) for i in range(length)]


def load_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS / spec["run"]
    config = read_json(run_dir / "config.json")
    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    if not metrics_rows:
        raise ValueError(f"Run {run_dir} has no metrics.")
    if int(metrics_rows[-1]["step"]) < int(config["train"]["train_steps"]):
        raise ValueError(f"Run {run_dir} is incomplete.")
    val_rows = [row for row in metrics_rows if "val/query_accuracy" in row]
    best_row = max(val_rows, key=lambda row: row["val/query_accuracy"])
    last_row = val_rows[-1]
    coverage = read_json(run_dir / "coverage_summary.json")
    history = coverage["history"]
    stages = coverage["stages"]
    final_stage = stages[-1]
    evals = extract_eval_metrics(run_dir)

    return {
        **spec,
        "run_dir": str(run_dir),
        "config": config,
        "best_val": {
            "step": int(best_row["step"]),
            "query_accuracy": float(best_row["val/query_accuracy"]),
            "query_first_hop_home_rate": float(best_row["val/query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(best_row["val/writer_first_hop_home_rate"]),
            "query_delivery_rate": float(best_row["val/query_delivery_rate"]),
            "query_home_to_output_rate": float(best_row["val/query_home_to_output_rate"]),
        },
        "last_val": {
            "step": int(last_row["step"]),
            "query_accuracy": float(last_row["val/query_accuracy"]),
            "query_first_hop_home_rate": float(last_row["val/query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(last_row["val/writer_first_hop_home_rate"]),
            "query_delivery_rate": float(last_row["val/query_delivery_rate"]),
            "query_home_to_output_rate": float(last_row["val/query_home_to_output_rate"]),
        },
        "coverage_checkpoints": {str(step): metrics_at_step(history, step) for step in (10, 50, 100, 200)},
        "history_steps": [int(row["step"]) for row in history],
        "history_task_visit": [float(row["task_visit_coverage"]) for row in history],
        "history_task_grad": [float(row["task_gradient_coverage"]) for row in history],
        "history_val_steps": [int(row["step"]) for row in val_rows],
        "history_val_accuracy": [float(row["val/query_accuracy"]) for row in val_rows],
        "final_stage": {
            "active_node_ids": list(final_stage["active_node_ids"]),
            "task_visit_coverage_at": {k: float(v) for k, v in final_stage["task_visit_coverage_at"].items()},
            "task_grad_coverage_at": {k: float(v) for k, v in final_stage["task_grad_coverage_at"].items()},
            "task_visit_ge5_at": {k: float(v) for k, v in final_stage["task_visit_ge5_at"].items()},
            "task_visit_entropy_at": {k: float(v) for k, v in final_stage["task_visit_entropy_at"].items()},
            "task_visit_gini_at": {k: float(v) for k, v in final_stage["task_visit_gini_at"].items()},
            "task_time_to_visit": {k: (None if v is None else int(v)) for k, v in final_stage["task_time_to_visit"].items()},
            "task_time_to_grad": {k: (None if v is None else int(v)) for k, v in final_stage["task_time_to_grad"].items()},
            "task_visit_ema": [float(v) for v in final_stage["task_visit_ema"]],
            "task_grad_ema": [float(v) for v in final_stage["task_grad_ema"]],
            "task_success_ema": [float(v) for v in final_stage.get("task_success_ema", [])],
            "split_stats": final_stage["split_stats"],
        },
        "evals": evals,
    }


def summarize_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    result = {
        "count": len(records),
        "best_val": mean_std([record["best_val"]["query_accuracy"] for record in records]),
        "last_val": mean_std([record["last_val"]["query_accuracy"] for record in records]),
        "best_query_first_hop": mean_std([record["best_val"]["query_first_hop_home_rate"] for record in records]),
        "best_writer_first_hop": mean_std([record["best_val"]["writer_first_hop_home_rate"] for record in records]),
        "best_delivery": mean_std([record["best_val"]["query_delivery_rate"] for record in records]),
        "best_home_to_output": mean_std([record["best_val"]["query_home_to_output_rate"] for record in records]),
        "task_visit_10": mean_std([record["coverage_checkpoints"]["10"]["task_visit_coverage"] for record in records]),
        "task_grad_10": mean_std([record["coverage_checkpoints"]["10"]["task_gradient_coverage"] for record in records]),
        "task_ge5_50": mean_std([record["coverage_checkpoints"]["50"]["task_nodes_ge5_visit_fraction"] for record in records]),
        "task_gini_50": mean_std([record["coverage_checkpoints"]["50"]["task_visit_gini"] for record in records]),
    }
    for tag in ("best_k2", "best_k6", "best_k10", "last_k2", "last_k6", "last_k10", "k2", "k6", "k10"):
        values = [record["evals"][tag]["query_accuracy"] for record in records if tag in record["evals"]]
        if values:
            result[f"eval_{tag}"] = mean_std(values)
    return result


def plot_accuracy_bars(summary: dict[str, Any]) -> None:
    arms = ["A", "B", "C", "R", "U"]
    labels = ["A\nstatic", "B\nstaged", "C\nclone", "R\nrandom", "U\nutility"]
    metrics = [
        ("best_val", "Best Val"),
        ("last_val", "Last Val"),
        ("eval_best_k6", "Best Eval K6"),
        ("eval_last_k10", "Last Eval K10"),
    ]
    colors = ["#8da0cb", "#66c2a5", "#fc8d62", "#e78ac3", "#a6d854"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        means = [summary["arm_summary"].get(arm, {}).get(metric_key, {}).get("mean", 0.0) for arm in arms]
        stds = [summary["arm_summary"].get(arm, {}).get(metric_key, {}).get("std", 0.0) for arm in arms]
        ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
        ax.set_ylim(0.0, max(means) + max(stds) + 0.05 if max(means) > 0 else 0.1)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("V8 Initial Mean±Std Across Arms")
    fig.savefig(REPORTS / "v8_mean_std_accuracy_bars.png", dpi=180)
    plt.close(fig)


def plot_coverage_curves(records: list[dict[str, Any]], key: str, title: str, filename: str) -> None:
    colors = {"A": "#8da0cb", "B": "#66c2a5", "C": "#fc8d62", "R": "#e78ac3", "U": "#a6d854"}
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm in ("A", "B", "C", "R", "U"):
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        mean_series = series_mean([record[key] for record in arm_records])
        ax.plot(range(1, len(mean_series) + 1), mean_series, label=arm, color=colors[arm])
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Coverage Fraction")
    ax.set_title(title)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / filename, dpi=180)
    plt.close(fig)


def plot_stagewise_training(records: list[dict[str, Any]]) -> None:
    colors = {"A": "#8da0cb", "B": "#66c2a5", "C": "#fc8d62", "R": "#e78ac3", "U": "#a6d854"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for arm in ("A", "B", "C", "R", "U"):
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        x = arm_records[0]["history_val_steps"]
        y = series_mean([record["history_val_accuracy"] for record in arm_records])
        ax.plot(x, y, label=arm, color=colors[arm])
    for arm, style in (("C_followup", "--"), ("U_followup", "--"), ("R_followup", "--"), ("UM_followup", "--")):
        for record in [r for r in records if r["arm"] == arm]:
            ax.plot(record["history_val_steps"], record["history_val_accuracy"], linestyle=style, alpha=0.8, label=f"{arm} s{record['seed']}")
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Validation Query Accuracy")
    ax.set_title("V8 Stage-Wise Validation Accuracy")
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=8)
    fig.savefig(REPORTS / "v8_stagewise_training_curves.png", dpi=180)
    plt.close(fig)


def plot_final_stage_coverage(records: list[dict[str, Any]]) -> None:
    xs = [10, 50, 100, 200]
    colors = {"B": "#66c2a5", "C": "#fc8d62", "R": "#e78ac3", "U": "#a6d854"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm in ("B", "C", "R", "U"):
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        visit = [mean([float(record["final_stage"]["task_visit_coverage_at"][str(x)]) for record in arm_records]) for x in xs]
        grad = [mean([float(record["final_stage"]["task_grad_coverage_at"][str(x)]) for record in arm_records]) for x in xs]
        axes[0].plot(xs, visit, marker="o", label=arm, color=colors[arm])
        axes[1].plot(xs, grad, marker="o", label=arm, color=colors[arm])
    axes[0].set_title("Final-Stage Task Visit Coverage")
    axes[1].set_title("Final-Stage Task Gradient Coverage")
    for ax in axes:
        ax.set_xlabel("Final-Stage Local Step")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(REPORTS / "v8_final_stage_coverage.png", dpi=180)
    plt.close(fig)


def plot_selected_vs_unselected(records: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for arm, color in (("U", "#a6d854"), ("R", "#e78ac3")):
        arm_records = [record for record in records if record["arm"] == arm]
        selected = []
        unselected = []
        usefulness = []
        for record in arm_records:
            for stage in read_json(Path(record["run_dir"]) / "coverage_summary.json")["stages"]:
                stats = stage["split_stats"]
                if not stats:
                    continue
                selected.append(float(stats.get("selected_parent_utility_mean", 0.0)))
                unselected.append(float(stats.get("unselected_parent_utility_mean", 0.0)))
                corr = stats.get("utility_usefulness_correlation")
                if corr is not None:
                    usefulness.append(float(corr))
        if selected:
            xs = range(len(selected))
            axes[0].plot(xs, selected, marker="o", color=color, label=f"{arm} selected")
            axes[0].plot(xs, unselected, marker="x", linestyle="--", color=color, alpha=0.8, label=f"{arm} unselected")
        if usefulness:
            axes[1].plot(range(len(usefulness)), usefulness, marker="o", color=color, label=arm)
    axes[0].set_title("Selected vs Unselected Parent Utility")
    axes[0].set_xlabel("Transition Index")
    axes[0].set_ylabel("Mean Utility")
    axes[0].grid(alpha=0.2)
    axes[0].legend()
    axes[1].set_title("Parent Utility vs Later Child Usefulness")
    axes[1].set_xlabel("Transition Index")
    axes[1].set_ylabel("Correlation")
    axes[1].grid(alpha=0.2)
    axes[1].legend()
    fig.savefig(REPORTS / "v8_selected_vs_unselected_utility.png", dpi=180)
    plt.close(fig)


def plot_child_usefulness(records: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm, color in (("C", "#fc8d62"), ("R", "#e78ac3"), ("U", "#a6d854"), ("UM_followup", "#ffd92f")):
        xs = []
        ys = []
        for record in [r for r in records if r["arm"] == arm]:
            coverage = read_json(Path(record["run_dir"]) / "coverage_summary.json")
            for stage in coverage["stages"]:
                stats = stage["split_stats"]
                if not stats:
                    continue
                for parent, usefulness in stats.get("parent_child_usefulness", {}).items():
                    score = stats.get("parent_components", {}).get(parent, {}).get("score")
                    if score is None:
                        continue
                    xs.append(float(score))
                    ys.append(float(usefulness))
        if xs:
            ax.scatter(xs, ys, alpha=0.6, color=color, label=arm)
    ax.set_title("Child Usefulness vs Parent Utility Score")
    ax.set_xlabel("Parent Utility Score")
    ax.set_ylabel("Mean Child Usefulness")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / "v8_child_usefulness_vs_parent_utility.png", dpi=180)
    plt.close(fig)


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    detailed_arm_groups = ["A", "B", "C", "R", "U", "C_followup", "U_followup", "R_followup", "UM_followup"]
    detailed_arm_summary = {
        arm: summarize_arm([record for record in records if record["arm"] == arm])
        for arm in detailed_arm_groups
        if any(record["arm"] == arm for record in records)
    }
    combined_arm_groups = {
        "A": ["A"],
        "B": ["B"],
        "C": ["C", "C_followup"],
        "R": ["R", "R_followup"],
        "U": ["U", "U_followup"],
        "UM": ["UM_followup"],
    }
    arm_summary = {
        arm: summarize_arm([record for record in records if record["arm"] in members])
        for arm, members in combined_arm_groups.items()
        if any(record["arm"] in members for record in records)
    }
    return {
        "visible_gpu_count": 2,
        "benchmark": {
            "final_leaf_homes": 32,
            "stage_schedule": [4, 6, 8, 12, 16, 24, 32],
            "initial_train_steps": 4000,
            "followup_train_steps": 5500,
            "final_stage_minimum_fraction": 0.5,
            "writers_per_episode_train": 2,
            "start_node_pool_size": 2,
            "query_ttl_range": [2, 3],
            "max_rollout_steps": 12,
            "utility_alpha": 0.75,
        },
        "runs": records,
        "arm_summary": arm_summary,
        "detailed_arm_summary": detailed_arm_summary,
    }


def main() -> None:
    specs = discover_runs()
    records = [load_run(spec) for spec in specs]
    summary = build_summary(records)
    (REPORTS / "summary_metrics_v8.json").write_text(json.dumps(summary, indent=2))

    initial_records = [record for record in records if record["arm"] in {"A", "B", "C", "R", "U"}]
    plot_accuracy_bars(summary)
    plot_coverage_curves(initial_records, "history_task_visit", "V8 Task-Only Visit Coverage", "v8_task_visit_coverage_curves.png")
    plot_coverage_curves(initial_records, "history_task_grad", "V8 Task-Only Gradient Coverage", "v8_task_gradient_coverage_curves.png")
    plot_stagewise_training(records)
    plot_final_stage_coverage(initial_records)
    plot_selected_vs_unselected(records)
    plot_child_usefulness(records)


if __name__ == "__main__":
    main()
