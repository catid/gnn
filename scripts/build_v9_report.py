#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
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
    "B": ["v9-staged-static-selective-long-s"],
    "C": ["v9-clone-selective-long-s"],
    "U": ["v9-utility-selective-long-s"],
    "UM": ["v9-utility-mutate-long-s"],
    "US": ["v9-utility-nosuccess-long-s"],
    "UG": ["v9-utility-nograd-long-s"],
    "U_followup": ["v9-utility-selective-longplus-s"],
    "UM_followup": ["v9-utility-mutate-longplus-s"],
    "US_followup": ["v9-utility-nosuccess-longplus-s"],
    "UG_followup": ["v9-utility-nograd-longplus-s"],
    "RM": ["v9-random-mutate-long-s"],
    "transfer_U": ["v9-transfer-h1-utility-longplus-s", "v9-transfer-h1-utility-long-s"],
    "transfer_UM": ["v9-transfer-h1-utility-mutate-longplus-s", "v9-transfer-h1-utility-mutate-long-s"],
}

ARM_LABELS = {
    "B": "B staged-static",
    "C": "C clone",
    "U": "U utility",
    "UM": "UM utility+mutate",
    "US": "US no-success",
    "UG": "UG no-grad",
    "RM": "RM random+mutate",
    "transfer_U": "transfer U",
    "transfer_UM": "transfer UM",
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


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) <= 1 or len(xs) != len(ys):
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    x_centered = [x - mx for x in xs]
    y_centered = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denom <= 1e-12:
        return 0.0
    return float(sum(x * y for x, y in zip(x_centered, y_centered)) / denom)


def bootstrap_mean_diff(
    a: list[float],
    b: list[float],
    *,
    trials: int = 4000,
    seed: int = 0,
) -> dict[str, float]:
    if not a or not b:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(seed)
    diffs: list[float] = []
    for _ in range(trials):
        sample_a = [rng.choice(a) for _ in range(len(a))]
        sample_b = [rng.choice(b) for _ in range(len(b))]
        diffs.append(mean(sample_a) - mean(sample_b))
    diffs.sort()
    low = diffs[int(0.025 * len(diffs))]
    high = diffs[int(0.975 * len(diffs))]
    return {"mean_diff": mean(a) - mean(b), "ci_low": low, "ci_high": high}


def latest_specs() -> list[dict[str, Any]]:
    latest: dict[tuple[str, int], dict[str, Any]] = {}
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
                previous = latest.get(key)
                if previous is None or spec["run"] > previous["run"]:
                    latest[key] = spec
    specs = list(latest.values())
    specs.sort(key=lambda spec: (spec["arm"], spec["seed"], spec["run"]))
    return specs


def extract_evals(run_dir: Path) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for path in sorted(run_dir.glob("eval_*.json")):
        payload = read_json(path)
        metrics = payload.get("metrics", payload)
        tag = path.stem.removeprefix("eval_")
        result[tag] = {
            "query_accuracy": float(metrics["query_accuracy"]),
            "query_first_hop_home_rate": float(metrics["query_first_hop_home_rate"]),
            "writer_first_hop_home_rate": float(metrics["writer_first_hop_home_rate"]),
            "query_delivery_rate": float(metrics["query_delivery_rate"]),
            "query_home_to_output_rate": float(metrics["query_home_to_output_rate"]),
        }
    return result


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


def last_n_mean(values: list[float], n: int = 5) -> float:
    if not values:
        return 0.0
    window = values[-min(n, len(values)) :]
    return mean(window)


def transition_stage_records(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    split_events: list[dict[str, Any]] = []
    for stage in stages[:-1]:
        split_stats = stage.get("split_stats", {})
        if not split_stats:
            continue
        split_events.append(
            {
                "stage_name": stage["stage_name"],
                "selection_policy": split_stats.get("selection_policy", ""),
                "selected_parent_utility_mean": float(split_stats.get("selected_parent_utility_mean", 0.0)),
                "unselected_parent_utility_mean": float(split_stats.get("unselected_parent_utility_mean", 0.0)),
                "selected_parent_child_usefulness_mean": float(
                    split_stats.get("selected_parent_child_usefulness_mean", 0.0),
                ),
                "unselected_parent_child_usefulness_mean": float(
                    split_stats.get("unselected_parent_child_usefulness_mean", 0.0),
                ),
                "utility_usefulness_correlation": float(split_stats.get("utility_usefulness_correlation", 0.0)),
                "mutated_child_usefulness_win_rate": float(
                    split_stats.get("mutated_child_usefulness_win_rate", 0.0),
                ),
                "mutated_child_visit_share_mean": float(
                    split_stats.get("mutated_child_visit_share_mean", 0.0),
                ),
                "mutated_child_grad_share_mean": float(
                    split_stats.get("mutated_child_grad_share_mean", 0.0),
                ),
                "mutated_child_usefulness_share_mean": float(
                    split_stats.get("mutated_child_usefulness_share_mean", 0.0),
                ),
                "sibling_divergence": float(split_stats.get("sibling_divergence", 0.0)),
                "selected_parent_scores": split_stats.get("selected_parent_scores", {}),
                "unselected_parent_scores": split_stats.get("unselected_parent_scores", {}),
                "parent_child_usefulness": split_stats.get("parent_child_usefulness", {}),
                "parent_components": split_stats.get("parent_components", {}),
            }
        )
    return split_events


def load_run(spec: dict[str, Any]) -> dict[str, Any]:
    run_dir = RUNS / spec["run"]
    metrics_rows = read_jsonl(run_dir / "metrics.jsonl")
    val_rows = [row for row in metrics_rows if "val/query_accuracy" in row]
    if not val_rows:
        raise ValueError(f"Run {run_dir} has no validation rows")
    best_row = max(val_rows, key=lambda row: float(row["val/query_accuracy"]))
    last_row = val_rows[-1]
    coverage = read_json(run_dir / "coverage_summary.json")
    history = coverage["history"]
    stages = coverage["stages"]
    final_stage = stages[-1]
    evals = extract_evals(run_dir)
    return {
        **spec,
        "run_dir": str(run_dir),
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
        "last_n_val_mean": last_n_mean([float(row["val/query_accuracy"]) for row in val_rows]),
        "best_to_last_drop": float(best_row["val/query_accuracy"]) - float(last_row["val/query_accuracy"]),
        "coverage_checkpoints": {str(step): metrics_at_step(history, step) for step in (10, 50, 100, 200)},
        "history_steps": [int(row["step"]) for row in history],
        "history_task_visit": [float(row["task_visit_coverage"]) for row in history],
        "history_task_grad": [float(row["task_gradient_coverage"]) for row in history],
        "history_val_steps": [int(row["step"]) for row in val_rows],
        "history_val_accuracy": [float(row["val/query_accuracy"]) for row in val_rows],
        "final_stage": {
            "stage_name": final_stage["stage_name"],
            "start_step": int(final_stage["start_step"]),
            "task_visit_histogram": final_stage["task_visit_histogram"],
            "task_grad_histogram": final_stage["task_grad_histogram"],
            "task_visit_entropy_at": final_stage.get("task_visit_entropy_at", {}),
            "task_visit_gini_at": final_stage.get("task_visit_gini_at", {}),
            "task_time_to_visit": final_stage.get("task_time_to_visit", {}),
            "task_time_to_grad": final_stage.get("task_time_to_grad", {}),
        },
        "split_events": transition_stage_records(stages),
        "evals": evals,
    }


def summarize_arm(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    out = {
        "count": len(records),
        "best_val": mean_std([r["best_val"]["query_accuracy"] for r in records]),
        "last_val": mean_std([r["last_val"]["query_accuracy"] for r in records]),
        "last_n_val_mean": mean_std([r["last_n_val_mean"] for r in records]),
        "best_to_last_drop": mean_std([r["best_to_last_drop"] for r in records]),
        "best_query_first_hop": mean_std([r["best_val"]["query_first_hop_home_rate"] for r in records]),
        "best_writer_first_hop": mean_std([r["best_val"]["writer_first_hop_home_rate"] for r in records]),
        "best_delivery": mean_std([r["best_val"]["query_delivery_rate"] for r in records]),
        "best_home_to_output": mean_std([r["best_val"]["query_home_to_output_rate"] for r in records]),
        "task_visit_10": mean_std([r["coverage_checkpoints"]["10"]["task_visit_coverage"] for r in records]),
        "task_grad_10": mean_std([r["coverage_checkpoints"]["10"]["task_gradient_coverage"] for r in records]),
        "task_ge5_50": mean_std([r["coverage_checkpoints"]["50"]["task_nodes_ge5_visit_fraction"] for r in records]),
        "task_visit_gini_50": mean_std([r["coverage_checkpoints"]["50"]["task_visit_gini"] for r in records]),
    }
    eval_tags = sorted({tag for record in records for tag in record["evals"]})
    for tag in eval_tags:
        values = [r["evals"][tag]["query_accuracy"] for r in records if tag in r["evals"]]
        if values:
            out[f"eval_{tag}"] = mean_std(values)
    split_events = [event for record in records for event in record["split_events"] if event["parent_components"]]
    if split_events:
        out["selected_parent_utility_mean"] = mean_std([e["selected_parent_utility_mean"] for e in split_events])
        out["unselected_parent_utility_mean"] = mean_std([e["unselected_parent_utility_mean"] for e in split_events])
        out["selected_parent_child_usefulness_mean"] = mean_std(
            [e["selected_parent_child_usefulness_mean"] for e in split_events]
        )
        out["unselected_parent_child_usefulness_mean"] = mean_std(
            [e["unselected_parent_child_usefulness_mean"] for e in split_events]
        )
        out["utility_usefulness_correlation"] = mean_std(
            [e["utility_usefulness_correlation"] for e in split_events]
        )
        components: dict[str, list[float]] = defaultdict(list)
        usefulness: list[float] = []
        for event in split_events:
            for parent, child_usefulness in event["parent_child_usefulness"].items():
                component = event["parent_components"].get(parent)
                if not component:
                    continue
                usefulness.append(float(child_usefulness))
                for key in ("visit", "grad", "success", "score"):
                    components[key].append(float(component.get(key, 0.0)))
        if usefulness:
            out["component_usefulness_correlation"] = {
                key: pearson(values, usefulness) for key, values in components.items() if len(values) == len(usefulness)
            }
    mutation_events = [
        event
        for record in records
        for event in record["split_events"]
        if event.get("mutated_child_usefulness_win_rate", 0.0) or event.get("mutated_child_visit_share_mean", 0.0)
    ]
    if mutation_events:
        for key in (
            "mutated_child_usefulness_win_rate",
            "mutated_child_visit_share_mean",
            "mutated_child_grad_share_mean",
            "mutated_child_usefulness_share_mean",
            "sibling_divergence",
        ):
            out[key] = mean_std([event[key] for event in mutation_events])
    return out


def mean_of(records: list[dict[str, Any]], field: str) -> list[float]:
    return [record[field]["query_accuracy"] for record in records]


def metric_values(records: list[dict[str, Any]], metric: str) -> list[float]:
    if metric == "best_val":
        return [r["best_val"]["query_accuracy"] for r in records]
    if metric == "last_val":
        return [r["last_val"]["query_accuracy"] for r in records]
    if metric == "last_n_val_mean":
        return [r["last_n_val_mean"] for r in records]
    if metric.startswith("eval_"):
        tag = metric.removeprefix("eval_")
        return [r["evals"][tag]["query_accuracy"] for r in records if tag in r["evals"]]
    raise KeyError(metric)


def plot_accuracy_bars(summary: dict[str, Any]) -> None:
    arms = ["B", "C", "U", "UM", "US", "UG"]
    labels = ["B\nstaged", "C\nclone", "U\nutility", "UM\nutil+mut", "US\nno-success", "UG\nno-grad"]
    colors = ["#66c2a5", "#fc8d62", "#a6d854", "#ffd92f", "#8da0cb", "#e78ac3"]
    metrics = [("best_val", "Best Val"), ("last_val", "Last Val"), ("eval_best_k6", "Best K6")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, (metric_key, title) in zip(axes, metrics):
        means = [summary["arm_summary"].get(arm, {}).get(metric_key, {}).get("mean", 0.0) for arm in arms]
        stds = [summary["arm_summary"].get(arm, {}).get(metric_key, {}).get("std", 0.0) for arm in arms]
        ax.bar(labels, means, yerr=stds, capsize=4, color=colors)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylim(0.0, max(0.7, max(means) + max(stds) + 0.05))
    fig.savefig(REPORTS / "v9_mean_std_accuracy_bars.png", dpi=180)
    plt.close(fig)


def plot_coverage_curves(records: list[dict[str, Any]], key: str, title: str, filename: str) -> None:
    colors = {"B": "#66c2a5", "C": "#fc8d62", "U": "#a6d854", "UM": "#ffd92f", "US": "#8da0cb", "UG": "#e78ac3"}
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for arm in ("B", "C", "U", "UM", "US", "UG"):
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        mean_series = series_mean([record[key] for record in arm_records])
        ax.plot(range(1, len(mean_series) + 1), mean_series, label=arm, color=colors[arm])
    ax.set_xlabel("Optimizer Step")
    ax.set_ylabel("Coverage Fraction")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.savefig(REPORTS / filename, dpi=180)
    plt.close(fig)


def plot_stability(records: list[dict[str, Any]]) -> None:
    arms = ["B", "C", "U", "UM", "US", "UG"]
    colors = {"B": "#66c2a5", "C": "#fc8d62", "U": "#a6d854", "UM": "#ffd92f", "US": "#8da0cb", "UG": "#e78ac3"}
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for arm in arms:
        xs = [record["best_val"]["query_accuracy"] for record in records if record["arm"] == arm]
        ys = [record["last_val"]["query_accuracy"] for record in records if record["arm"] == arm]
        if xs:
            ax.scatter(xs, ys, color=colors[arm], label=arm)
    lim = max([0.1] + [record["best_val"]["query_accuracy"] for record in records] + [record["last_val"]["query_accuracy"] for record in records]) + 0.02
    ax.plot([0.0, lim], [0.0, lim], linestyle="--", color="black", alpha=0.5)
    ax.set_xlabel("Best Val Accuracy")
    ax.set_ylabel("Last Val Accuracy")
    ax.set_title("Best-to-Last Stability by Seed")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.savefig(REPORTS / "v9_best_to_last_stability.png", dpi=180)
    plt.close(fig)


def plot_late_stage_curves(records: list[dict[str, Any]]) -> None:
    colors = {"B": "#66c2a5", "C": "#fc8d62", "U": "#a6d854", "UM": "#ffd92f", "US": "#8da0cb", "UG": "#e78ac3"}
    fig, ax = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for arm in ("B", "C", "U", "UM", "US", "UG"):
        arm_records = [record for record in records if record["arm"] == arm]
        if not arm_records:
            continue
        histories: list[list[float]] = []
        for record in arm_records:
            steps = record["history_val_steps"]
            values = record["history_val_accuracy"]
            trimmed = [value for step, value in zip(steps, values) if step >= 2100]
            if trimmed:
                histories.append(trimmed)
        if histories:
            mean_series = series_mean(histories)
            ax.plot(range(len(mean_series)), mean_series, color=colors[arm], label=arm)
    ax.set_xlabel("Eval Index After Final Stage Start")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Late-Stage Rolling Validation")
    ax.grid(alpha=0.2)
    ax.legend(ncol=3)
    fig.savefig(REPORTS / "v9_late_stage_rolling_validation.png", dpi=180)
    plt.close(fig)


def plot_selected_vs_unselected(records: list[dict[str, Any]]) -> None:
    arms = ["C", "U", "UM", "US", "UG"]
    labels = [ARM_LABELS[arm] for arm in arms]
    selected = []
    unselected = []
    selected_use = []
    unselected_use = []
    for arm in arms:
        events = [event for record in records if record["arm"] == arm for event in record["split_events"] if event["parent_components"]]
        selected.append(mean([event["selected_parent_utility_mean"] for event in events]))
        unselected.append(mean([event["unselected_parent_utility_mean"] for event in events]))
        selected_use.append(mean([event["selected_parent_child_usefulness_mean"] for event in events]))
        unselected_use.append(mean([event["unselected_parent_child_usefulness_mean"] for event in events]))
    x = range(len(arms))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar([i - width / 2 for i in x], selected, width=width, label="selected")
    axes[0].bar([i + width / 2 for i in x], unselected, width=width, label="unselected")
    axes[0].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[0].set_title("Parent Utility: Selected vs Unselected")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()
    axes[1].bar([i - width / 2 for i in x], selected_use, width=width, label="selected")
    axes[1].bar([i + width / 2 for i in x], unselected_use, width=width, label="unselected")
    axes[1].set_xticks(list(x), labels, rotation=20, ha="right")
    axes[1].set_title("Later Child Usefulness")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()
    fig.savefig(REPORTS / "v9_selected_vs_unselected_utility.png", dpi=180)
    plt.close(fig)


def plot_child_usefulness_vs_parent_utility(records: list[dict[str, Any]]) -> None:
    colors = {"U": "#1b9e77", "UM": "#d95f02", "US": "#7570b3", "UG": "#e7298a"}
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for arm in ("U", "UM", "US", "UG"):
        xs: list[float] = []
        ys: list[float] = []
        for record in records:
            if record["arm"] != arm:
                continue
            for event in record["split_events"]:
                for parent, usefulness in event["parent_child_usefulness"].items():
                    components = event["parent_components"].get(parent)
                    if components is None:
                        continue
                    xs.append(float(components.get("score", 0.0)))
                    ys.append(float(usefulness))
        if xs:
            ax.scatter(xs, ys, alpha=0.35, s=18, label=f"{arm} r={pearson(xs, ys):.2f}", color=colors[arm])
    ax.set_xlabel("Parent Utility Score")
    ax.set_ylabel("Later Child Usefulness")
    ax.set_title("Child Usefulness vs Parent Utility")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / "v9_child_usefulness_vs_parent_utility.png", dpi=180)
    plt.close(fig)


def plot_mutation_vs_sibling(records: list[dict[str, Any]]) -> None:
    mutation_events = [
        event
        for record in records
        if record["arm"] in {"UM", "UM_followup", "transfer_UM"}
        for event in record["split_events"]
        if event["mutated_child_usefulness_share_mean"] > 0.0 or event["sibling_divergence"] > 0.0
    ]
    if not mutation_events:
        return
    labels = ["visit share", "grad share", "usefulness share", "win rate"]
    means = [
        mean([event["mutated_child_visit_share_mean"] for event in mutation_events]),
        mean([event["mutated_child_grad_share_mean"] for event in mutation_events]),
        mean([event["mutated_child_usefulness_share_mean"] for event in mutation_events]),
        mean([event["mutated_child_usefulness_win_rate"] for event in mutation_events]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].bar(labels, means, color="#d95f02")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Mutated Child vs Sibling Shares")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=20)
    divergences = [event["sibling_divergence"] for event in mutation_events]
    axes[1].hist(divergences, bins=min(12, max(4, len(divergences))), color="#7570b3", alpha=0.8)
    axes[1].set_title("Sibling Divergence After Mutation")
    axes[1].set_xlabel("Divergence")
    axes[1].grid(alpha=0.2)
    fig.savefig(REPORTS / "v9_mutated_child_vs_sibling_usefulness.png", dpi=180)
    plt.close(fig)


def plot_transfer(records: list[dict[str, Any]]) -> None:
    transfer_records = [record for record in records if record["arm"] in {"transfer_U", "transfer_UM"}]
    if not transfer_records:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for arm, color in (("transfer_U", "#1b9e77"), ("transfer_UM", "#d95f02")):
        arm_records = [record for record in transfer_records if record["arm"] == arm]
        ks = [4, 8, 12, 14]
        means = []
        for k in ks:
            tag = f"best_k{k}"
            vals = [record["evals"][tag]["query_accuracy"] for record in arm_records if tag in record["evals"]]
            means.append(mean(vals))
        ax.plot(ks, means, marker="o", label=arm.replace("transfer_", ""), color=color)
    ax.set_xlabel("Writers Per Episode")
    ax.set_ylabel("Best Checkpoint Query Accuracy")
    ax.set_title("H1 Transfer / Stress Test")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(REPORTS / "v9_transfer_comparison.png", dpi=180)
    plt.close(fig)


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    combined_arms = {
        "B": ["B"],
        "C": ["C"],
        "U": ["U", "U_followup"],
        "UM": ["UM", "UM_followup"],
        "US": ["US", "US_followup"],
        "UG": ["UG", "UG_followup"],
        "RM": ["RM"],
        "transfer_U": ["transfer_U"],
        "transfer_UM": ["transfer_UM"],
    }
    arm_summary = {
        arm: summarize_arm([record for record in records if record["arm"] in members])
        for arm, members in combined_arms.items()
        if any(record["arm"] in members for record in records)
    }
    detailed_arm_summary = {
        arm: summarize_arm([record for record in records if record["arm"] == arm])
        for arm in ARM_PATTERNS
        if any(record["arm"] == arm for record in records)
    }
    arm_records = {arm: [record for record in records if record["arm"] in members] for arm, members in combined_arms.items()}
    effects = {}
    for lhs, rhs in (("U", "C"), ("UM", "U"), ("U", "US"), ("U", "UG")):
        effects[f"{lhs}_minus_{rhs}"] = {
            metric: bootstrap_mean_diff(metric_values(arm_records[lhs], metric), metric_values(arm_records[rhs], metric), seed=hash((lhs, rhs, metric)) & 0xFFFFFFFF)
            for metric in ("best_val", "last_val", "eval_best_k6", "eval_best_k10")
        }
    return {
        "visible_gpu_count": 2,
        "benchmark": {
            "final_leaf_homes": 32,
            "stage_schedule": [4, 6, 8, 12, 16, 24, 32],
            "train_steps": 6500,
            "stage_steps": [250, 250, 300, 300, 400, 600, 4400],
            "followup_train_steps": 8000,
            "followup_stage_steps": [250, 250, 300, 300, 400, 600, 5900],
            "writers_per_episode_train": 2,
            "writers_per_episode_eval": [2, 6, 10],
            "start_node_pool_size": 2,
            "query_ttl_range": [2, 3],
            "bootstrap_steps": 75,
            "utility_formula_full": "z(visits)+z(grad)+0.75*z(success)",
            "utility_formula_nosuccess": "z(visits)+z(grad)",
            "utility_formula_nograd": "z(visits)+0.75*z(success)",
            "mutation_policy": "single child mutated with modest local perturbations on child-local routing/delay and small MLP weights",
            "transfer_regime": "H1: train writers=4, eval writers=4/8/12/14, same longplus stage schedule",
        },
        "runs": records,
        "arm_summary": arm_summary,
        "detailed_arm_summary": detailed_arm_summary,
        "effects": effects,
    }


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def fmt_effect(effect: dict[str, float]) -> str:
    return f"{effect['mean_diff']:+.4f} [{effect['ci_low']:+.4f}, {effect['ci_high']:+.4f}]"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(summary: dict[str, Any]) -> None:
    arm_summary = summary["arm_summary"]
    runs = summary["runs"]
    core_rows = []
    for arm in ("B", "C", "U", "UM", "US", "UG"):
        stat = arm_summary[arm]
        core_rows.append(
            [
                arm,
                str(stat["count"]),
                fmt_stat(stat["best_val"]),
                fmt_stat(stat["last_val"]),
                fmt_stat(stat["eval_best_k2"]),
                fmt_stat(stat["eval_best_k6"]),
                fmt_stat(stat["eval_best_k10"]),
            ]
        )
    per_seed_rows = []
    for record in runs:
        if record["arm"] in {"transfer_U", "transfer_UM"}:
            continue
        per_seed_rows.append(
            [
                record["arm"],
                str(record["seed"]),
                f"{record['best_val']['query_accuracy']:.4f}",
                f"{record['last_val']['query_accuracy']:.4f}",
                f"{record['evals'].get('best_k2', {}).get('query_accuracy', 0.0):.4f}",
                f"{record['evals'].get('best_k6', {}).get('query_accuracy', 0.0):.4f}",
                f"{record['evals'].get('best_k10', {}).get('query_accuracy', 0.0):.4f}",
            ]
        )
    transfer_rows = []
    for arm in ("transfer_U", "transfer_UM"):
        stat = arm_summary.get(arm)
        if not stat:
            continue
        transfer_rows.append(
            [
                arm,
                str(stat["count"]),
                fmt_stat(stat["best_val"]),
                fmt_stat(stat.get("eval_best_k4", {"mean": 0.0, "std": 0.0})),
                fmt_stat(stat.get("eval_best_k8", {"mean": 0.0, "std": 0.0})),
                fmt_stat(stat.get("eval_best_k12", {"mean": 0.0, "std": 0.0})),
                fmt_stat(stat.get("eval_best_k14", {"mean": 0.0, "std": 0.0})),
            ]
        )

    effects = summary["effects"]
    utility_corr = arm_summary["U"].get("component_usefulness_correlation", {})
    mutation_summary = arm_summary["UM"]
    text = f"""# APSGNN v9: Mutation And Utility

## What Changed From v8

V9 keeps the v8 hard selective-growth benchmark and asks two narrower questions: whether utility+mutation (UM) is a real upgrade over utility-only selective growth (U), and which parts of the utility score actually matter. The core architecture is unchanged from the v8 winner: the v3 key-centric first-hop router, the v4 implicit learned retrieval path, and the v8 selective-growth machinery with task-only coverage/utility accounting.

The main long schedule uses 6500 optimizer steps with stage steps `[250, 250, 300, 300, 400, 600, 4400]` for the selective active-node schedule `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`. Follow-up and transfer confirmation runs use an 8000-step `longplus` schedule with the final stage extended to 5900 steps. Bootstrap remains 75 steps at each stage start and is excluded from task-only coverage and utility scoring.

## Benchmark

- Final compute leaves: `32`
- Output node: `0`
- Task: write-then-query memory routing
- Core train writers/episode: `2`
- Core eval writers/episode: `2, 6, 10`
- Start node pool size: `2`
- Query TTL: `2..3`
- Max rollout steps: `12`
- Visible GPUs used: `{summary['visible_gpu_count']}`

## Utility Formulas

- Full score `U`: `z(visits) + z(grad) + 0.75 * z(success)`
- No-success `US`: `z(visits) + z(grad)`
- No-grad `UG`: `z(visits) + 0.75 * z(success)`

Mutation policy for `UM`: one child of each selected split is mutated with a modest local perturbation to child-local routing/delay heads and small MLP weights, preserving overall function as much as possible.

## Exact Configs Used

- Core anchors: `configs/v9_staged_static_selective_long.yaml`, `configs/v9_clone_selective_long.yaml`
- Core utility family: `configs/v9_utility_selective_long.yaml`, `configs/v9_utility_mutate_long.yaml`, `configs/v9_utility_nosuccess_long.yaml`, `configs/v9_utility_nograd_long.yaml`
- Longplus follow-up: `configs/v9_utility_selective_longplus.yaml`, `configs/v9_utility_mutate_longplus.yaml`, `configs/v9_utility_nosuccess_longplus.yaml`, `configs/v9_utility_nograd_longplus.yaml`
- Transfer H1: `configs/v9_transfer_h1_utility_longplus.yaml`, `configs/v9_transfer_h1_utility_mutate_longplus.yaml`

## Core Mean/Std Summary

{markdown_table(["Arm", "Seeds", "Best Val", "Last Val", "Best K2", "Best K6", "Best K10"], core_rows)}

## Per-Seed Core And Follow-Up Runs

{markdown_table(["Arm", "Seed", "Best Val", "Last Val", "Best K2", "Best K6", "Best K10"], per_seed_rows)}

## Effect Summaries

- `U - C`: last-val {fmt_effect(effects['U_minus_C']['last_val'])}, K6 {fmt_effect(effects['U_minus_C']['eval_best_k6'])}, K10 {fmt_effect(effects['U_minus_C']['eval_best_k10'])}
- `UM - U`: last-val {fmt_effect(effects['UM_minus_U']['last_val'])}, K6 {fmt_effect(effects['UM_minus_U']['eval_best_k6'])}, K10 {fmt_effect(effects['UM_minus_U']['eval_best_k10'])}
- `U - US`: last-val {fmt_effect(effects['U_minus_US']['last_val'])}, K6 {fmt_effect(effects['U_minus_US']['eval_best_k6'])}, K10 {fmt_effect(effects['U_minus_US']['eval_best_k10'])}
- `U - UG`: last-val {fmt_effect(effects['U_minus_UG']['last_val'])}, K6 {fmt_effect(effects['U_minus_UG']['eval_best_k6'])}, K10 {fmt_effect(effects['U_minus_UG']['eval_best_k10'])}

## Follow-Up Round

The initial 20-run matrix left `U`, `UM`, `US`, and `UG` too close to settle from the 6500-step schedule alone. I therefore chose the “late-emerging” follow-up path: two additional `U` seeds and two additional `UM` seeds on the 8000-step longplus schedule, plus one extra `US` seed and one extra `UG` seed. This isolates whether the utility-score and mutation effects mainly appear deep into the final 32-node stage.

The longplus follow-up raised the ceiling for both `U` and `UM`, but it did not flip the overall diagnosis. `UM` showed higher upside on some longplus seeds, yet the combined six-seed `U` aggregate remained more stable from best to last checkpoint (`{fmt_stat(arm_summary['U']['best_to_last_drop'])}` vs `{fmt_stat(arm_summary['UM']['best_to_last_drop'])}` drop), and `U` retained a small edge on combined last-checkpoint mean.

I did not rerun random-mutate (`RM`) in v9. The initial matrix never established a clean `UM > U` advantage, so the most diagnostic follow-up was to extend `U/UM/US/UG` into a longer final-stage regime rather than spend the follow-up budget on a mutation-without-utility arm before mutation itself had cleared the stronger utility-only baseline.

## Transfer / Stress Round

I used stress regime `H1` because it increases retrieval and late-stage generalization pressure directly while keeping the selective schedule fixed:

- Train writers/episode: `4`
- Eval writers/episode: `4, 8, 12, 14`
- Same longplus stage schedule and bootstrap logic

The two strongest core contenders were `U` and `UM`, so the transfer round compares exactly those.

{markdown_table(["Arm", "Seeds", "Best Val", "Best K4", "Best K8", "Best K12", "Best K14"], transfer_rows)}

Under H1, `U` generalizes more reliably than `UM`: the mean best-checkpoint transfer accuracies are higher for `U` at `K4`, `K8`, and `K12`, while `UM` only catches up at `K14`.

## Mechanism Diagnostics

Early task-only coverage is already saturated across the selective-growth arms by step 10, so the core `U/UM/US/UG` differences are not being driven by broader early exploration. The remaining signal is late-stage allocation.

For `U`, selected parents have much higher utility than unselected eligible parents, and they also lead to more useful children:

- Selected parent utility: {fmt_stat(arm_summary['U']['selected_parent_utility_mean'])}
- Unselected parent utility: {fmt_stat(arm_summary['U']['unselected_parent_utility_mean'])}
- Selected parent child usefulness: {fmt_stat(arm_summary['U']['selected_parent_child_usefulness_mean'])}
- Unselected parent child usefulness: {fmt_stat(arm_summary['U']['unselected_parent_child_usefulness_mean'])}
- Parent utility vs later child usefulness correlation: {fmt_stat(arm_summary['U']['utility_usefulness_correlation'])}

Component-level usefulness correlations in `U`:

- Visit component: `{utility_corr.get('visit', 0.0):.4f}`
- Grad component: `{utility_corr.get('grad', 0.0):.4f}`
- Success component: `{utility_corr.get('success', 0.0):.4f}`
- Full score: `{utility_corr.get('score', 0.0):.4f}`

These diagnostics match the ablation results: visit and gradient signals are doing most of the work, while the success-conditioned traffic term is comparatively weak. Removing the success term (`US`) does not hurt materially. Removing the gradient term (`UG`) is somewhat worse overall, especially on late-stage stability.

Mutation diagnostics across `UM` runs:

- Mutated child usefulness win rate over its sibling: {fmt_stat(mutation_summary['mutated_child_usefulness_win_rate'])}
- Mutated child visit share: {fmt_stat(mutation_summary['mutated_child_visit_share_mean'])}
- Mutated child grad share: {fmt_stat(mutation_summary['mutated_child_grad_share_mean'])}
- Mutated child usefulness share: {fmt_stat(mutation_summary['mutated_child_usefulness_share_mean'])}
- Sibling divergence: {fmt_stat(mutation_summary['sibling_divergence'])}

Mutation is real, but it is not yet a reliable average win: mutated children do diverge, and some seeds exploit that well, but the aggregate usefulness win rate stays below 0.5 and the stability penalty is still visible in `UM`.

## Conclusions

1. **UM vs U:** `UM` is not yet a reliable upgrade over `U`. On combined means they tie on best validation (`{fmt_stat(arm_summary['U']['best_val'])}` vs `{fmt_stat(arm_summary['UM']['best_val'])}`) but `U` is slightly better on last-checkpoint stability and transfer.
2. **Utility score parts:** the gradient term appears more important than the success-conditioned term. `US` stays essentially tied with `U`, while `UG` is weaker overall. The raw component correlations also point the same way: gradient and visit predict later child usefulness better than success traffic.
3. **Mechanism:** the `U` advantage is late-stage allocation, not early task coverage. Early task-only coverage saturates quickly for all selective arms, but the selected-vs-unselected utility and child-usefulness gaps remain strong, especially in later transitions.
4. **Stress transfer:** the core winner remains `U`. Under the harder H1 regime, `U` beats `UM` on most transfer writer densities and degrades more gracefully. There is still no evidence that mutation helps more generally without utility selection; v9 never established a strong enough `UM > U` margin to justify elevating mutation-first variants ahead of score refinement.

## Best Next Experiment

The next best move is **refining the utility score**, not switching to crossover or pruning/merge yet. The score is already doing real work, but the current success-conditioned term is not pulling its weight and mutation is still too noisy. The clean next experiment is to keep the same hard benchmark and long final-stage schedule, refine the utility target around later child usefulness or final-stage contribution, and only then reconsider conditional mutation.

## Artifacts

- Summary JSON: [summary_metrics_v9.json](/home/catid/gnn/reports/summary_metrics_v9.json)
- Mean/std bars: [v9_mean_std_accuracy_bars.png](/home/catid/gnn/reports/v9_mean_std_accuracy_bars.png)
- Stability: [v9_best_to_last_stability.png](/home/catid/gnn/reports/v9_best_to_last_stability.png)
- Visit coverage: [v9_task_visit_coverage_curves.png](/home/catid/gnn/reports/v9_task_visit_coverage_curves.png)
- Gradient coverage: [v9_task_gradient_coverage_curves.png](/home/catid/gnn/reports/v9_task_gradient_coverage_curves.png)
- Late-stage rolling validation: [v9_late_stage_rolling_validation.png](/home/catid/gnn/reports/v9_late_stage_rolling_validation.png)
- Selected vs unselected utility: [v9_selected_vs_unselected_utility.png](/home/catid/gnn/reports/v9_selected_vs_unselected_utility.png)
- Child usefulness vs parent utility: [v9_child_usefulness_vs_parent_utility.png](/home/catid/gnn/reports/v9_child_usefulness_vs_parent_utility.png)
- Mutated child diagnostics: [v9_mutated_child_vs_sibling_usefulness.png](/home/catid/gnn/reports/v9_mutated_child_vs_sibling_usefulness.png)
- Transfer comparison: [v9_transfer_comparison.png](/home/catid/gnn/reports/v9_transfer_comparison.png)
"""
    (REPORTS / "final_report_v9_mutation_and_utility.md").write_text(text)


def main() -> None:
    specs = latest_specs()
    records = [load_run(spec) for spec in specs]
    summary = build_summary(records)
    (REPORTS / "summary_metrics_v9.json").write_text(json.dumps(summary, indent=2))
    core_records = [record for record in records if record["arm"] in {"B", "C", "U", "UM", "US", "UG"}]
    plot_accuracy_bars(summary)
    plot_stability(core_records)
    plot_coverage_curves(core_records, "history_task_visit", "V9 Task-Only Visit Coverage", "v9_task_visit_coverage_curves.png")
    plot_coverage_curves(core_records, "history_task_grad", "V9 Task-Only Gradient Coverage", "v9_task_gradient_coverage_curves.png")
    plot_late_stage_curves(core_records)
    plot_selected_vs_unselected(records)
    plot_child_usefulness_vs_parent_utility(records)
    plot_mutation_vs_sibling(records)
    plot_transfer(records)
    write_report(summary)


if __name__ == "__main__":
    main()
