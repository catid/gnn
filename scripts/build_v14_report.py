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


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def stat(values: list[float]) -> dict[str, Any]:
    return {
        "mean": mean(values),
        "std": std(values),
        "per_seed": values,
        "seeds": len(values),
    }


def fmt(entry: dict[str, Any]) -> str:
    return f"{entry['mean']:.4f} ± {entry['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def summarize_runs(run_dirs: list[Path], eval_tags: list[str]) -> dict[str, Any]:
    best_vals: list[float] = []
    last_vals: list[float] = []
    metrics = {tag: [] for tag in eval_tags}
    for run_dir in run_dirs:
        rows = [row for row in read_jsonl(run_dir / "metrics.jsonl") if "val/query_accuracy" in row]
        best = max(rows, key=lambda row: float(row["val/query_accuracy"]))
        last = rows[-1]
        best_vals.append(float(best["val/query_accuracy"]))
        last_vals.append(float(last["val/query_accuracy"]))
        for tag in eval_tags:
            payload = read_json(run_dir / f"eval_best_{tag}.json")
            metric_blob = payload.get("metrics", payload)
            metrics[tag].append(float(metric_blob["query_accuracy"]))
    out = {"best": stat(best_vals), "last": stat(last_vals)}
    out.update({tag: stat(values) for tag, values in metrics.items()})
    return out


def normalize_v10_core(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": {
            "mean": float(entry["best_val"]["mean"]),
            "std": float(entry["best_val"]["std"]),
            "seeds": int(entry["count"]),
        },
        "last": {
            "mean": float(entry["last_val"]["mean"]),
            "std": float(entry["last_val"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k2": {
            "mean": float(entry["best_k2"]["mean"]),
            "std": float(entry["best_k2"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k6": {
            "mean": float(entry["best_k6"]["mean"]),
            "std": float(entry["best_k6"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k10": {
            "mean": float(entry["best_k10"]["mean"]),
            "std": float(entry["best_k10"]["std"]),
            "seeds": int(entry["count"]),
        },
    }


def normalize_v10_h1(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": {
            "mean": float(entry["best_val"]["mean"]),
            "std": float(entry["best_val"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k4": {
            "mean": float(entry["best_k4"]["mean"]),
            "std": float(entry["best_k4"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k8": {
            "mean": float(entry["best_k8"]["mean"]),
            "std": float(entry["best_k8"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k12": {
            "mean": float(entry["best_k12"]["mean"]),
            "std": float(entry["best_k12"]["std"]),
            "seeds": int(entry["count"]),
        },
        "k14": {
            "mean": float(entry["best_k14"]["mean"]),
            "std": float(entry["best_k14"]["std"]),
            "seeds": int(entry["count"]),
        },
    }


def normalize_flat_core(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": {
            "mean": float(entry["best_val_mean"]),
            "std": float(entry["best_val_std"]),
            "seeds": int(entry["seeds"]),
        },
        "last": {
            "mean": float(entry["last_val_mean"]),
            "std": float(entry["last_val_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k2": {
            "mean": float(entry["k2_mean"]),
            "std": float(entry["k2_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k6": {
            "mean": float(entry["k6_mean"]),
            "std": float(entry["k6_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k10": {
            "mean": float(entry["k10_mean"]),
            "std": float(entry["k10_std"]),
            "seeds": int(entry["seeds"]),
        },
    }


def normalize_flat_h1(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "best": {
            "mean": float(entry["best_val_mean"]),
            "std": float(entry["best_val_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k4": {
            "mean": float(entry["k4_mean"]),
            "std": float(entry["k4_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k8": {
            "mean": float(entry["k8_mean"]),
            "std": float(entry["k8_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k12": {
            "mean": float(entry["k12_mean"]),
            "std": float(entry["k12_std"]),
            "seeds": int(entry["seeds"]),
        },
        "k14": {
            "mean": float(entry["k14_mean"]),
            "std": float(entry["k14_std"]),
            "seeds": int(entry["seeds"]),
        },
    }


def plot_summary(core: dict[str, Any], transfer_h1: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    core_arms = [
        ("v10_querygrad", "v10 QG"),
        ("v10_querygrad_mutate", "v10 QG+M"),
        ("v11_conditional_mutation", "v11 CM"),
        ("v12_adaptive_mutation", "v12 AM"),
        ("v13_high_confidence_mutation", "v13 HCM"),
        ("v14_component_agreement", "v14 CAG"),
    ]
    core_means = [core[name]["k10"]["mean"] for name, _ in core_arms]
    core_stds = [core[name]["k10"]["std"] for name, _ in core_arms]
    axes[0].bar([label for _, label in core_arms], core_means, yerr=core_stds, capsize=4)
    axes[0].set_title("Core K10")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=20)

    h1_arms = [
        ("v10_querygrad", "v10 QG"),
        ("v13_high_confidence_mutation", "v13 HCM"),
        ("v14_component_agreement", "v14 CAG"),
    ]
    h1_means = [transfer_h1[name]["k12"]["mean"] for name, _ in h1_arms]
    h1_stds = [transfer_h1[name]["k12"]["std"] for name, _ in h1_arms]
    axes[1].bar([label for _, label in h1_arms], h1_means, yerr=h1_stds, capsize=4)
    axes[1].set_title("H1 K12")
    axes[1].grid(axis="y", alpha=0.2)

    fig.savefig(REPORTS / "v14_component_agreement_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    v10 = read_json(REPORTS / "summary_metrics_v10.json")["summary"]
    v11 = read_json(REPORTS / "summary_metrics_v11.json")
    v12 = read_json(REPORTS / "summary_metrics_v12.json")
    v13 = read_json(REPORTS / "summary_metrics_v13.json")

    v14_core_runs = [
        RUNS / "20260318-113852-v14-utility-querygrad-agree-longplus-s1234",
        RUNS / "20260318-114530-v14-utility-querygrad-agree-longplus-s2234",
        RUNS / "20260318-115206-v14-utility-querygrad-agree-longplus-s3234",
        RUNS / "20260318-115843-v14-utility-querygrad-agree-longplus-s4234",
    ]
    v14_transfer_runs = [
        RUNS / "20260318-121124-v14-transfer-h1-utility-querygrad-agree-longplus-s1234",
        RUNS / "20260318-121818-v14-transfer-h1-utility-querygrad-agree-longplus-s2234",
    ]

    core = {
        "v10_querygrad": normalize_v10_core(v10["querygrad"]),
        "v10_querygrad_mutate": normalize_v10_core(v10["querygrad_mutate"]),
        "v11_conditional_mutation": normalize_flat_core(v11["core"]["v11_conditional_mutation"]),
        "v12_adaptive_mutation": normalize_flat_core(v12["core"]["v12_adaptive_mutation"]),
        "v13_high_confidence_mutation": v13["core"]["v13_high_confidence_mutation"],
        "v14_component_agreement": summarize_runs(v14_core_runs, ["k2", "k6", "k10"]),
    }
    transfer_h1 = {
        "v10_querygrad": normalize_v10_h1(v10["transfer_querygrad"]),
        "v10_querygrad_mutate": normalize_v10_h1(v10["transfer_querygrad_mutate"]),
        "v11_conditional_mutation": normalize_flat_h1(v11["transfer_h1"]["v11_conditional_mutation"]),
        "v12_adaptive_mutation": normalize_flat_h1(v12["transfer_h1"]["v12_adaptive_mutation"]),
        "v13_high_confidence_mutation": v13["transfer_h1"]["v13_high_confidence_mutation"],
        "v14_component_agreement": summarize_runs(v14_transfer_runs, ["k4", "k8", "k12", "k14"]),
    }

    summary = {
        "core": core,
        "transfer_h1": transfer_h1,
        "conclusion": {
            "default": "utility-only querygrad selective growth",
            "component_agreement_status": "competitive core gate but weaker-than-baseline transfer behavior",
            "next_move": "keep mutation off by default; if mutation is revisited, use a stronger confidence or stagnation trigger",
        },
    }
    (REPORTS / "summary_metrics_v14.json").write_text(json.dumps(summary, indent=2))
    plot_summary(core, transfer_h1)

    core_rows = []
    for arm, label in (
        ("v10_querygrad", "v10 `querygrad`"),
        ("v10_querygrad_mutate", "v10 `querygrad+mutate`"),
        ("v11_conditional_mutation", "v11 conditional mutation"),
        ("v12_adaptive_mutation", "v12 adaptive mutation"),
        ("v13_high_confidence_mutation", "v13 high-confidence mutation"),
        ("v14_component_agreement", "v14 component-agreement mutation"),
    ):
        entry = core[arm]
        core_rows.append(
            [
                label,
                str(entry["best"]["seeds"]),
                f"`{fmt(entry['best'])}`",
                f"`{fmt(entry['last'])}`",
                f"`{fmt(entry['k2'])}`",
                f"`{fmt(entry['k6'])}`",
                f"`{fmt(entry['k10'])}`",
            ]
        )

    transfer_rows = []
    for arm, label in (
        ("v10_querygrad", "v10 `querygrad`"),
        ("v10_querygrad_mutate", "v10 `querygrad+mutate`"),
        ("v11_conditional_mutation", "v11 conditional mutation"),
        ("v12_adaptive_mutation", "v12 adaptive mutation"),
        ("v13_high_confidence_mutation", "v13 high-confidence mutation"),
        ("v14_component_agreement", "v14 component-agreement mutation"),
    ):
        entry = transfer_h1[arm]
        transfer_rows.append(
            [
                label,
                str(entry["best"]["seeds"]),
                f"`{fmt(entry['best'])}`",
                f"`{fmt(entry['k4'])}`",
                f"`{fmt(entry['k8'])}`",
                f"`{fmt(entry['k12'])}`",
                f"`{fmt(entry['k14'])}`",
            ]
        )

    lines = [
        "# APSGNN v14 Component-Agreement Mutation",
        "",
        "## Goal",
        "",
        "Validate whether mutation should require agreement between the two active `querygrad` utility components rather than a score margin alone.",
        "",
        "The v14 rule keeps the v10 `querygrad` selector fixed and mutates only on the final `24 -> 32` transition, only when the selected parent clears three gates:",
        "",
        "- mutate only at stage index `6` (`24 -> 32`)",
        "- require `selected_score >= reference + 0.75`",
        "- require both `visit_z >= 0.25` and `query_grad_z >= 0.25`",
        "",
        "This tests whether component agreement improves over the v13 high-confidence margin gate without reopening broader mutation.",
        "",
        "## Configs",
        "",
        "- Core: `configs/v14_utility_querygrad_agree_longplus.yaml`",
        "- Transfer H1: `configs/v14_transfer_h1_utility_querygrad_agree_longplus.yaml`",
        "- Visible GPUs actually used: `2`",
        "",
        "Core v14 run directories:",
        "",
        *[f"- `{run}`" for run in [str(path.relative_to(ROOT)) for path in v14_core_runs]],
        "",
        "Transfer v14 run directories:",
        "",
        *[f"- `{run}`" for run in [str(path.relative_to(ROOT)) for path in v14_transfer_runs]],
        "",
        "## Core Results",
        "",
        markdown_table(["Arm", "Seeds", "Best val", "Last val", "K2", "K6", "K10"], core_rows),
        "",
        "## H1 Transfer Results",
        "",
        "H1 keeps the selective-growth mechanism fixed and raises retrieval pressure:",
        "",
        "- train writers per episode: `4`",
        "- eval writers per episode: `4, 8, 12, 14`",
        "",
        markdown_table(["Arm", "Seeds", "Best val", "K4", "K8", "K12", "K14"], transfer_rows),
        "",
        "## Interpretation",
        "",
        "v14 is a negative but useful result.",
        "",
        "On the core long schedule, v14 component-agreement mutation is credible: it matches v13 on mean best-val, improves mean last-val over v13, and keeps strong `K2/K10`. But it does not improve `K6` over plain `querygrad`, so the core case was already marginal.",
        "",
        "The H1 transfer pair settles the question. v14 falls below both plain `querygrad` and v13 high-confidence mutation on mean `K8`, `K12`, and `K14`, with only one strong seed carrying the pair. That means the extra component-agreement gate is still not enough to turn mutation into a robust default. It narrows the damage relative to broader mutation, but it does not beat staying mutation-free.",
        "",
        "## Conclusion",
        "",
        "The most defensible default remains **utility-only `querygrad` selective growth**.",
        "",
        "v14 component-agreement mutation should be treated as an informative failure mode:",
        "",
        "- stronger late-core stability than some earlier mutation gates",
        "- but still worse transfer means than both v10 `querygrad` and v13 high-confidence mutation",
        "- so mutation should remain opt-in and highly constrained, not the default growth policy",
        "",
        f"Plot: `reports/{(REPORTS / 'v14_component_agreement_comparison.png').name}`",
        "",
    ]

    (REPORTS / "final_report_v14_component_agreement_mutation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
