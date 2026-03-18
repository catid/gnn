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


def plot_summary(core: dict[str, Any], transfer3: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    core_arms = [
        ("v10_querygrad", "v10 QG"),
        ("v10_querygrad_mutate", "v10 QG+M"),
        ("v11_conditional_mutation", "v11 CM"),
        ("v12_adaptive_mutation", "v12 AM"),
        ("v13_high_confidence_mutation", "v13 HCM"),
    ]
    core_means = [core[name]["k10"]["mean"] for name, _ in core_arms]
    core_stds = [core[name]["k10"]["std"] for name, _ in core_arms]
    axes[0].bar([label for _, label in core_arms], core_means, yerr=core_stds, capsize=4)
    axes[0].set_title("Core K10")
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].tick_params(axis="x", rotation=20)

    h1_arms = [("v10_querygrad", "v10 QG"), ("v13_high_confidence_mutation", "v13 HCM")]
    h1_means = [transfer3[name]["k8"]["mean"] for name, _ in h1_arms]
    h1_stds = [transfer3[name]["k8"]["std"] for name, _ in h1_arms]
    axes[1].bar([label for _, label in h1_arms], h1_means, yerr=h1_stds, capsize=4)
    axes[1].set_title("Matched 3-Seed H1 K8")
    axes[1].grid(axis="y", alpha=0.2)

    fig.savefig(REPORTS / "v13_high_confidence_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    v10 = read_json(REPORTS / "summary_metrics_v10.json")["summary"]
    v11 = read_json(REPORTS / "summary_metrics_v11.json")
    v12 = read_json(REPORTS / "summary_metrics_v12.json")

    v13_core_runs = [
        RUNS / "20260318-102658-v13-utility-querygrad-hiconf-longplus-s1234",
        RUNS / "20260318-103333-v13-utility-querygrad-hiconf-longplus-s2234",
        RUNS / "20260318-104010-v13-utility-querygrad-hiconf-longplus-s3234",
        RUNS / "20260318-104647-v13-utility-querygrad-hiconf-longplus-s4234",
    ]
    v13_transfer_runs = [
        RUNS / "20260318-105603-v13-transfer-h1-utility-querygrad-hiconf-longplus-s1234",
        RUNS / "20260318-110259-v13-transfer-h1-utility-querygrad-hiconf-longplus-s2234",
    ]
    v10_transfer_3_runs = [
        RUNS / "20260318-075401-v10-transfer-h1-utility-querygrad-longplus-s1234",
        RUNS / "20260318-080752-v10-transfer-h1-utility-querygrad-longplus-s2234",
        RUNS / "20260318-111409-v10-transfer-h1-utility-querygrad-longplus-s3234",
    ]
    v13_transfer_3_runs = v13_transfer_runs + [
        RUNS / "20260318-112106-v13-transfer-h1-utility-querygrad-hiconf-longplus-s3234",
    ]

    core = {
        "v10_querygrad": normalize_v10_core(v10["querygrad"]),
        "v10_querygrad_mutate": normalize_v10_core(v10["querygrad_mutate"]),
        "v11_conditional_mutation": normalize_flat_core(v11["core"]["v11_conditional_mutation"]),
        "v12_adaptive_mutation": normalize_flat_core(v12["core"]["v12_adaptive_mutation"]),
        "v13_high_confidence_mutation": summarize_runs(v13_core_runs, ["k2", "k6", "k10"]),
    }
    transfer_h1 = {
        "v10_querygrad": normalize_v10_h1(v10["transfer_querygrad"]),
        "v10_querygrad_mutate": normalize_v10_h1(v10["transfer_querygrad_mutate"]),
        "v11_conditional_mutation": normalize_flat_h1(v11["transfer_h1"]["v11_conditional_mutation"]),
        "v12_adaptive_mutation": normalize_flat_h1(v12["transfer_h1"]["v12_adaptive_mutation"]),
        "v13_high_confidence_mutation": summarize_runs(v13_transfer_runs, ["k4", "k8", "k12", "k14"]),
    }
    transfer_h1_matched3 = {
        "v10_querygrad": summarize_runs(v10_transfer_3_runs, ["k4", "k8", "k12", "k14"]),
        "v13_high_confidence_mutation": summarize_runs(v13_transfer_3_runs, ["k4", "k8", "k12", "k14"]),
    }

    summary = {
        "core": core,
        "transfer_h1": transfer_h1,
        "transfer_h1_matched3": transfer_h1_matched3,
        "conclusion": {
            "default": "utility-only querygrad selective growth",
            "high_confidence_status": "credible core variant but not a transfer-default upgrade",
            "next_move": "keep mutation off by default; if revisited, gate on a stronger confidence signal or stagnation trigger",
        },
    }
    (REPORTS / "summary_metrics_v13.json").write_text(json.dumps(summary, indent=2))
    plot_summary(core, transfer_h1_matched3)

    core_rows = []
    for arm, label in (
        ("v10_querygrad", "v10 `querygrad`"),
        ("v10_querygrad_mutate", "v10 `querygrad+mutate`"),
        ("v11_conditional_mutation", "v11 conditional mutation"),
        ("v12_adaptive_mutation", "v12 adaptive mutation"),
        ("v13_high_confidence_mutation", "v13 high-confidence mutation"),
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

    matched_rows = []
    for arm, label in (
        ("v10_querygrad", "v10 `querygrad`"),
        ("v13_high_confidence_mutation", "v13 high-confidence mutation"),
    ):
        entry = transfer_h1_matched3[arm]
        matched_rows.append(
            [
                label,
                str(entry["best"]["seeds"]),
                f"`{fmt(entry['best'])}`",
                f"`{fmt(entry['last'])}`",
                f"`{fmt(entry['k4'])}`",
                f"`{fmt(entry['k8'])}`",
                f"`{fmt(entry['k12'])}`",
                f"`{fmt(entry['k14'])}`",
            ]
        )

    lines = [
        "# APSGNN v13 High-Confidence Mutation",
        "",
        "## Goal",
        "",
        "Validate a stricter mutation gate after v12.",
        "",
        "The v13 rule keeps the v10 `querygrad` selector fixed and mutates only on the final `24 -> 32` transition, only when the selected parent clears a large utility margin:",
        "",
        "- mutate only at stage index `6` (`24 -> 32`)",
        "- require `selected_score >= reference + 0.75`",
        "- `reference = max(unselected eligible score)` when available",
        "- otherwise fall back to the mean selected score",
        "",
        "This tests whether a very conservative, final-stage-only mutation gate can preserve late-stage core gains without repeating the H1 transfer regressions seen with broader mutation.",
        "",
        "## Configs",
        "",
        "- Core: `configs/v13_utility_querygrad_hiconf_longplus.yaml`",
        "- Transfer H1: `configs/v13_transfer_h1_utility_querygrad_hiconf_longplus.yaml`",
        "- Visible GPUs actually used: `2`",
        "",
        "Core v13 run directories:",
        "",
        *[f"- `{run}`" for run in [str(path.relative_to(ROOT)) for path in v13_core_runs]],
        "",
        "Transfer v13 run directories:",
        "",
        *[f"- `{run}`" for run in [str(path.relative_to(ROOT)) for path in v13_transfer_3_runs]],
        "",
        "## Core Results",
        "",
        markdown_table(
            ["Arm", "Seeds", "Best val", "Last val", "K2", "K6", "K10"],
            core_rows,
        ),
        "",
        "## H1 Transfer Results",
        "",
        "H1 keeps the selective-growth mechanism fixed and raises retrieval pressure:",
        "",
        "- train writers per episode: `4`",
        "- eval writers per episode: `4, 8, 12, 14`",
        "",
        markdown_table(
            ["Arm", "Seeds", "Best val", "K4", "K8", "K12", "K14"],
            transfer_rows,
        ),
        "",
        "## Matched 3-Seed Transfer Confirmation",
        "",
        "To reduce the variance in the key comparison, v10 `querygrad` and v13 high-confidence mutation were extended to a matched 3-seed H1 pair.",
        "",
        markdown_table(
            ["Arm", "Seeds", "Best val", "Last val", "K4", "K8", "K12", "K14"],
            matched_rows,
        ),
        "",
        "## Interpretation",
        "",
        "High-confidence mutation is a credible core variant, not a new default.",
        "",
        "On the core long schedule, v13 lifted mean `K10` over v12 and v11, stayed close to v10 `querygrad+mutate`, and improved mean best-val versus v11/v12. It did not clearly beat plain v10 `querygrad` on mean best-val or mean last-val.",
        "",
        "The matched 3-seed H1 comparison is the deciding result. Extending both v10 `querygrad` and v13 to three seeds did not flip the transfer conclusion: v13 is effectively tied on `K8/K12`, slightly ahead on `K4`, and clearly worse on `K14` and mean best-val. That means the stricter gate reduces some mutation damage, but still does not turn mutation into a robust transfer-default upgrade.",
        "",
        "## Conclusion",
        "",
        "The most defensible default remains **utility-only `querygrad` selective growth**.",
        "",
        "v13 high-confidence mutation is useful as a narrow late-stage variant:",
        "",
        "- stronger core `K10` than v11/v12",
        "- better transfer behavior than some broader mutation policies",
        "- but still not enough to replace mutation-free `querygrad` under the matched 3-seed H1 comparison",
        "",
        "## Next Move",
        "",
        "If mutation is revisited, it should be **more selective than v13**:",
        "",
        "- gate on a stronger confidence signal than a fixed margin",
        "- or trigger only under late-stage stagnation",
        "- otherwise keep mutation off and refine the utility score instead",
        "",
        f"Plot: `reports/{(REPORTS / 'v13_high_confidence_comparison.png').name}`",
        "",
    ]

    (REPORTS / "final_report_v13_high_confidence_mutation.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
