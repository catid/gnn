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
SUMMARY_PATH = REPORTS / "summary_metrics_v21.json"
REPORT_PATH = REPORTS / "final_report_v21_selector_family_funnel.md"

SELECTORS = [
    "visitonly",
    "visit_taskgrad",
    "visit_querygrad",
    "full_querygrad",
    "querygradonly",
    "visit_taskgrad_half",
    "visit_querygrad_half",
]

DISPLAY = {
    "visitonly": "V",
    "visit_taskgrad": "VT",
    "visit_querygrad": "VQ",
    "full_querygrad": "VTQ",
    "querygradonly": "Q",
    "visit_taskgrad_half": "VT-half",
    "visit_querygrad_half": "VQ-half",
}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
}

LABELS = {
    "core_s": "Core-S",
    "t1_s": "T1-S",
    "core_m": "Core-M",
    "t1_m": "T1-M",
    "t2a_l": "T2a-L",
    "core_l": "Core-L",
    "t1_l": "T1-L",
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
        "best_query_first_hop": float(best.get("val/query_first_hop_home_rate", 0.0)),
        "best_delivery": float(best.get("val/query_delivery_rate", 0.0)),
        "best_home_to_output": float(best.get("val/query_home_to_output_rate", 0.0)),
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
        split_stages = [stage.get("split_stats", {}) for stage in stages if stage.get("split_stats")]
        record["score_usefulness_corr"] = mean(
            [float(stage.get("utility_usefulness_correlation", 0.0)) for stage in split_stages]
        )
        record["score_traffic_corr"] = mean(
            [float(stage.get("utility_traffic_correlation", 0.0)) for stage in split_stages]
        )
        record["selected_usefulness"] = mean(
            [float(stage.get("selected_parent_child_usefulness_mean", 0.0)) for stage in split_stages]
        )
        record["unselected_usefulness"] = mean(
            [float(stage.get("unselected_parent_child_usefulness_mean", 0.0)) for stage in split_stages]
        )
    else:
        record["score_usefulness_corr"] = 0.0
        record["score_traffic_corr"] = 0.0
        record["selected_usefulness"] = 0.0
        record["unselected_usefulness"] = 0.0
    return record


def phase_records(selector: str, regime: str, schedule: str) -> list[dict[str, Any]]:
    prefix = f"v21-{regime}-{selector}-32-{schedule}"
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
            "score_usefulness_corr": mean_std([record["score_usefulness_corr"] for record in selector_records]),
            "score_traffic_corr": mean_std([record["score_traffic_corr"] for record in selector_records]),
            "selected_usefulness": mean_std([record["selected_usefulness"] for record in selector_records]),
            "unselected_usefulness": mean_std([record["unselected_usefulness"] for record in selector_records]),
        }
        for key in dense_keys:
            out[selector][key] = mean_std([record[key] for record in selector_records])
    return out


def rank_phase(summary: dict[str, dict[str, Any]]) -> list[str]:
    return sorted(
        summary,
        key=lambda selector: summary[selector]["screen_composite"]["mean"],
        reverse=True,
    )


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def plot_phase(summary: dict[str, dict[str, Any]], selectors: list[str], title: str, metrics: list[tuple[str, str]], path: Path) -> None:
    if not selectors:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.4), constrained_layout=True)
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


def build_report(summary: dict[str, Any]) -> str:
    core_screen = summary["phase_summaries"].get("core_s", {})
    t1_screen = summary["phase_summaries"].get("t1_s", {})
    lines: list[str] = []
    lines.append("# APSGNN v21 Selector Family Funnel")
    lines.append("")
    lines.append("## What Changed From v20")
    lines.append("")
    lines.append("- Switched from an infeasible scale-first campaign to a tractable 32-leaf funnel.")
    lines.append("- Added single-GPU wrappers so the two visible GPUs can run independent jobs in parallel.")
    lines.append("- Expanded the selector family to include `VT-half` and `VQ-half` before retiring gradient-bearing selectors.")
    lines.append("- Used a screening -> transfer screening -> confirmation -> stress -> fresh-seed rerun funnel instead of another fantasy-scale full matrix.")
    lines.append("")
    lines.append("## Calibration And Budgets")
    lines.append("")
    lines.append(f"- Visible GPU count used: `{summary['visible_gpu_count']}`")
    lines.append(f"- Calibration runtime at 100 steps: `~{summary['calibration']['runtime_seconds_approx']}s`")
    lines.append(f"- Chosen schedules: `S={summary['budgets']['S']}`, `M={summary['budgets']['M']}`, `L={summary['budgets']['L']}` steps")
    lines.append("- Schedule ranking composite: `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`")
    lines.append("")
    lines.append("## Completed Runs")
    lines.append("")
    lines.append(run_table(summary["records"], summary["table_eval_keys"]))
    lines.append("")
    lines.append("## Screening Summary")
    lines.append("")
    screening_rows = [
        [
            DISPLAY[selector],
            fmt_stat(core_screen[selector]["screen_composite"]) if selector in core_screen else "-",
            fmt_stat(t1_screen[selector]["screen_composite"]) if selector in t1_screen else "-",
            fmt_stat(core_screen[selector]["dense_mean"]) if selector in core_screen else "-",
            fmt_stat(t1_screen[selector]["dense_mean"]) if selector in t1_screen else "-",
        ]
        for selector in SELECTORS
        if selector in core_screen or selector in t1_screen
    ]
    if screening_rows:
        lines.append(markdown_table(
            ["Selector", "Core-S composite", "T1-S composite", "Core-S dense", "T1-S dense"],
            screening_rows,
        ))
        lines.append("")
        lines.append(f"- Top 4 promoted to T1 screening: `{', '.join(DISPLAY[s] for s in summary['top4_screening'])}`")
        lines.append(f"- Top 2 promoted to confirmation/stress: `{', '.join(DISPLAY[s] for s in summary['top2_overall'])}`")
    else:
        lines.append("No screening runs have completed yet.")
    lines.append("")
    lines.append("## Confirmatory Summary")
    lines.append("")
    if summary["confirmatory_summary"]:
        lines.append(markdown_table(
            ["Selector", "Core-M", "T1-M", "Core-M dense", "T1-M dense"],
            [
                [
                    DISPLAY[selector],
                    fmt_stat(summary["confirmatory_summary"]["core_m"][selector]["last_val"]) if selector in summary["confirmatory_summary"]["core_m"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["t1_m"][selector]["last_val"]) if selector in summary["confirmatory_summary"]["t1_m"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["core_m"][selector]["dense_mean"]) if selector in summary["confirmatory_summary"]["core_m"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["t1_m"][selector]["dense_mean"]) if selector in summary["confirmatory_summary"]["t1_m"] else "-",
                ]
                for selector in summary["top2_overall"]
            ],
        ))
    else:
        lines.append("Confirmatory runs have not completed yet.")
    lines.append("")
    lines.append("## Stress And Fresh-Seed Checks")
    lines.append("")
    if summary["stress_summary"]:
        lines.append(markdown_table(
            ["Selector", "T2a-L last", "T2a-L dense", "Core-L/T1-L rerun last"],
            [
                [
                    DISPLAY[selector],
                    fmt_stat(summary["stress_summary"]["t2a_l"][selector]["last_val"]) if selector in summary["stress_summary"]["t2a_l"] else "-",
                    fmt_stat(summary["stress_summary"]["t2a_l"][selector]["dense_mean"]) if selector in summary["stress_summary"]["t2a_l"] else "-",
                    fmt_stat(summary["rerun_summary"][summary["rerun_phase"]][selector]["last_val"]) if summary["rerun_phase"] and selector in summary["rerun_summary"].get(summary["rerun_phase"], {}) else "-",
                ]
                for selector in summary["top2_overall"]
            ],
        ))
    else:
        lines.append("Stress and rerun rounds have not completed yet.")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(summary["interpretation"])
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for regime, schedules in {"core": ["s", "m", "l"], "t1": ["s", "m", "l"], "t2a": ["l"]}.items():
        for schedule in schedules:
            phase = f"{regime}_{schedule}"
            phase_records_list: list[dict[str, Any]] = []
            for selector in SELECTORS:
                phase_records_list.extend(phase_records(selector, regime, schedule))
            if not phase_records_list:
                continue
            all_records.extend(phase_records_list)
            phase_summaries[phase] = summarize_phase(phase_records_list, regime)

    core_screen_rank = rank_phase(phase_summaries.get("core_s", {}))
    top4_screening = core_screen_rank[:4]
    if "t1_s" in phase_summaries:
        transfer_rank = rank_phase(phase_summaries["t1_s"])
        combined_scores: dict[str, float] = {}
        for selector in set(core_screen_rank) | set(transfer_rank):
            combined_scores[selector] = (
                phase_summaries.get("core_s", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
                + phase_summaries.get("t1_s", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
            )
        top2_overall = sorted(combined_scores, key=lambda selector: combined_scores[selector], reverse=True)[:2]
    else:
        top2_overall = core_screen_rank[:2]

    confirmatory_summary = {
        phase: phase_summaries[phase]
        for phase in ("core_m", "t1_m")
        if phase in phase_summaries
    }
    stress_summary = {
        phase: phase_summaries[phase]
        for phase in ("t2a_l",)
        if phase in phase_summaries
    }
    rerun_phase = None
    if "t1_l" in phase_summaries:
        rerun_phase = "t1_l"
    elif "core_l" in phase_summaries:
        rerun_phase = "core_l"
    rerun_summary = {phase: phase_summaries[phase] for phase in ("core_l", "t1_l") if phase in phase_summaries}

    screening_plot = REPORTS / "v21_screening_summary.png"
    if "core_s" in phase_summaries:
        plot_phase(phase_summaries["core_s"], core_screen_rank, "v21 Core-S", [("dense_mean", "Dense Mean"), ("last_val", "Last Val"), ("screen_composite", "Composite")], screening_plot)
    if "t1_s" in phase_summaries:
        plot_phase(phase_summaries["t1_s"], rank_phase(phase_summaries["t1_s"]), "v21 T1-S", [("dense_mean", "Dense Mean"), ("last_val", "Last Val"), ("screen_composite", "Composite")], REPORTS / "v21_t1_screening_summary.png")
    if "core_m" in phase_summaries:
        plot_phase(phase_summaries["core_m"], top2_overall, "v21 Confirmatory Core-M", [("dense_mean", "Dense Mean"), ("last_val", "Last Val")], REPORTS / "v21_confirmatory_core.png")
    if "t1_m" in phase_summaries:
        plot_phase(phase_summaries["t1_m"], top2_overall, "v21 Confirmatory T1-M", [("dense_mean", "Dense Mean"), ("last_val", "Last Val")], REPORTS / "v21_confirmatory_t1.png")

    summary = {
        "visible_gpu_count": 2,
        "budgets": {"S": 1530, "M": 2550, "L": 3570},
        "calibration": {"runtime_seconds_approx": 44.0},
        "table_eval_keys": ["k2", "k6", "k10", "k4", "k8", "k12", "k14"],
        "records": all_records,
        "phase_summaries": phase_summaries,
        "top4_screening": top4_screening,
        "top2_overall": top2_overall,
        "confirmatory_summary": confirmatory_summary,
        "stress_summary": stress_summary,
        "rerun_phase": rerun_phase,
        "rerun_summary": rerun_summary,
        "interpretation": (
            "v21 rejects `visitonly` as the default once the selector family is screened more broadly at 32 leaves. "
            "The real contest is between `Q` (`querygrad-only`) and `VT-half` (`visit + 0.5 * task_grad`). "
            "`VT-half` is stronger on the mid-budget transfer confirmation (`T1-M`) and on Core-M dense evals, while "
            "`Q` is stronger on the longer transfer reruns (`T1-L`) and has the better last-checkpoint result on the "
            "ingress-stress round (`T2a-L`). The honest v21 diagnosis is therefore regime- and schedule-dependent rather "
            "than a universal winner: use `Q` as the better long-transfer selector, keep `VT-half` as the strongest "
            "moderate-budget dense/stability challenger, and do not keep `visitonly` as the unchallenged default."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
