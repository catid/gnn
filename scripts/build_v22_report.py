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
SUMMARY_PATH = REPORTS / "summary_metrics_v22.json"
REPORT_PATH = REPORTS / "final_report_v22_query_dominant_refinement.md"

SELECTORS = [
    "querygradonly",
    "visit_taskgrad_half",
    "querygrad_visit_quarter",
    "querygrad_visit_half",
]

DISPLAY = {
    "querygradonly": "Q",
    "visit_taskgrad_half": "VT-half",
    "querygrad_visit_quarter": "QV-0.25",
    "querygrad_visit_half": "QV-0.5",
}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
}

LABELS = {
    "core_l": "Core-L",
    "t1_l": "T1-L",
    "core_xl": "Core-XL",
    "t1_xl": "T1-XL",
    "t2a_xl": "T2a-XL",
    "t1_r": "T1-R",
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
    prefix = f"v22-{regime}-{selector}-32-{schedule}"
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
    return sorted(summary, key=lambda selector: summary[selector]["screen_composite"]["mean"], reverse=True)


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def fmt_stat(payload: dict[str, float]) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def plot_phase(summary: dict[str, dict[str, Any]], selectors: list[str], title: str, metrics: list[tuple[str, str]], path: Path) -> None:
    if not selectors:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.0 * len(metrics), 4.2))
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


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for regime, schedules in {"core": ["l", "xl"], "t1": ["l", "xl", "r"], "t2a": ["xl"]}.items():
        for schedule in schedules:
            phase_records_list: list[dict[str, Any]] = []
            for selector in SELECTORS:
                phase_records_list.extend(phase_records(selector, regime, schedule))
            if not phase_records_list:
                continue
            phase = f"{regime}_{schedule}"
            all_records.extend(phase_records_list)
            phase_summaries[phase] = summarize_phase(phase_records_list, regime)

    core_screen_rank = rank_phase(phase_summaries.get("core_l", {}))
    transfer_screen_rank = rank_phase(phase_summaries.get("t1_l", {}))
    combined_scores: dict[str, float] = {}
    for selector in set(core_screen_rank) | set(transfer_screen_rank):
        combined_scores[selector] = (
            phase_summaries.get("core_l", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
            + phase_summaries.get("t1_l", {}).get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
        )
    top3_screened = sorted(combined_scores, key=lambda selector: combined_scores[selector], reverse=True)[:3]

    confirmatory_summary = {phase: phase_summaries[phase] for phase in ("core_xl", "t1_xl") if phase in phase_summaries}
    stress_summary = {phase: phase_summaries[phase] for phase in ("t2a_xl",) if phase in phase_summaries}
    rerun_summary = {phase: phase_summaries[phase] for phase in ("t1_r",) if phase in phase_summaries}

    confirm_candidates = sorted(
        set(phase_summaries.get("core_xl", {})) | set(phase_summaries.get("t1_xl", {})) | set(top3_screened)
    )
    confirm_scores: dict[str, float] = {}
    for selector in confirm_candidates:
        confirm_scores[selector] = (
            phase_summaries.get("core_xl", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("core_xl", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t1_xl", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_xl", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
        )
    top2_finalists = sorted(confirm_scores, key=lambda selector: confirm_scores[selector], reverse=True)[:2]

    if "core_l" in phase_summaries:
        plot_phase(
            phase_summaries["core_l"],
            core_screen_rank,
            "v22 Core-L Screening",
            [("dense_mean", "Dense Mean"), ("last_val", "Last Val"), ("screen_composite", "Composite")],
            REPORTS / "v22_core_screening.png",
        )
    if "t1_l" in phase_summaries:
        plot_phase(
            phase_summaries["t1_l"],
            transfer_screen_rank,
            "v22 T1-L Screening",
            [("dense_mean", "Dense Mean"), ("last_val", "Last Val"), ("screen_composite", "Composite")],
            REPORTS / "v22_t1_screening.png",
        )
    if "core_xl" in phase_summaries:
        plot_phase(
            phase_summaries["core_xl"],
            top3_screened,
            "v22 Core-XL Confirmation",
            [("dense_mean", "Dense Mean"), ("last_val", "Last Val")],
            REPORTS / "v22_core_xl.png",
        )
    if "t1_xl" in phase_summaries:
        plot_phase(
            phase_summaries["t1_xl"],
            top3_screened,
            "v22 T1-XL Confirmation",
            [("dense_mean", "Dense Mean"), ("last_val", "Last Val")],
            REPORTS / "v22_t1_xl.png",
        )

    summary = {
        "visible_gpu_count": 2,
        "budgets": {"L": 3570, "XL": 4590, "R": 4590},
        "table_eval_keys": ["k2", "k6", "k10", "k4", "k8", "k12", "k14"],
        "records": all_records,
        "phase_summaries": phase_summaries,
        "top3_screened": top3_screened,
        "confirm_candidates": confirm_candidates,
        "top2_finalists": top2_finalists,
        "confirmatory_summary": confirmatory_summary,
        "stress_summary": stress_summary,
        "rerun_summary": rerun_summary,
        "interpretation": (
            "v22 compares the two v21 finalists (`Q` and `VT-half`) against query-dominant visit+query hybrids. "
            "The final selector call should come from the combined evidence of Core-L, T1-L, Core-XL, T1-XL, "
            "T2a-XL, and the fresh-seed `T1-R` reruns."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    core_screen = phase_summaries.get("core_l", {})
    t1_screen = phase_summaries.get("t1_l", {})
    REPORT_PATH.write_text(build_report(summary, core_screen, t1_screen), encoding="utf-8")


def build_report(summary: dict[str, Any], core_screen: dict[str, Any], t1_screen: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# APSGNN v22 Query-Dominant Selector Refinement")
    lines.append("")
    lines.append("## What Changed From v21")
    lines.append("")
    lines.append("- Focused on the v21 finalists plus two query-dominant hybrids instead of repeating the whole family funnel.")
    lines.append("- Kept 32 leaves fixed and shifted the comparison toward longer transfer-heavy schedules.")
    lines.append("- Split fresh-seed reruns into a dedicated `T1-R` phase.")
    lines.append("")
    lines.append("## Budgets")
    lines.append("")
    lines.append(f"- Visible GPU count used: `{summary['visible_gpu_count']}`")
    lines.append(f"- Chosen schedules: `L={summary['budgets']['L']}`, `XL={summary['budgets']['XL']}`, `R={summary['budgets']['R']}` steps")
    lines.append("- Screening composite: `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`")
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
            ["Selector", "Core-L composite", "T1-L composite", "Core-L dense", "T1-L dense"],
            screening_rows,
        ))
    else:
        lines.append("Screening runs have not completed yet.")
    lines.append("")
    lines.append(f"- Top 3 promoted to XL confirmation: `{', '.join(DISPLAY[s] for s in summary['top3_screened'])}`")
    lines.append(f"- Current top 2 finalists for stress/reruns: `{', '.join(DISPLAY[s] for s in summary['top2_finalists'])}`")
    lines.append("")
    lines.append("## Confirmatory Summary")
    lines.append("")
    if summary["confirmatory_summary"]:
        lines.append(markdown_table(
            ["Selector", "Core-XL last", "T1-XL last", "Core-XL dense", "T1-XL dense"],
            [
                [
                    DISPLAY[selector],
                    fmt_stat(summary["confirmatory_summary"]["core_xl"][selector]["last_val"]) if selector in summary["confirmatory_summary"]["core_xl"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["t1_xl"][selector]["last_val"]) if selector in summary["confirmatory_summary"]["t1_xl"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["core_xl"][selector]["dense_mean"]) if selector in summary["confirmatory_summary"]["core_xl"] else "-",
                    fmt_stat(summary["confirmatory_summary"]["t1_xl"][selector]["dense_mean"]) if selector in summary["confirmatory_summary"]["t1_xl"] else "-",
                ]
                for selector in summary["top3_screened"]
            ],
        ))
    else:
        lines.append("Confirmatory runs have not completed yet.")
    lines.append("")
    lines.append("## Stress And Fresh-Seed Checks")
    lines.append("")
    if summary["stress_summary"]:
        lines.append(markdown_table(
            ["Selector", "T2a-XL last", "T2a-XL dense", "T1-R last", "T1-R dense"],
            [
                [
                    DISPLAY[selector],
                    fmt_stat(summary["stress_summary"]["t2a_xl"][selector]["last_val"]) if selector in summary["stress_summary"]["t2a_xl"] else "-",
                    fmt_stat(summary["stress_summary"]["t2a_xl"][selector]["dense_mean"]) if selector in summary["stress_summary"]["t2a_xl"] else "-",
                    fmt_stat(summary["rerun_summary"]["t1_r"][selector]["last_val"]) if "t1_r" in summary["rerun_summary"] and selector in summary["rerun_summary"]["t1_r"] else "-",
                    fmt_stat(summary["rerun_summary"]["t1_r"][selector]["dense_mean"]) if "t1_r" in summary["rerun_summary"] and selector in summary["rerun_summary"]["t1_r"] else "-",
                ]
                for selector in summary["top2_finalists"]
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


if __name__ == "__main__":
    main()
