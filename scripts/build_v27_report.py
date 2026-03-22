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
SUMMARY_PATH = REPORTS / "summary_metrics_v27.json"
REPORT_PATH = REPORTS / "final_report_v27_stageadaptive_refinement.md"
V26_SUMMARY_PATH = REPORTS / "summary_metrics_v26.json"

STAGEADAPTIVE_SELECTORS = [
    "stageadaptive_early_half",
    "stageadaptive_late_half",
    "stageadaptive_final_half",
    "stageadaptive_late_0375",
    "stageadaptive_late_0625",
]

BASELINE_SELECTORS = ["visitonly", "visit_taskgrad_half"]
V26_EQUIVALENTS = {
    "stageadaptive_late_half": "stageadaptive_vt",
}

DISPLAY = {
    "visitonly": "V",
    "visit_taskgrad_half": "VT-0.5",
    "stageadaptive_early_half": "StageEarly-0.5",
    "stageadaptive_late_half": "StageLate-0.5",
    "stageadaptive_final_half": "StageFinal-0.5",
    "stageadaptive_late_0375": "StageLate-0.375",
    "stageadaptive_late_0625": "StageLate-0.625",
}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
}

PHASE_LABELS = {
    "core_s": "Core-S",
    "core_m": "Core-M",
    "t1_s": "T1-S",
    "t1_m": "T1-M",
    "t1_l": "T1-L",
    "t2a_l": "T2a-L",
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


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def screening_composite(record: dict[str, Any]) -> float:
    return 0.45 * float(record["dense_mean"]) + 0.35 * float(record["last_val"]) + 0.20 * float(record["last5_val_mean"])


def summarize_run(run_dir: Path, selector: str, regime: str, schedule: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    writers = REGIME_WRITERS[regime]
    dense_keys = [f"k{writer}" for writer in writers[1:]]
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
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[key] for key in dense_keys])
    record["last_dense_mean"] = mean([record[f"last_{key}"] for key in dense_keys])
    record["screen_composite"] = screening_composite(record)
    return record


def phase_records(selector: str, regime: str, schedule: str) -> list[dict[str, Any]]:
    prefix = f"v27-{regime}-{selector}-32-{schedule}"
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
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "last5_val_mean": mean_std([record["last5_val_mean"] for record in selector_records]),
            "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in selector_records]),
            "dense_mean": mean_std([record["dense_mean"] for record in selector_records]),
            "last_dense_mean": mean_std([record["last_dense_mean"] for record in selector_records]),
            "screen_composite": mean_std([record["screen_composite"] for record in selector_records]),
        }
        for key in dense_keys:
            out[selector][key] = mean_std([record[key] for record in selector_records])
    return out


def rank_summary(summary: dict[str, dict[str, Any]], selectors: list[str]) -> list[str]:
    selectors = [selector for selector in selectors if selector in summary]
    return sorted(selectors, key=lambda selector: summary[selector]["screen_composite"]["mean"], reverse=True)


def combined_rank(summary_a: dict[str, dict[str, Any]], summary_b: dict[str, dict[str, Any]], selectors: list[str]) -> list[str]:
    selectors = [selector for selector in selectors if selector in summary_a or selector in summary_b]
    return sorted(
        selectors,
        key=lambda selector: summary_a.get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
        + summary_b.get(selector, {}).get("screen_composite", {}).get("mean", 0.0),
        reverse=True,
    )


def baseline_payload_for_phase(
    selector: str,
    phase: str,
    phase_summaries: dict[str, dict[str, Any]],
    v26_summary: dict[str, Any],
) -> dict[str, Any] | None:
    if selector in phase_summaries.get(phase, {}):
        return phase_summaries[phase][selector]

    v26_phase_summaries = v26_summary.get("phase_summaries", {})
    if selector in v26_phase_summaries.get(phase, {}):
        return v26_phase_summaries[phase][selector]

    alias = V26_EQUIVALENTS.get(selector)
    if alias and alias in v26_phase_summaries.get(phase, {}):
        return v26_phase_summaries[phase][alias]

    if phase == "t1_l" and alias and alias in v26_summary.get("showdown_summary", {}):
        return v26_summary["showdown_summary"][alias]

    return None


def fmt_stat(payload: dict[str, float]) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_table(records: list[dict[str, Any]]) -> str:
    headers = ["Phase", "Sel", "Seed", "Best", "Last", "Last5", "Dense"]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["phase"], item["selector"], item["seed"])):
        rows.append(
            [
                PHASE_LABELS.get(record["phase"], record["phase"]),
                record["selector_label"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['last5_val_mean']:.4f}",
                f"{record['dense_mean']:.4f}",
            ]
        )
    return markdown_table(headers, rows)


def plot_summary(summary: dict[str, dict[str, Any]], selectors: list[str], title: str, path: Path) -> None:
    selectors = [selector for selector in selectors if selector in summary]
    if not selectors:
        return
    labels = [DISPLAY[selector] for selector in selectors]
    metrics = [("screen_composite", "Composite"), ("dense_mean", "Dense"), ("last_val", "Last")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.0))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        means = [summary[selector][metric]["mean"] for selector in selectors]
        stds = [summary[selector][metric]["std"] for selector in selectors]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}

    for regime in ("core", "t1", "t2a"):
        for schedule in ("s", "m", "l"):
            records: list[dict[str, Any]] = []
            for selector in STAGEADAPTIVE_SELECTORS:
                records.extend(phase_records(selector, regime, schedule))
            if not records:
                continue
            phase = f"{regime}_{schedule}"
            all_records.extend(records)
            phase_summaries[phase] = summarize_phase(records, regime)

    core_rank = rank_summary(phase_summaries.get("core_s", {}), STAGEADAPTIVE_SELECTORS)
    promoted_t1 = core_rank[:3]
    combined = combined_rank(phase_summaries.get("core_s", {}), phase_summaries.get("t1_s", {}), promoted_t1)
    promoted_m = combined[:2]
    stage_winner = rank_summary(phase_summaries.get("t1_m", {}), promoted_m)[:1]
    stage_winner = stage_winner[0] if stage_winner else (promoted_m[0] if promoted_m else "")

    v26_summary = read_json(V26_SUMMARY_PATH) if V26_SUMMARY_PATH.exists() else {}
    equivalent_v26_selector = V26_EQUIVALENTS.get(stage_winner, "")

    final_compare_selectors = [selector for selector in [stage_winner, *BASELINE_SELECTORS] if selector]
    baseline_compare = {}
    for phase in ("t1_m", "t1_l", "core_m", "t2a_l"):
        baseline_compare[phase] = {}
        for selector in final_compare_selectors:
            payload = baseline_payload_for_phase(selector, phase, phase_summaries, v26_summary)
            if payload:
                baseline_compare[phase][selector] = payload

    summary = {
        "core_screen_rank": core_rank,
        "promoted_t1": promoted_t1,
        "combined_rank": combined,
        "promoted_m": promoted_m,
        "stage_winner": stage_winner,
        "equivalent_v26_selector": equivalent_v26_selector,
        "phase_summaries": phase_summaries,
        "baseline_compare": baseline_compare,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    plot_summary(phase_summaries.get("core_s", {}), core_rank, "v27 Core-S Stage-Adaptive", REPORTS / "v27_core_s.png")
    plot_summary(phase_summaries.get("t1_s", {}), promoted_t1, "v27 T1-S Stage-Adaptive", REPORTS / "v27_t1_s.png")
    plot_summary(phase_summaries.get("t1_m", {}), promoted_m, "v27 T1-M Stage-Adaptive", REPORTS / "v27_t1_m.png")

    screening_rows = [
        [
            DISPLAY[selector],
            fmt_stat(phase_summaries["core_s"][selector]["screen_composite"]),
            fmt_stat(phase_summaries["core_s"][selector]["dense_mean"]),
            fmt_stat(phase_summaries["core_s"][selector]["last_val"]),
        ]
        for selector in core_rank
        if "core_s" in phase_summaries and selector in phase_summaries["core_s"]
    ]
    t1_rows = [
        [
            DISPLAY[selector],
            fmt_stat(phase_summaries["t1_s"][selector]["screen_composite"]),
            fmt_stat(phase_summaries["t1_s"][selector]["dense_mean"]),
            fmt_stat(phase_summaries["t1_s"][selector]["last_val"]),
        ]
        for selector in promoted_t1
        if "t1_s" in phase_summaries and selector in phase_summaries["t1_s"]
    ]
    confirm_rows = [
        [
            DISPLAY[selector],
            fmt_stat(phase_summaries["t1_m"][selector]["best_val"]),
            fmt_stat(phase_summaries["t1_m"][selector]["last_val"]),
            fmt_stat(phase_summaries["t1_m"][selector]["dense_mean"]),
        ]
        for selector in promoted_m
        if "t1_m" in phase_summaries and selector in phase_summaries["t1_m"]
    ]

    compare_sections: list[str] = []
    for phase in ("t1_m", "t1_l", "core_m", "t2a_l"):
        winner_payload = baseline_compare.get(phase, {}).get(stage_winner) if stage_winner else None
        if not winner_payload:
            continue
        rows = []
        for selector in final_compare_selectors:
            payload = baseline_compare.get(phase, {}).get(selector)
            if not payload:
                continue
            rows.append(
                [
                    DISPLAY[selector],
                    fmt_stat(payload["best_val"]),
                    fmt_stat(payload["last_val"]),
                    fmt_stat(payload["dense_mean"]),
                ]
            )
        if rows:
            compare_sections.append(f"### {PHASE_LABELS[phase]}\n\n" + markdown_table(["Selector", "Best", "Last", "Dense"], rows))

    equivalence_note = ""
    if equivalent_v26_selector:
        equivalence_note = (
            f"\n## Equivalence Check\n\n"
            f"`{DISPLAY[stage_winner]}` is config-identical to the earlier v26 selector "
            f"`{equivalent_v26_selector}`: same stage schedule, same base visit-only utility, "
            f"same `adaptive_selector_stage_index_min`, and the same adaptive task-grad weight. "
            f"That means v27 did not discover a new late-stage rule here; it revalidated the same "
            f"late-switch policy under a cleaner focused screen.\n"
        )

    report = f"""# APSGNN v27 Stage-Adaptive Refinement

This round refines the stage-aware task-grad idea after v26. It screens new stage-switch variants against each other, then compares the surviving stage-adaptive winner back to the fixed v26 baselines `V` and `VT-0.5`.

## Core-S screening

{markdown_table(["Selector", "Composite", "Dense", "Last"], screening_rows) if screening_rows else "_Pending_"}

Promoted to `T1-S`: `{", ".join(DISPLAY[s] for s in promoted_t1) if promoted_t1 else "-"}`

## T1-S screening

{markdown_table(["Selector", "Composite", "Dense", "Last"], t1_rows) if t1_rows else "_Pending_"}

Promoted to `T1-M`: `{", ".join(DISPLAY[s] for s in promoted_m) if promoted_m else "-"}`

## T1-M confirmation

{markdown_table(["Selector", "Best", "Last", "Dense"], confirm_rows) if confirm_rows else "_Pending_"}

Current stage-adaptive winner: `{DISPLAY.get(stage_winner, "-")}`

{equivalence_note}

## Baseline Comparison

{"\n\n".join(compare_sections) if compare_sections else "_Pending_"}

## Completed v27 runs

{run_table(all_records) if all_records else "_Pending_"}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
