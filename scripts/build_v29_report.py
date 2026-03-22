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
SUMMARY_PATH = REPORTS / "summary_metrics_v29.json"
REPORT_PATH = REPORTS / "final_report_v29_stageadaptive_default_validation.md"

SELECTORS = {
    "visitonly": {
        "label": "V",
        "prefix": "v26-{regime}-visitonly-32-{schedule}",
    },
    "visit_taskgrad_half": {
        "label": "VT-0.5",
        "prefix": "v26-{regime}-visit_taskgrad_half-32-{schedule}",
    },
    "stageadaptive_late_half": {
        "label": "StageLate-0.5",
        "prefix": "v27-{regime}-stageadaptive_late_half-32-{schedule}",
    },
}

PHASES = {
    "core_m": {
        "regime": "core",
        "schedule": "m",
        "writers": [2, 6, 10],
        "seeds": [1234, 2234, 3234, 4234],
        "selectors": ["visitonly", "visit_taskgrad_half", "stageadaptive_late_half"],
        "title": "Core-M Matched",
    },
    "t1_m": {
        "regime": "t1",
        "schedule": "m",
        "writers": [4, 8, 12, 14],
        "seeds": [1234, 2234, 3234],
        "selectors": ["visitonly", "visit_taskgrad_half", "stageadaptive_late_half"],
        "title": "T1-M Matched",
    },
    "t2a_l": {
        "regime": "t2a",
        "schedule": "l",
        "writers": [4, 8, 12, 14],
        "seeds": [5234, 6234, 7234],
        "selectors": ["visitonly", "visit_taskgrad_half", "stageadaptive_late_half"],
        "title": "T2a-L Matched",
    },
    "t1_l": {
        "regime": "t1",
        "schedule": "l",
        "writers": [4, 8, 12, 14],
        "seeds": [8234, 9234, 12234],
        "selectors": ["visitonly", "visit_taskgrad_half", "stageadaptive_late_half"],
        "title": "T1-L Matched",
    },
    "core_l": {
        "regime": "core",
        "schedule": "l",
        "writers": [2, 6, 10],
        "seeds": [13234, 14234],
        "selectors": ["visitonly", "stageadaptive_late_half"],
        "prefix_overrides": {
            "visitonly": "v27-{regime}-visitonly-32-{schedule}",
        },
        "title": "Core-L Fresh Rerun",
    },
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


def latest_run(prefix: str, seed: int) -> Path | None:
    matches = sorted(RUNS.glob(f"*-{prefix}-s{seed}"))
    return matches[-1] if matches else None


def is_complete_run(run_dir: Path) -> bool:
    return (run_dir / "last.pt").exists() and (run_dir / "metrics.jsonl").exists()


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def summarize_run(run_dir: Path, selector: str, writers: list[int], phase: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    record: dict[str, Any] = {
        "phase": phase,
        "selector": selector,
        "selector_label": SELECTORS[selector]["label"],
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
    return record


def phase_records(phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    records: list[dict[str, Any]] = []
    for selector in spec["selectors"]:
        prefix_template = spec.get("prefix_overrides", {}).get(selector, SELECTORS[selector]["prefix"])
        prefix = prefix_template.format(regime=spec["regime"], schedule=spec["schedule"])
        for seed in spec["seeds"]:
            run_dir = latest_run(prefix, seed)
            if run_dir is None:
                continue
            if not is_complete_run(run_dir):
                continue
            records.append(summarize_run(run_dir, selector, spec["writers"], phase, seed))
    return records


def summarize_phase(records: list[dict[str, Any]], writers: list[int], selectors: list[str]) -> dict[str, dict[str, Any]]:
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    out: dict[str, dict[str, Any]] = {}
    for selector in selectors:
        selector_records = [record for record in records if record["selector"] == selector]
        if not selector_records:
            continue
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "last5_val_mean": mean_std([record["last5_val_mean"] for record in selector_records]),
            "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in selector_records]),
            "dense_mean": mean_std([record["dense_mean"] for record in selector_records]),
            "last_dense_mean": mean_std([record["last_dense_mean"] for record in selector_records]),
        }
        for key in dense_keys:
            out[selector][key] = mean_std([record[key] for record in selector_records])
    return out


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
                PHASES[record["phase"]]["title"],
                record["selector_label"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['last5_val_mean']:.4f}",
                f"{record['dense_mean']:.4f}",
            ]
        )
    return markdown_table(headers, rows)


def phase_table(summary: dict[str, dict[str, Any]], phase: str) -> str:
    writers = PHASES[phase]["writers"]
    headers = ["Selector", "Best", "Last", "Dense", *[f"K{writer}" for writer in writers[1:]]]
    rows: list[list[str]] = []
    for selector in PHASES[phase]["selectors"]:
        payload = summary.get(selector)
        if not payload:
            continue
        row = [
            SELECTORS[selector]["label"],
            fmt_stat(payload["best_val"]),
            fmt_stat(payload["last_val"]),
            fmt_stat(payload["dense_mean"]),
        ]
        for writer in writers[1:]:
            row.append(fmt_stat(payload[f"k{writer}"]))
        rows.append(row)
    return markdown_table(headers, rows)


def plot_phase(summary: dict[str, dict[str, Any]], phase: str, path: Path) -> None:
    selectors = [selector for selector in PHASES[phase]["selectors"] if selector in summary]
    if not selectors:
        return
    labels = [SELECTORS[selector]["label"] for selector in selectors]
    metrics = [("best_val", "Best"), ("last_val", "Last"), ("dense_mean", "Dense")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.0))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        means = [summary[selector][metric]["mean"] for selector in selectors]
        stds = [summary[selector][metric]["std"] for selector in selectors]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(PHASES[phase]["title"])
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def average_metric(summary: dict[str, dict[str, dict[str, Any]]], phases: list[str], selector: str, metric: str) -> float:
    values = [summary[phase][selector][metric]["mean"] for phase in phases if selector in summary.get(phase, {})]
    return mean(values)


def recommend(phase_summaries: dict[str, dict[str, dict[str, Any]]]) -> str:
    if "t1_m" not in phase_summaries or "core_l" not in phase_summaries:
        return "Pending: waiting for matched T1-M and fresh Core-L runs."

    stage = "stageadaptive_late_half"
    vt = "visit_taskgrad_half"
    v = "visitonly"
    transfer_phases = ["t1_m", "t1_l", "t2a_l"]
    stage_transfer_dense = average_metric(phase_summaries, transfer_phases, stage, "dense_mean")
    vt_transfer_dense = average_metric(phase_summaries, transfer_phases, vt, "dense_mean")
    v_transfer_dense = average_metric(phase_summaries, transfer_phases, v, "dense_mean")
    stage_core_dense = average_metric(phase_summaries, ["core_m", "core_l"], stage, "dense_mean")
    v_core_dense = average_metric(phase_summaries, ["core_m", "core_l"], v, "dense_mean")

    if stage_transfer_dense > vt_transfer_dense and stage_core_dense >= v_core_dense:
        return "StageLate-0.5 clears both the matched transfer bar and the fresh home-regime check strongly enough to be a plausible new default."
    if stage_transfer_dense > vt_transfer_dense and stage_core_dense < v_core_dense:
        return "StageLate-0.5 looks best as a transfer-specialist rule, but it does not clear the fresh home-regime bar against V."
    if vt_transfer_dense >= stage_transfer_dense and v_core_dense >= stage_core_dense:
        return "The existing split still stands: VT-0.5 remains the safest overall default, while V stays the stronger pure-home fallback."
    if stage_transfer_dense > v_transfer_dense and stage_core_dense == v_core_dense:
        return "Results remain regime-dependent: StageLate-0.5 improves transfer without establishing a clean home-regime edge."
    return "Matched results are still mixed; keep VT-0.5 as the default unless a transfer-specialist selector is explicitly desired."


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for phase, spec in PHASES.items():
        records = phase_records(phase)
        all_records.extend(records)
        phase_summaries[phase] = summarize_phase(records, spec["writers"], spec["selectors"])
        plot_phase(phase_summaries[phase], phase, REPORTS / f"v29_{phase}.png")

    recommendation = recommend(phase_summaries)
    summary = {
        "matched_seed_sets": {phase: spec["seeds"] for phase, spec in PHASES.items()},
        "phase_summaries": phase_summaries,
        "recommendation": recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sections = []
    for phase in ("core_m", "t1_m", "t2a_l", "t1_l", "core_l"):
        sections.append(f"## {PHASES[phase]['title']}\n\n{phase_table(phase_summaries.get(phase, {}), phase)}")

    report = f"""# APSGNN v29 StageAdaptive Default Validation

This round asks the remaining post-v28 question directly: does `StageLate-0.5` do enough on matched transfer and fresh home-regime reruns to replace `VT-0.5` as the default selector, or does the previous mixed recommendation still stand?

The exact matched seed sets in this report are:
- `Core-M`: {", ".join(str(seed) for seed in PHASES["core_m"]["seeds"])}
- `T1-M`: {", ".join(str(seed) for seed in PHASES["t1_m"]["seeds"])}
- `T2a-L`: {", ".join(str(seed) for seed in PHASES["t2a_l"]["seeds"])}
- `T1-L`: {", ".join(str(seed) for seed in PHASES["t1_l"]["seeds"])}
- `Core-L`: {", ".join(str(seed) for seed in PHASES["core_l"]["seeds"])}

{"\n\n".join(sections) if sections else "_Pending_"}

## Recommendation

{recommendation}

## Completed matched runs

{run_table(all_records) if all_records else "_Pending_"}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
