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
SUMMARY_PATH = REPORTS / "summary_metrics_v28.json"
REPORT_PATH = REPORTS / "final_report_v28_regime_dependent_selector_validation.md"

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
        "title": "Core-M",
    },
    "t2a_l": {
        "regime": "t2a",
        "schedule": "l",
        "writers": [4, 8, 12, 14],
        "seeds": [5234, 6234, 7234],
        "title": "T2a-L",
    },
    "t1_l": {
        "regime": "t1",
        "schedule": "l",
        "writers": [4, 8, 12, 14],
        "seeds": [8234, 9234, 12234],
        "title": "T1-L Matched",
    },
}

LEGACY_CONTEXT = {
    "t1_m": {
        "title": "T1-M Legacy Context",
        "summary_path": REPORTS / "summary_metrics_v26.json",
        "selectors": ["visitonly", "visit_taskgrad_half"],
    },
    "v27_t1_m": {
        "title": "T1-M Legacy Context",
        "summary_path": REPORTS / "summary_metrics_v27.json",
        "selectors": ["stageadaptive_late_half"],
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
    for selector, selector_spec in SELECTORS.items():
        prefix = selector_spec["prefix"].format(regime=spec["regime"], schedule=spec["schedule"])
        for seed in spec["seeds"]:
            run_dir = latest_run(prefix, seed)
            if run_dir is None:
                continue
            if not (run_dir / "metrics.jsonl").exists():
                continue
            records.append(summarize_run(run_dir, selector, spec["writers"], phase, seed))
    return records


def summarize_phase(records: list[dict[str, Any]], writers: list[int]) -> dict[str, dict[str, Any]]:
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    out: dict[str, dict[str, Any]] = {}
    for selector in SELECTORS:
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


def plot_phase(summary: dict[str, dict[str, Any]], phase: str, path: Path) -> None:
    selectors = [selector for selector in SELECTORS if selector in summary]
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


def load_legacy_t1_m() -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for payload in LEGACY_CONTEXT.values():
        data = read_json(payload["summary_path"])
        for selector in payload["selectors"]:
            phase_payload = data.get("phase_summaries", {}).get("t1_m", {}).get(selector)
            if phase_payload:
                merged[selector] = phase_payload
    return merged


def phase_table(summary: dict[str, dict[str, Any]], phase: str) -> str:
    writers = PHASES[phase]["writers"]
    dense_headers = [f"K{writer}" for writer in writers[1:]]
    headers = ["Selector", "Best", "Last", "Dense", *dense_headers]
    rows: list[list[str]] = []
    for selector in SELECTORS:
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


def legacy_table(summary: dict[str, dict[str, Any]]) -> str:
    headers = ["Selector", "Best", "Last", "Dense"]
    rows: list[list[str]] = []
    for selector in ("visitonly", "visit_taskgrad_half", "stageadaptive_late_half"):
        payload = summary.get(selector)
        if not payload:
            continue
        rows.append(
            [
                SELECTORS[selector]["label"],
                fmt_stat(payload["best_val"]),
                fmt_stat(payload["last_val"]),
                fmt_stat(payload["dense_mean"]),
            ]
        )
    return markdown_table(headers, rows)


def recommend(matched: dict[str, dict[str, dict[str, Any]]]) -> str:
    core = matched.get("core_m", {})
    t2a = matched.get("t2a_l", {})
    t1 = matched.get("t1_l", {})
    stage = "stageadaptive_late_half"
    vt = "visit_taskgrad_half"
    v = "visitonly"
    if stage not in t1 or stage not in t2a or stage not in core:
        return "Pending: missing matched runs."

    stage_transfer_dense = t1[stage]["dense_mean"]["mean"] + t2a[stage]["dense_mean"]["mean"]
    vt_transfer_dense = t1[vt]["dense_mean"]["mean"] + t2a[vt]["dense_mean"]["mean"]
    stage_last = t1[stage]["last_val"]["mean"] + t2a[stage]["last_val"]["mean"]
    vt_last = t1[vt]["last_val"]["mean"] + t2a[vt]["last_val"]["mean"]

    if stage_transfer_dense > vt_transfer_dense and stage_last <= vt_last and core[stage]["dense_mean"]["mean"] <= core[vt]["dense_mean"]["mean"]:
        return "StageLate-0.5 is best read as a transfer-specialist variant, not a global replacement for VT-0.5."
    if stage_transfer_dense > vt_transfer_dense and stage_last > vt_last:
        return "The answer is regime-dependent: StageLate-0.5 is stronger on matched transfer/stress, while VT-0.5 remains the simpler global baseline."
    if vt_transfer_dense >= stage_transfer_dense and core[vt]["dense_mean"]["mean"] >= core[stage]["dense_mean"]["mean"]:
        return "VT-0.5 remains the safest overall default; StageLate-0.5 does not clear a matched transfer/stress bar strongly enough to justify switching."
    if core[v]["dense_mean"]["mean"] >= core[vt]["dense_mean"]["mean"] and vt_transfer_dense > stage_transfer_dense:
        return "VT-0.5 remains the overall default, with V still a credible home-regime fallback."
    return "Matched results remain mixed: keep VT-0.5 as default unless a transfer-specialist policy is explicitly desired."


def main() -> None:
    REPORTS.mkdir(exist_ok=True)

    all_records: list[dict[str, Any]] = []
    matched_phase_summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for phase, spec in PHASES.items():
        records = phase_records(phase)
        all_records.extend(records)
        matched_phase_summaries[phase] = summarize_phase(records, spec["writers"])
        plot_phase(matched_phase_summaries[phase], phase, REPORTS / f"v28_{phase}.png")

    legacy_t1_m = load_legacy_t1_m()
    recommendation = recommend(matched_phase_summaries)

    summary = {
        "matched_seed_sets": {phase: spec["seeds"] for phase, spec in PHASES.items()},
        "matched_phase_summaries": matched_phase_summaries,
        "legacy_t1_m": legacy_t1_m,
        "recommendation": recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    sections: list[str] = []
    for phase in ("core_m", "t2a_l", "t1_l"):
        sections.append(f"## {PHASES[phase]['title']}\n\n{phase_table(matched_phase_summaries.get(phase, {}), phase)}")

    report = f"""# APSGNN v28 Regime-Dependent Selector Validation

This round answers the remaining v26-v27 question with matched seed sets: does the late-stage adaptive selector justify a regime-dependent rule, or does `VT-0.5` stay the safest global default?

The exact matched comparisons in this report are:
- `Core-M` on seeds `{", ".join(str(seed) for seed in PHASES['core_m']['seeds'])}`
- `T2a-L` on seeds `{", ".join(str(seed) for seed in PHASES['t2a_l']['seeds'])}`
- `T1-L` on seeds `{", ".join(str(seed) for seed in PHASES['t1_l']['seeds'])}`

Legacy context:
- `T1-M` remains included from the earlier v26/v27 reports, but it is not seed-matched across all three selectors.

## Legacy T1-M Context

{legacy_table(legacy_t1_m) if legacy_t1_m else "_Pending_"}

{"\n\n".join(sections) if sections else "_Pending_"}

## Recommendation

{recommendation}

## Completed matched runs

{run_table(all_records) if all_records else "_Pending_"}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
