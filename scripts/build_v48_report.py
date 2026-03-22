#!/usr/bin/env python3
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v48.json"
REPORT_PATH = REPORTS / "final_report_v48_matched_stagelate_tiebreak.md"

SELECTORS = {
    "visit_taskgrad_half": {"label": "VT-0.5"},
    "stageadaptive_late_half": {"label": "StageLate-0.5"},
}

PHASES = {
    "t1_xl": {"regime": "t1", "train_steps": 4590, "writers": [4, 8, 12, 14]},
    "t1r_xl": {"regime": "t1r", "train_steps": 4590, "writers": [4, 8, 12, 14]},
    "t2a_xl": {"regime": "t2a", "train_steps": 4590, "writers": [4, 8, 12, 14]},
}

POOLED_PREFIXES = {
    "t1_xl": ["v46-t1", "v47-t1", "v48-t1"],
    "t1r_xl": ["v46-t1r", "v47-t1r", "v48-t1r"],
    "t2a_xl": ["v46-t2a", "v47-t2a", "v48-t2a"],
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


def latest_runs(prefix: str) -> list[tuple[Path, int]]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [(latest_by_seed[seed], seed) for seed in sorted(latest_by_seed)]


def is_complete_run(run_dir: Path, expected_train_steps: int) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists() or not (run_dir / "last.pt").exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != expected_train_steps:
        return False
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= expected_train_steps


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def summarize_run(run_dir: Path, selector: str, phase: str, seed: int) -> dict[str, Any]:
    spec = PHASES[phase]
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
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
    for writer in spec["writers"]:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in spec["writers"][1:]])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in spec["writers"][1:]])
    record["score"] = record["last_val"] + record["dense_mean"]
    return record


def phase_records_from_prefixes(phase: str, prefixes: list[str]) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for prefix_root in prefixes:
            prefix = f"{prefix_root}-{selector}-32-xl"
            for run_dir, seed in latest_runs(prefix):
                if not is_complete_run(run_dir, spec["train_steps"]):
                    continue
                records.append(summarize_run(run_dir, selector, phase, seed))
    return records


def summarize_phase(records: list[dict[str, Any]], phase: str) -> dict[str, dict[str, Any]]:
    writers = PHASES[phase]["writers"]
    out: dict[str, dict[str, Any]] = {}
    for selector in SELECTORS:
        selector_records = [record for record in records if record["selector"] == selector]
        if not selector_records:
            continue
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "dense_mean": mean_std([record["dense_mean"] for record in selector_records]),
            "last_dense_mean": mean_std([record["last_dense_mean"] for record in selector_records]),
            "score": mean_std([record["score"] for record in selector_records]),
        }
        for writer in writers[1:]:
            out[selector][f"k{writer}"] = mean_std([record[f"k{writer}"] for record in selector_records])
    return out


def total_score(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    final_score: dict[str, float] = {}
    for selector in SELECTORS:
        total = 0.0
        for phase in PHASES:
            payload = phase_summaries.get(phase, {}).get(selector)
            if payload:
                total += payload["score"]["mean"]
        final_score[selector] = total
    return final_score


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    current_phase_summaries: dict[str, dict[str, Any]] = {}
    pooled_phase_summaries: dict[str, dict[str, Any]] = {}
    current_records: list[dict[str, Any]] = []
    pooled_records: list[dict[str, Any]] = []

    for phase in PHASES:
        current = phase_records_from_prefixes(phase, [f"v48-{PHASES[phase]['regime']}"])
        if current:
            current_records.extend(current)
            current_phase_summaries[phase] = summarize_phase(current, phase)
        pooled = phase_records_from_prefixes(phase, POOLED_PREFIXES[phase])
        if pooled:
            pooled_records.extend(pooled)
            pooled_phase_summaries[phase] = summarize_phase(pooled, phase)

    final_score = total_score(current_phase_summaries)
    pooled_final_score = total_score(pooled_phase_summaries)
    final_recommendation = max(final_score, key=final_score.get) if final_score else "visit_taskgrad_half"
    pooled_final_recommendation = (
        max(pooled_final_score, key=pooled_final_score.get) if pooled_final_score else "visit_taskgrad_half"
    )

    payload = {
        "current_phase_summaries": current_phase_summaries,
        "pooled_phase_summaries": pooled_phase_summaries,
        "phase_summaries": current_phase_summaries,
        "pooled_records": pooled_records,
        "records": current_records,
        "final_score": final_score,
        "pooled_final_score": pooled_final_score,
        "final_recommendation": final_recommendation,
        "pooled_final_recommendation": pooled_final_recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# APSGNN v48: Matched StageLate Tie-Break",
                "",
                "## What Changed",
                "",
                "v48 runs one more fresh matched VT-0.5 vs StageLate-0.5 transfer-only comparison and pools v46, v47, and v48 under the same T1/T1r/T2a XL structure.",
                "",
                f"- Summary JSON: [summary_metrics_v48.json]({SUMMARY_PATH})",
                f"- Report: [final_report_v48_matched_stagelate_tiebreak.md]({REPORT_PATH})",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
