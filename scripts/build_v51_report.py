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
SUMMARY_PATH = REPORTS / "summary_metrics_v51.json"
REPORT_PATH = REPORTS / "final_report_v51_regime_keyed_rule_confirmation.md"

SELECTORS = {
    "visit_taskgrad_half": {"label": "VT-0.5"},
    "stageadaptive_late_half": {"label": "StageLate-0.5"},
}

PHASES = {
    "core_xl": {"regime": "core", "train_steps": 4590, "writers": [2, 6, 10], "family": "home"},
    "t1_xl": {"regime": "t1", "train_steps": 4590, "writers": [4, 8, 12, 14], "family": "transfer"},
    "t1r_xl": {"regime": "t1r", "train_steps": 4590, "writers": [4, 8, 12, 14], "family": "transfer"},
    "t2a_xl": {"regime": "t2a", "train_steps": 4590, "writers": [4, 8, 12, 14], "family": "ingress"},
    "t2b_xl": {"regime": "t2b", "train_steps": 4590, "writers": [4, 8, 12, 14], "family": "non_ingress_stress"},
    "t2c_xl": {"regime": "t2c", "train_steps": 4590, "writers": [6, 10, 14, 16], "family": "non_ingress_stress"},
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


def phase_records(phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        prefix = f"v51-{spec['regime']}-{selector}-32-xl"
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


def family_scores(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    families: dict[str, dict[str, float]] = {}
    for phase, spec in PHASES.items():
        family = spec["family"]
        families.setdefault(family, {selector: 0.0 for selector in SELECTORS})
        for selector in SELECTORS:
            payload = phase_summaries.get(phase, {}).get(selector)
            if payload:
                families[family][selector] += payload["score"]["mean"]
    return families


def total_score(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {
        selector: sum(
            phase_summaries.get(phase, {}).get(selector, {}).get("score", {}).get("mean", 0.0)
            for phase in PHASES
        )
        for selector in SELECTORS
    }


def phase_winners(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for phase in PHASES:
        payload = {
            selector: phase_summaries.get(phase, {}).get(selector, {}).get("score", {}).get("mean", float("-inf"))
            for selector in SELECTORS
        }
        winners[phase] = max(payload, key=payload.get)
    return winners


def family_winners(family_score: dict[str, dict[str, float]]) -> dict[str, str]:
    return {family: max(scores, key=scores.get) for family, scores in family_score.items()}


def rule_recommendation(family_score: dict[str, dict[str, float]], final_recommendation: str) -> str:
    winners = family_winners(family_score)
    home = winners.get("home")
    transfer = winners.get("transfer")
    ingress = winners.get("ingress")
    non_ingress = winners.get("non_ingress_stress")
    if home == "visit_taskgrad_half" and ingress == "visit_taskgrad_half":
        if transfer == "stageadaptive_late_half" and non_ingress == "stageadaptive_late_half":
            return "regime_keyed"
    if transfer == ingress == non_ingress == home:
        return "single_selector"
    return "mixed"


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        records = phase_records(phase)
        if not records:
            continue
        all_records.extend(records)
        phase_summaries[phase] = summarize_phase(records, phase)

    final_score = total_score(phase_summaries)
    family_score = family_scores(phase_summaries)
    winners = phase_winners(phase_summaries)
    final_recommendation = max(final_score, key=final_score.get) if final_score else "visit_taskgrad_half"
    family_rule = rule_recommendation(family_score, final_recommendation)
    family_winner = family_winners(family_score)

    payload = {
        "phases": phase_summaries,
        "phase_summaries": phase_summaries,
        "records": all_records,
        "family_score": family_score,
        "phase_winners": winners,
        "family_winners": family_winner,
        "rule_recommendation": family_rule,
        "final_score": final_score,
        "final_recommendation": final_recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# APSGNN v51: Regime-Keyed Selector Rule Confirmation",
                "",
                "## What Changed",
                "",
                "v51 runs a fresh matched same-seed confirmation of the current two-arm selector rule across Core, T1, T1r, T2a, T2b, and T2c at 32 leaves.",
                "",
                "## Recommendation",
                "",
                f"Current overall winner on the fresh v51 matrix: {SELECTORS[final_recommendation]['label']}.",
                "",
                f"Current rule interpretation: `{family_rule}`.",
                "",
                f"- Summary JSON: [summary_metrics_v51.json]({SUMMARY_PATH})",
                f"- Report: [final_report_v51_regime_keyed_rule_confirmation.md]({REPORT_PATH})",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
