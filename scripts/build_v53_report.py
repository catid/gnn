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
SUMMARY_PATH = REPORTS / "summary_metrics_v53.json"
REPORT_PATH = REPORTS / "final_report_v53_t2c_writer_density_tiebreak.md"

SELECTORS = {
    "visit_taskgrad_half": {"label": "VT-0.5"},
    "stageadaptive_late_half": {"label": "StageLate-0.5"},
}

PHASE = {"regime": "t2c", "train_steps": 4590, "writers": [6, 10, 14, 16]}
POOLED_PREFIXES = ["v50-t2c", "v51-t2c", "v52-t2c", "v53-t2c"]


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


def summarize_run(run_dir: Path, selector: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    record: dict[str, Any] = {
        "selector": selector,
        "selector_label": SELECTORS[selector]["label"],
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
    }
    for writer in PHASE["writers"]:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in PHASE["writers"][1:]])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in PHASE["writers"][1:]])
    record["score"] = record["last_val"] + record["dense_mean"]
    return record


def records_from_prefixes(prefix_roots: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for prefix_root in prefix_roots:
            prefix = f"{prefix_root}-{selector}-32-xl"
            for run_dir, seed in latest_runs(prefix):
                if not is_complete_run(run_dir, PHASE["train_steps"]):
                    continue
                records.append(summarize_run(run_dir, selector, seed))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
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
        for writer in PHASE["writers"][1:]:
            out[selector][f"k{writer}"] = mean_std([record[f"k{writer}"] for record in selector_records])
    return out


def total_score(summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    return {selector: summary.get(selector, {}).get("score", {}).get("mean", 0.0) for selector in SELECTORS}


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    current_records = records_from_prefixes(["v53-t2c"])
    pooled_records = records_from_prefixes(POOLED_PREFIXES)
    current_summary = summarize(current_records)
    pooled_summary = summarize(pooled_records)
    final_score = total_score(current_summary)
    pooled_final_score = total_score(pooled_summary)
    final_recommendation = max(final_score, key=final_score.get) if final_score else "visit_taskgrad_half"
    pooled_final_recommendation = max(pooled_final_score, key=pooled_final_score.get) if pooled_final_score else "visit_taskgrad_half"

    payload = {
        "phase_summary": current_summary,
        "pooled_phase_summary": pooled_summary,
        "records": current_records,
        "pooled_records": pooled_records,
        "final_score": final_score,
        "pooled_final_score": pooled_final_score,
        "final_recommendation": final_recommendation,
        "pooled_final_recommendation": pooled_final_recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# APSGNN v53: T2c Writer-Density Tie-Break",
                "",
                "## What Changed",
                "",
                "v53 runs one more fresh matched same-seed T2c XL comparison between VT-0.5 and StageLate-0.5 and pools it with v50-v52 to decide the writer-density boundary.",
                "",
                f"- Summary JSON: [summary_metrics_v53.json]({SUMMARY_PATH})",
                f"- Report: [final_report_v53_t2c_writer_density_tiebreak.md]({REPORT_PATH})",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
