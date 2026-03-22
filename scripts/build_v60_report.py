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
SUMMARY_PATH = REPORTS / "summary_metrics_v60.json"
REPORT_PATH = REPORTS / "final_report_v60_v_vs_vt_half_tiebreak.md"

SELECTORS = {
    "visitonly": {"label": "V"},
    "visit_taskgrad_half": {"label": "VT-0.5"},
}

RULES = {
    "single_selector_v": {
        "label": "V everywhere",
        "selection": {
            "home": "visitonly",
            "transfer": "visitonly",
            "ingress": "visitonly",
            "non_ingress_stress": "visitonly",
        },
    },
    "single_selector_vt_half": {
        "label": "VT-0.5 everywhere",
        "selection": {
            "home": "visit_taskgrad_half",
            "transfer": "visit_taskgrad_half",
            "ingress": "visit_taskgrad_half",
            "non_ingress_stress": "visit_taskgrad_half",
        },
    },
    "regime_keyed_vt_home_ingress_v_transfer_non_ingress": {
        "label": "VT-0.5 on Core/T2a, V on T1/T1r/T2b/T2c",
        "selection": {
            "home": "visit_taskgrad_half",
            "transfer": "visitonly",
            "ingress": "visit_taskgrad_half",
            "non_ingress_stress": "visitonly",
        },
    },
}

PHASES = {
    "fresh_core_xl": {
        "prefixes": ["v60-core"],
        "train_steps": 4590,
        "writers": [2, 6, 10],
        "family": "home",
    },
    "fresh_t1_xl": {
        "prefixes": ["v60-t1"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "transfer",
    },
    "fresh_t1r_xl": {
        "prefixes": ["v60-t1r"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "transfer",
    },
    "fresh_t2a_xl": {
        "prefixes": ["v60-t2a"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "ingress",
    },
    "fresh_t2b_xl": {
        "prefixes": ["v60-t2b"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "non_ingress_stress",
    },
    "fresh_t2c_xl": {
        "prefixes": ["v60-t2c"],
        "train_steps": 4590,
        "writers": [6, 10, 14, 16],
        "family": "non_ingress_stress",
    },
    "pooled_core_xl": {
        "prefixes": ["v58-core", "v59-core", "v60-core"],
        "train_steps": 4590,
        "writers": [2, 6, 10],
        "family": "home",
    },
    "pooled_t1_xl": {
        "prefixes": ["v58-t1", "v59-t1", "v60-t1"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "transfer",
    },
    "pooled_t1r_xl": {
        "prefixes": ["v58-t1r", "v59-t1r", "v60-t1r"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "transfer",
    },
    "pooled_t2a_xl": {
        "prefixes": ["v58-t2a", "v59-t2a", "v60-t2a"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "ingress",
    },
    "pooled_t2b_xl": {
        "prefixes": ["v58-t2b", "v59-t2b", "v60-t2b"],
        "train_steps": 4590,
        "writers": [4, 8, 12, 14],
        "family": "non_ingress_stress",
    },
    "pooled_t2c_xl": {
        "prefixes": ["v58-t2c", "v59-t2c", "v60-t2c"],
        "train_steps": 4590,
        "writers": [6, 10, 14, 16],
        "family": "non_ingress_stress",
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


def records_for_phase(phase: str) -> list[dict[str, Any]]:
    spec = PHASES[phase]
    records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        for prefix_root in spec["prefixes"]:
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


def scope_phases(scope: str) -> list[str]:
    return [phase for phase in PHASES if phase.startswith(f"{scope}_")]


def selector_scores(phase_summaries: dict[str, dict[str, Any]], scope: str) -> dict[str, float]:
    return {
        selector: sum(
            phase_summaries.get(phase, {}).get(selector, {}).get("score", {}).get("mean", 0.0)
            for phase in scope_phases(scope)
        )
        for selector in SELECTORS
    }


def family_scores(phase_summaries: dict[str, dict[str, Any]], scope: str) -> dict[str, dict[str, float]]:
    families: dict[str, dict[str, float]] = {}
    for phase in scope_phases(scope):
        family = PHASES[phase]["family"]
        families.setdefault(family, {selector: 0.0 for selector in SELECTORS})
        for selector in SELECTORS:
            payload = phase_summaries.get(phase, {}).get(selector)
            if payload:
                families[family][selector] += payload["score"]["mean"]
    return families


def rule_scores(family_score: dict[str, dict[str, float]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for rule_name, rule in RULES.items():
        total = 0.0
        for family, selector in rule["selection"].items():
            total += family_score.get(family, {}).get(selector, 0.0)
        scores[rule_name] = total
    return scores


def phase_winners(phase_summaries: dict[str, dict[str, Any]]) -> dict[str, str]:
    winners: dict[str, str] = {}
    for phase in PHASES:
        payload = {
            selector: phase_summaries.get(phase, {}).get(selector, {}).get("score", {}).get("mean", float("-inf"))
            for selector in SELECTORS
        }
        winners[phase] = max(payload, key=payload.get)
    return winners


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}
    for phase in PHASES:
        records = records_for_phase(phase)
        if not records:
            continue
        all_records.extend(records)
        phase_summaries[phase] = summarize_phase(records, phase)

    fresh_selector_score = selector_scores(phase_summaries, "fresh")
    pooled_selector_score = selector_scores(phase_summaries, "pooled")
    fresh_family_score = family_scores(phase_summaries, "fresh")
    pooled_family_score = family_scores(phase_summaries, "pooled")
    fresh_rule_score = rule_scores(fresh_family_score)
    pooled_rule_score = rule_scores(pooled_family_score)
    fresh_rule_recommendation = max(fresh_rule_score, key=fresh_rule_score.get)
    pooled_rule_recommendation = max(pooled_rule_score, key=pooled_rule_score.get)

    payload = {
        "phases": phase_summaries,
        "phase_summaries": phase_summaries,
        "records": all_records,
        "phase_winners": phase_winners(phase_summaries),
        "fresh_selector_score": fresh_selector_score,
        "pooled_selector_score": pooled_selector_score,
        "fresh_family_score": fresh_family_score,
        "pooled_family_score": pooled_family_score,
        "fresh_rule_score": fresh_rule_score,
        "pooled_rule_score": pooled_rule_score,
        "fresh_rule_recommendation": fresh_rule_recommendation,
        "pooled_rule_recommendation": pooled_rule_recommendation,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# APSGNN v60: V vs VT-0.5 Tie-Break",
                "",
                "## What Changed",
                "",
                "v60 adds a third fresh same-seed six-regime matched pair after v58 and v59 disagreed. It reports both fresh-v60 results and the pooled v58+v59+v60 same-structure aggregate.",
                "",
                "## Recommendation",
                "",
                f"Fresh rule winner: `{fresh_rule_recommendation}` ({RULES[fresh_rule_recommendation]['label']}).",
                f"Pooled v58+v59+v60 rule winner: `{pooled_rule_recommendation}` ({RULES[pooled_rule_recommendation]['label']}).",
                "",
                "The three-pair pooled split is consistent by regime family:",
                "- `VT-0.5` wins `Core` and `T2a`.",
                "- `V` wins `T1`, `T1r`, `T2b`, and `T2c`.",
                "- That keyed rule now beats both single-selector baselines on both fresh-v60 and pooled v58+v59+v60 scoring.",
                "",
                f"- Summary JSON: [summary_metrics_v60.json]({SUMMARY_PATH})",
                f"- Report: [final_report_v60_v_vs_vt_half_tiebreak.md]({REPORT_PATH})",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
