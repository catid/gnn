#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v64.json"
REPORT_PATH = REPORTS / "final_report_v64_ds_contract_factorization.md"

BASE_LR = 2.0e-4
ROLLING_N = 5
EXPECTED_TRAIN_STEPS = {"p": 420, "s": 1134, "m": 2268, "l": 3024}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t1r": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
    "t2b": [4, 8, 12, 14],
    "t2c": [6, 10, 14, 16],
    "hmid": [3, 7, 11],
    "hmix": [3, 7, 11],
}

ANCHOR_REGIMES = ["core", "t1", "t2a"]
HMIX_TIEBREAK_REGIMES = ["hmix"]
CONFIRMATION_REGIMES = ["core", "t1", "t2a", "hmix"]
HOLDOUT_REGIMES = ["t1r", "t2b", "t2c", "hmid"]
EXTRA_DEPTH_REGIMES = ["core", "t1", "t2a", "hmix"]
FINAL_RERUN_CANDIDATE_REGIMES = ["core", "t1", "t2a", "hmix"]
RAW_CORE_CONTRACTS = [
    "d",
    "ds_p005",
    "ds_p010",
    "ds_p025",
    "ds_p040",
    "ds_fixed1step",
    "ds_fixed2step",
]
MAIN_CONTRACTS = ["d", "ds_core_best", "ds_core_runner_up", "ds_auxanneal", "ds_randdepth"]

SELECTOR_PREFIXES = {
    "visitonly": "V",
    "visit_taskgrad_half": "VT-0.5",
}

CONTRACT_LABELS = {
    "d": "D",
    "ds_p005": "DS-p0.05",
    "ds_p010": "DS-p0.10",
    "ds_p025": "DS-p0.25",
    "ds_p040": "DS-p0.40",
    "ds_fixed1step": "DS-fixed1step",
    "ds_fixed2step": "DS-fixed2step",
    "ds_core_best": "DS-core-best",
    "ds_core_runner_up": "DS-core-runner-up",
    "ds_auxanneal_050": "DS+AuxAnneal(0.50)",
    "ds_auxanneal_025": "DS+AuxAnneal(0.25)",
    "ds_auxanneal": "DS+AuxAnneal",
    "ds_randdepth": "DS+RandDepth",
}

RUN_RE = re.compile(
    r"v64-(?P<regime>[^-]+)-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": std(values)}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def visible_gpu_count() -> int:
    result = subprocess.run(
        ["bash", "-lc", "nvidia-smi --list-gpus | wc -l"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return max(int(result.stdout.strip()), 0)
    except ValueError:
        return 0


def parse_pair(pair_key: str) -> dict[str, str]:
    if pair_key.startswith("visit_taskgrad_half_"):
        contract = pair_key[len("visit_taskgrad_half_") :]
        return {
            "selector": "visit_taskgrad_half",
            "selector_label": "VT-0.5",
            "contract": contract,
            "contract_label": CONTRACT_LABELS.get(contract, contract),
            "pair_label": f"VT-0.5/{CONTRACT_LABELS.get(contract, contract)}",
        }
    if pair_key.startswith("visitonly_"):
        contract = pair_key[len("visitonly_") :]
        return {
            "selector": "visitonly",
            "selector_label": "V",
            "contract": contract,
            "contract_label": CONTRACT_LABELS.get(contract, contract),
            "pair_label": f"V/{CONTRACT_LABELS.get(contract, contract)}",
        }
    raise KeyError(f"Unrecognized v64 pair key: {pair_key}")


def parse_notes(notes: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for chunk in str(notes).split(";"):
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def latest_runs() -> list[tuple[dict[str, str], Path]]:
    pilot_runs: list[tuple[dict[str, str], Path]] = []
    latest: dict[tuple[str, str, str, str, str], tuple[dict[str, str], Path]] = {}
    for candidate in sorted(RUNS.glob("*-v64-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["schedule"] == "p":
            pilot_runs.append((meta, candidate))
            continue
        latest[(meta["regime"], meta["pair"], meta["schedule"], meta["tag"], meta["seed"])] = (meta, candidate)
    return pilot_runs + list(latest.values())


def is_complete_run(run_dir: Path, schedule: str) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    last_path = run_dir / "last.pt"
    if not metrics_path.exists() or not config_path.exists() or not last_path.exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != EXPECTED_TRAIN_STEPS[schedule]:
        return False
    max_step = 0
    for row in read_jsonl(metrics_path):
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= EXPECTED_TRAIN_STEPS[schedule]


def extract_eval(run_dir: Path, kind: str, writers: int, *, depth_tag: str | None = None) -> float:
    suffix = f"_{depth_tag}" if depth_tag else ""
    path = run_dir / f"eval_{kind}{suffix}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def extract_eval_metrics(run_dir: Path, kind: str, writers: int, *, depth_tag: str | None = None) -> dict[str, float]:
    suffix = f"_{depth_tag}" if depth_tag else ""
    path = run_dir / f"eval_{kind}{suffix}_k{writers}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def _history_value(history: list[dict[str, Any]], step: int, key: str) -> float:
    for row in history:
        if int(row.get("step", 0)) >= step:
            return float(row.get(key, 0.0))
    if history:
        return float(history[-1].get(key, 0.0))
    return 0.0


def _time_to_threshold(history: list[dict[str, Any]], key: str, threshold: float) -> float:
    bootstrap_end = None
    for row in history:
        if not bool(row.get("bootstrap_active", False)):
            bootstrap_end = int(row.get("step", 0))
            break
    if bootstrap_end is None:
        return 0.0
    for row in history:
        step = int(row.get("step", 0))
        if step < bootstrap_end:
            continue
        if float(row.get(key, 0.0)) >= threshold:
            return float(step - bootstrap_end)
    return 0.0


def extract_coverage_and_split_metrics(run_dir: Path) -> dict[str, float]:
    path = run_dir / "coverage_summary.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    history = payload.get("history", [])
    stages = payload.get("stages", [])
    out = {
        "task_visit_cov_10": _history_value(history, 10, "task_visit_coverage"),
        "task_visit_cov_50": _history_value(history, 50, "task_visit_coverage"),
        "task_visit_cov_100": _history_value(history, 100, "task_visit_coverage"),
        "task_visit_cov_200": _history_value(history, 200, "task_visit_coverage"),
        "task_grad_cov_10": _history_value(history, 10, "task_gradient_coverage"),
        "task_grad_cov_50": _history_value(history, 50, "task_gradient_coverage"),
        "task_grad_cov_100": _history_value(history, 100, "task_gradient_coverage"),
        "task_grad_cov_200": _history_value(history, 200, "task_gradient_coverage"),
        "time_to_visit_50": _time_to_threshold(history, "task_visit_coverage", 0.50),
        "time_to_visit_75": _time_to_threshold(history, "task_visit_coverage", 0.75),
        "time_to_visit_100": _time_to_threshold(history, "task_visit_coverage", 1.00),
        "time_to_grad_50": _time_to_threshold(history, "task_gradient_coverage", 0.50),
        "time_to_grad_75": _time_to_threshold(history, "task_gradient_coverage", 0.75),
        "time_to_grad_100": _time_to_threshold(history, "task_gradient_coverage", 1.00),
    }
    split_blocks = [stage.get("split_stats", {}) for stage in stages if stage.get("split_stats")]
    if split_blocks:
        def split_scalar(value: Any) -> float:
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, dict):
                return mean([float(item) for item in value.values()]) if value else 0.0
            return 0.0

        for key in (
            "selected_parent_child_usefulness_mean",
            "unselected_parent_child_usefulness_mean",
            "child_visit_share",
            "child_grad_share",
            "utility_usefulness_correlation",
        ):
            values = [split_scalar(block.get(key, 0.0)) for block in split_blocks if key in block]
            out[key] = mean(values)
    return out


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(ROLLING_N, len(vals)) :]
    pair_info = parse_pair(meta["pair"])
    writers = REGIME_WRITERS[meta["regime"]]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    notes = parse_notes(config.get("runtime", {}).get("notes", ""))
    record: dict[str, Any] = {
        "run": run_dir.name,
        "pair": meta["pair"],
        "pair_label": pair_info["pair_label"],
        "selector": pair_info["selector"],
        "selector_label": pair_info["selector_label"],
        "contract": pair_info["contract"],
        "contract_label": pair_info["contract_label"],
        "regime": meta["regime"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"]),
        "tag": meta["tag"],
        "config_name": f"configs/v64_{meta['regime']}_{meta['pair']}_32_{meta['schedule']}.yaml",
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "contract_kind": str(config["train"].get("contract_kind", "baseline")),
        "p_keep_prev": float(config["train"].get("contract_penultimate_keep_prob", 0.0)),
        "aux_anneal_final_multiplier": float(config["train"].get("contract_aux_anneal_final_multiplier", 1.0)),
        "aux_anneal_start_fraction": float(config["train"].get("contract_aux_anneal_start_fraction", 1.0)),
        "rand_depth_train_fraction": float(config["train"].get("contract_rand_depth_train_fraction", 0.0)),
        "rand_depth_multipliers": list(config["train"].get("contract_rand_depth_multipliers", []) or []),
        "routing_aux_multiplier": float(last.get("train/routing_aux_multiplier", 0.0)),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
        "source_contract": notes.get("source", pair_info["contract"]),
        "core_best_source": notes.get("core_best", ""),
        "core_runner_up_source": notes.get("core_runner_up", ""),
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in dense_writers])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in dense_writers])
    record["composite"] = (
        0.40 * record["dense_mean"]
        + 0.30 * record["last_val"]
        + 0.15 * record["last5_val_mean"]
        - 0.15 * record["best_to_last_drop"]
    )
    record.update(extract_coverage_and_split_metrics(run_dir))
    return record


def summarize_group(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    out = {
        "count": len(records),
        "best_val": mean_std([record["best_val"] for record in records]),
        "last_val": mean_std([record["last_val"] for record in records]),
        "last5_val_mean": mean_std([record["last5_val_mean"] for record in records]),
        "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in records]),
        "dense_mean": mean_std([record["dense_mean"] for record in records]),
        "last_dense_mean": mean_std([record["last_dense_mean"] for record in records]),
        "composite": mean_std([record["composite"] for record in records]),
        "lr_multiplier": mean_std([record["lr_multiplier"] for record in records]),
        "p_keep_prev": mean_std([record["p_keep_prev"] for record in records]),
        "aux_anneal_final_multiplier": mean_std([record["aux_anneal_final_multiplier"] for record in records]),
        "rand_depth_train_fraction": mean_std([record["rand_depth_train_fraction"] for record in records]),
        "query_first_hop_home_rate": mean_std([record.get("query_first_hop_home_rate", 0.0) for record in records]),
        "delivery_rate": mean_std([record.get("delivery_rate", 0.0) for record in records]),
        "home_to_out_rate": mean_std([record.get("home_to_out_rate", 0.0) for record in records]),
        "task_visit_cov_10": mean_std([record.get("task_visit_cov_10", 0.0) for record in records]),
        "task_visit_cov_50": mean_std([record.get("task_visit_cov_50", 0.0) for record in records]),
        "task_visit_cov_100": mean_std([record.get("task_visit_cov_100", 0.0) for record in records]),
        "task_visit_cov_200": mean_std([record.get("task_visit_cov_200", 0.0) for record in records]),
        "task_grad_cov_10": mean_std([record.get("task_grad_cov_10", 0.0) for record in records]),
        "task_grad_cov_50": mean_std([record.get("task_grad_cov_50", 0.0) for record in records]),
        "task_grad_cov_100": mean_std([record.get("task_grad_cov_100", 0.0) for record in records]),
        "task_grad_cov_200": mean_std([record.get("task_grad_cov_200", 0.0) for record in records]),
        "time_to_visit_50": mean_std([record.get("time_to_visit_50", 0.0) for record in records]),
        "time_to_visit_75": mean_std([record.get("time_to_visit_75", 0.0) for record in records]),
        "time_to_visit_100": mean_std([record.get("time_to_visit_100", 0.0) for record in records]),
        "time_to_grad_50": mean_std([record.get("time_to_grad_50", 0.0) for record in records]),
        "time_to_grad_75": mean_std([record.get("time_to_grad_75", 0.0) for record in records]),
        "time_to_grad_100": mean_std([record.get("time_to_grad_100", 0.0) for record in records]),
        "selected_parent_child_usefulness_mean": mean_std(
            [record.get("selected_parent_child_usefulness_mean", 0.0) for record in records]
        ),
        "unselected_parent_child_usefulness_mean": mean_std(
            [record.get("unselected_parent_child_usefulness_mean", 0.0) for record in records]
        ),
        "child_visit_share": mean_std([record.get("child_visit_share", 0.0) for record in records]),
        "child_grad_share": mean_std([record.get("child_grad_share", 0.0) for record in records]),
        "utility_usefulness_correlation": mean_std(
            [record.get("utility_usefulness_correlation", 0.0) for record in records]
        ),
        "runs": [record["run"] for record in records],
    }
    all_writers = sorted({writer for record in records for writer in REGIME_WRITERS[record["regime"]]})
    for writer in all_writers:
        if all(f"k{writer}" in record for record in records):
            out[f"k{writer}"] = mean_std([record[f"k{writer}"] for record in records])
        if all(f"last_k{writer}" in record for record in records):
            out[f"last_k{writer}"] = mean_std([record[f"last_k{writer}"] for record in records])
    return out


def pilot_candidate_key(record: dict[str, Any], *, logical_pair: str | None = None) -> str:
    key = logical_pair or record["pair"]
    parts = [f"lr{record['lr_multiplier']:.1f}"]
    parts.append(f"p{record['p_keep_prev']:.2f}")
    if record["aux_anneal_final_multiplier"] < 0.999999:
        parts.append(f"aux{record['aux_anneal_final_multiplier']:.2f}")
    if record["rand_depth_train_fraction"] > 0.0:
        parts.append(f"rd{record['rand_depth_train_fraction']:.2f}")
    return f"{key}@{'@'.join(parts)}"


def logical_pair_key(pair: str) -> str:
    if pair.endswith("_ds_auxanneal_050"):
        return pair[: -len("_050")]
    if pair.endswith("_ds_auxanneal_025"):
        return pair[: -len("_025")]
    return pair


def choose_aux_anneal_final_multiplier(pilot_records: list[dict[str, Any]]) -> float:
    aux_records = [record for record in pilot_records if record["contract"] in {"ds_auxanneal_050", "ds_auxanneal_025"}]
    if not aux_records:
        return 0.50
    grouped: dict[float, list[dict[str, Any]]] = {}
    for record in aux_records:
        grouped.setdefault(record["aux_anneal_final_multiplier"], []).append(record)
    rankings = [
        {
            "aux_anneal_final_multiplier": multiplier,
            "summary": {"composite": mean_std([float(record.get("composite", 0.0)) for record in records])},
        }
        for multiplier, records in grouped.items()
    ]
    rankings.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    return float(rankings[0]["aux_anneal_final_multiplier"])


def rank_pairs(records: list[dict[str, Any]], allowed_pairs: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(allowed_pairs) if allowed_pairs is not None else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if allowed is not None and record["pair"] not in allowed:
            continue
        grouped.setdefault(record["pair"], []).append(record)
    rows: list[dict[str, Any]] = []
    for pair, pair_records in grouped.items():
        info = parse_pair(pair)
        rows.append(
            {
                "pair": pair,
                "pair_label": info["pair_label"],
                "selector": info["selector"],
                "selector_label": info["selector_label"],
                "contract": info["contract"],
                "contract_label": info["contract_label"],
                "summary": summarize_group(pair_records),
            }
        )
    rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    return rows


def rank_contracts(records: list[dict[str, Any]], allowed_contracts: list[str] | None = None) -> list[dict[str, Any]]:
    allowed = set(allowed_contracts) if allowed_contracts is not None else None
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        contract = record["contract"]
        if allowed is not None and contract not in allowed:
            continue
        grouped.setdefault(contract, []).append(record)
    rows: list[dict[str, Any]] = []
    for contract, contract_records in grouped.items():
        pair_summaries = rank_pairs(contract_records)
        selector_scores = [row["summary"]["composite"]["mean"] for row in pair_summaries]
        selector_score_stds = [row["summary"]["composite"]["std"] for row in pair_summaries]
        rows.append(
            {
                "contract": contract,
                "contract_label": CONTRACT_LABELS.get(contract, contract),
                "selector_pairs": [row["pair"] for row in pair_summaries],
                "selector_pair_labels": [row["pair_label"] for row in pair_summaries],
                "summary": {
                    "composite": mean_std(selector_scores),
                    "selector_composite_std": mean_std(selector_score_stds),
                },
            }
        )
    rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    return rows


def summarize_by_regime_and_pair(records: list[dict[str, Any]], allowed_pairs: list[str] | None = None) -> dict[str, dict[str, Any]]:
    allowed = set(allowed_pairs) if allowed_pairs is not None else None
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted({record["regime"] for record in records}):
        regime_records = [
            record for record in records if record["regime"] == regime and (allowed is None or record["pair"] in allowed)
        ]
        if not regime_records:
            continue
        out[regime] = {}
        for pair in sorted({record["pair"] for record in regime_records}):
            out[regime][pair] = summarize_group([record for record in regime_records if record["pair"] == pair])
    return out


def pair_total(summary: dict[str, dict[str, Any]], pair: str, regimes: list[str]) -> float:
    return sum(summary.get(regime, {}).get(pair, {}).get("composite", {}).get("mean", 0.0) for regime in regimes)


def select_core_contracts(pilot_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str | None, str | None]:
    candidates = [
        record
        for record in pilot_records
        if record["selector"] == "visitonly"
        and record["contract"] in RAW_CORE_CONTRACTS
        and record["regime"] in {"core", "t1"}
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        grouped.setdefault(pilot_candidate_key(record), []).append(record)
    rows: list[dict[str, Any]] = []
    for key, key_records in grouped.items():
        rows.append(
            {
                "candidate": key,
                "contract": key_records[0]["contract"],
                "contract_label": CONTRACT_LABELS.get(key_records[0]["contract"], key_records[0]["contract"]),
                "lr_multiplier": key_records[0]["lr_multiplier"],
                "p_keep_prev": key_records[0]["p_keep_prev"],
                "summary": summarize_group(key_records),
            }
        )
    rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
    best = rows[0]["contract"] if rows else None
    runner_up = rows[1]["contract"] if len(rows) > 1 else None
    return rows, best, runner_up


def select_main_pair_settings(
    pilot_records: list[dict[str, Any]],
    *,
    chosen_aux_anneal_final_multiplier: float,
) -> dict[str, dict[str, Any]]:
    candidates = [
        record
        for record in pilot_records
        if record["contract"] in {
            "d",
            "ds_core_best",
            "ds_core_runner_up",
            "ds_auxanneal_050",
            "ds_auxanneal_025",
            "ds_randdepth",
        }
        and record["regime"] in {"core", "t1"}
    ]
    candidates = [
        record
        for record in candidates
        if record["contract"] not in {"ds_auxanneal_050", "ds_auxanneal_025"}
        or abs(record["aux_anneal_final_multiplier"] - chosen_aux_anneal_final_multiplier) <= 1.0e-9
    ]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in candidates:
        logical_pair = logical_pair_key(record["pair"])
        grouped.setdefault(pilot_candidate_key(record, logical_pair=logical_pair), []).append(record)
    per_pair: dict[str, list[dict[str, Any]]] = {}
    for key, key_records in grouped.items():
        logical_pair = logical_pair_key(key_records[0]["pair"])
        row = {
            "candidate": key,
            "pair": logical_pair,
            "source_pair": key_records[0]["pair"],
            "pair_label": parse_pair(logical_pair)["pair_label"],
            "summary": summarize_group(key_records),
            "lr_multiplier": key_records[0]["lr_multiplier"],
            "p_keep_prev": key_records[0]["p_keep_prev"],
            "aux_anneal_final_multiplier": key_records[0]["aux_anneal_final_multiplier"],
        }
        per_pair.setdefault(logical_pair, []).append(row)
    selected: dict[str, dict[str, Any]] = {}
    for pair, rows in per_pair.items():
        rows.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
        selected[pair] = rows[0]
    return selected


def promote_contracts_after_screening(rankings: list[dict[str, Any]]) -> list[str]:
    return [row["contract"] for row in rankings[:3]]


def choose_top2_contracts_after_hmix(
    contract_rankings: list[dict[str, Any]],
    hmix_summary: dict[str, dict[str, Any]],
    candidate_contracts: list[str],
) -> list[str]:
    if not candidate_contracts:
        return []
    rank_map = {row["contract"]: row for row in contract_rankings}
    scores: dict[str, float] = {}
    for contract in candidate_contracts:
        score = rank_map.get(contract, {}).get("summary", {}).get("composite", {}).get("mean", 0.0)
        score += hmix_summary.get("hmix", {}).get(contract, {}).get("composite", {}).get("mean", 0.0)
        scores[contract] = score
    return sorted(candidate_contracts, key=lambda contract: scores.get(contract, 0.0), reverse=True)[:2]


def choose_top2_pairs(
    pair_rankings: list[dict[str, Any]],
    holdout_summary: dict[str, dict[str, Any]] | None = None,
    allowed_pairs: list[str] | None = None,
) -> list[str]:
    allowed = set(allowed_pairs) if allowed_pairs is not None else None
    rows = [row for row in pair_rankings if allowed is None or row["pair"] in allowed]
    if holdout_summary:
        def score(row: dict[str, Any]) -> float:
            return row["summary"]["composite"]["mean"] + pair_total(holdout_summary, row["pair"], HOLDOUT_REGIMES)
        rows.sort(key=score, reverse=True)
    return [row["pair"] for row in rows[:2]]


def choose_final_rerun_regimes(top2_pairs: list[str], confirmation_summary: dict[str, dict[str, Any]]) -> list[str]:
    if len(top2_pairs) < 2:
        return []
    margins: list[tuple[float, int, str]] = []
    for index, regime in enumerate(FINAL_RERUN_CANDIDATE_REGIMES):
        a = confirmation_summary.get(regime, {}).get(top2_pairs[0], {}).get("composite", {}).get("mean", 0.0)
        b = confirmation_summary.get(regime, {}).get(top2_pairs[1], {}).get("composite", {}).get("mean", 0.0)
        margins.append((abs(a - b), index, regime))
    margins.sort()
    return [item[2] for item in margins[:2]]


def summarize_hmix_by_contract(records: list[dict[str, Any]], contracts: list[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {"hmix": {}}
    for contract in contracts:
        contract_records = [record for record in records if record["contract"] == contract and record["regime"] == "hmix"]
        out["hmix"][contract] = summarize_group(contract_records)
    return out


def summarize_extra_depth_for_pair(pair: str, regime: str, schedule: str = "m") -> dict[str, Any]:
    latest: Path | None = None
    latest_seed = -1
    for candidate in sorted(RUNS.glob(f"*-v64-{regime}-{pair}-32-{schedule}-s*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None or meta["tag"]:
            continue
        seed = int(meta["seed"])
        if seed >= latest_seed:
            latest = candidate
            latest_seed = seed
    if latest is None:
        return {}
    writers = REGIME_WRITERS[regime]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    out: dict[str, Any] = {"run": latest.name, "depths": {}}
    for depth_tag in ("standard", "plus50", "plus100", "settle"):
        best_dense = mean([extract_eval(latest, "best", writer, depth_tag=depth_tag) for writer in dense_writers])
        last_dense = mean([extract_eval(latest, "last", writer, depth_tag=depth_tag) for writer in dense_writers])
        settle_metrics = extract_eval_metrics(latest, "best", dense_writers[-1], depth_tag=depth_tag)
        out["depths"][depth_tag] = {
            "best_dense_mean": best_dense,
            "last_dense_mean": last_dense,
            "settle_rate": float(settle_metrics.get("settle_rate", 0.0)),
            "steps_to_settle": float(settle_metrics.get("steps_to_settle", 0.0)),
            "settled_accuracy": float(settle_metrics.get("settled_accuracy", 0.0)),
            "accept_on_settle_accuracy": float(settle_metrics.get("accept_on_settle_accuracy", 0.0)),
            "non_settling_rate": float(settle_metrics.get("non_settling_rate", 0.0)),
        }
    return out


def determine_outcome(
    *,
    top2_pairs: list[str],
    confirmation_summary: dict[str, dict[str, Any]],
    holdout_summary: dict[str, dict[str, Any]],
    rerun_summary: dict[str, dict[str, Any]],
    ambiguity_summary: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if len(top2_pairs) < 2:
        return {"outcome": "unresolved", "winner": "unresolved", "reason": "fewer_than_two_finalists"}
    final_totals = {}
    for pair in top2_pairs:
        final_totals[pair] = (
            pair_total(confirmation_summary, pair, CONFIRMATION_REGIMES)
            + pair_total(holdout_summary, pair, HOLDOUT_REGIMES)
            + pair_total(rerun_summary, pair, list(rerun_summary))
            + pair_total(ambiguity_summary, pair, list(ambiguity_summary))
        )
    winner, runner_up = sorted(top2_pairs, key=lambda pair: final_totals[pair], reverse=True)
    margin = final_totals[winner] - final_totals[runner_up]
    rerun_margin = pair_total(rerun_summary, winner, list(rerun_summary)) - pair_total(
        rerun_summary, runner_up, list(rerun_summary)
    )
    if abs(margin) <= 1.0e-9 or abs(rerun_margin) <= 1.0e-9:
        return {
            "outcome": "unresolved",
            "winner": "unresolved",
            "reason": "final_totals_or_reruns_tied",
            "winner_pair": winner,
            "runner_up_pair": runner_up,
            "margin": margin,
        }
    winner_info = parse_pair(winner)
    runner_info = parse_pair(runner_up)
    if winner_info["contract"] == runner_info["contract"]:
        outcome = "universal_selector_under_contract"
    else:
        outcome = "universal_pair_under_specific_contract"
    return {
        "outcome": outcome,
        "winner": winner,
        "winner_label": winner_info["pair_label"],
        "runner_up": runner_up,
        "runner_up_label": runner_info["pair_label"],
        "margin": margin,
        "totals": final_totals,
    }


def render_table(headers: list[str], rows: list[list[object]]) -> str:
    if not rows:
        return ""
    header_line = "| " + " | ".join(headers) + " |"
    sep_line = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join([header_line, sep_line, *body])


def all_unique_config_names(records: list[dict[str, Any]]) -> list[str]:
    return sorted({record["config_name"] for record in records})


def main() -> None:
    latest = latest_runs()
    completed_records: list[dict[str, Any]] = []
    for meta, run_dir in latest:
        if not is_complete_run(run_dir, meta["schedule"]):
            continue
        completed_records.append(summarize_run(run_dir, meta))

    pilot_records = [record for record in completed_records if record["schedule"] == "p"]
    screen_records = [record for record in completed_records if record["schedule"] == "s" and record["tag"] == ""]
    confirm_records = [record for record in completed_records if record["schedule"] == "m" and record["tag"] == ""]
    holdout_records = [record for record in completed_records if record["schedule"] == "l" and record["tag"] == ""]
    rerun_records = [record for record in completed_records if record["schedule"] == "l" and record["tag"] == "rerun"]
    ambiguity_records = [record for record in completed_records if record["schedule"] == "l" and record["tag"] == "amb"]

    core_rankings, core_best, core_runner_up = select_core_contracts(pilot_records)
    chosen_aux_anneal_final_multiplier = choose_aux_anneal_final_multiplier(pilot_records)
    selected_settings = select_main_pair_settings(
        pilot_records,
        chosen_aux_anneal_final_multiplier=chosen_aux_anneal_final_multiplier,
    )

    screening_pairs = [
        record
        for record in screen_records
        if record["regime"] in ANCHOR_REGIMES and record["contract"] in MAIN_CONTRACTS
    ]
    screening_rankings = rank_pairs(screening_pairs)
    contract_screen_rankings = rank_contracts(screening_pairs, MAIN_CONTRACTS)
    promoted_contracts_screening = promote_contracts_after_screening(contract_screen_rankings)

    hmix_records = [
        record
        for record in screen_records
        if record["regime"] == "hmix" and record["contract"] in promoted_contracts_screening
    ]
    hmix_summary = summarize_hmix_by_contract(hmix_records, promoted_contracts_screening)
    top2_contracts_after_hmix = choose_top2_contracts_after_hmix(
        contract_screen_rankings,
        hmix_summary,
        promoted_contracts_screening,
    )

    confirmation_pairs = [
        record
        for record in confirm_records
        if record["regime"] in CONFIRMATION_REGIMES and record["contract"] in top2_contracts_after_hmix
    ]
    confirmation_rankings = rank_pairs(confirmation_pairs)
    confirmation_summary = summarize_by_regime_and_pair(confirmation_pairs)
    top2_pairs_after_confirmation = choose_top2_pairs(confirmation_rankings)

    holdout_pairs = [
        record for record in holdout_records if record["regime"] in HOLDOUT_REGIMES and record["pair"] in top2_pairs_after_confirmation
    ]
    holdout_summary = summarize_by_regime_and_pair(holdout_pairs)
    top2_pairs_after_holdout = choose_top2_pairs(
        confirmation_rankings,
        holdout_summary=holdout_summary,
        allowed_pairs=top2_pairs_after_confirmation,
    )

    rerun_regimes = choose_final_rerun_regimes(top2_pairs_after_holdout, confirmation_summary)
    rerun_summary = summarize_by_regime_and_pair(rerun_records, top2_pairs_after_holdout)
    ambiguity_summary = summarize_by_regime_and_pair(ambiguity_records, top2_pairs_after_holdout)
    extra_depth_summary = {
        pair: {
            regime: summarize_extra_depth_for_pair(pair, regime)
            for regime in EXTRA_DEPTH_REGIMES
        }
        for pair in top2_pairs_after_holdout
    }
    outcome = determine_outcome(
        top2_pairs=top2_pairs_after_holdout,
        confirmation_summary=confirmation_summary,
        holdout_summary=holdout_summary,
        rerun_summary=rerun_summary,
        ambiguity_summary=ambiguity_summary,
    )

    screening_contract_rows = [
        {
            "contract": row["contract"],
            "contract_label": row["contract_label"],
            "summary": {
                "composite": row["summary"]["composite"],
            },
        }
        for row in contract_screen_rankings
    ]

    summary = {
        "budgets": {"p": 420, "s": 1134, "m": 2268, "l": 3024},
        "visible_gpu_count": visible_gpu_count(),
        "selected_core_contracts": {
            "best": core_best,
            "runner_up": core_runner_up,
            "rankings": core_rankings,
        },
        "pilot_choices": selected_settings,
        "selected_settings": selected_settings,
        "chosen_aux_anneal_final_multiplier": chosen_aux_anneal_final_multiplier,
        "screening_rankings": screening_rankings,
        "screening_contract_rankings": screening_contract_rows,
        "promoted_contracts_screening": promoted_contracts_screening,
        "hmix_contract_tiebreak": hmix_summary,
        "top2_contracts_after_hmix": top2_contracts_after_hmix,
        "confirmation_rankings": confirmation_rankings,
        "confirmation_summary": confirmation_summary,
        "top2_pairs_after_confirmation": top2_pairs_after_confirmation,
        "holdout_verification": holdout_summary,
        "top2_pairs_after_holdout": top2_pairs_after_holdout,
        "extra_depth_summary": extra_depth_summary,
        "final_rerun_regimes": rerun_regimes,
        "rerun_summary": rerun_summary,
        "ambiguity_breaker": ambiguity_summary,
        "outcome": outcome,
        "winner": outcome.get("winner", "unresolved"),
        "configs_used": all_unique_config_names(completed_records),
        "completed_run_count": len(completed_records),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    screening_rows = [
        [
            row["pair_label"],
            f"{row['summary']['dense_mean']['mean']:.4f}",
            f"{row['summary']['last_val']['mean']:.4f}",
            f"{row['summary']['last5_val_mean']['mean']:.4f}",
            f"{row['summary']['best_to_last_drop']['mean']:.4f}",
            f"{row['summary']['composite']['mean']:.4f}",
        ]
        for row in screening_rankings
    ]
    contract_rows = [
        [row["contract_label"], f"{row['summary']['composite']['mean']:.4f}", f"{row['summary']['composite']['std']:.4f}"]
        for row in screening_contract_rows
    ]
    hmix_rows = [
        [
            CONTRACT_LABELS.get(contract, contract),
            f"{hmix_summary.get('hmix', {}).get(contract, {}).get('dense_mean', {}).get('mean', 0.0):.4f}",
            f"{hmix_summary.get('hmix', {}).get(contract, {}).get('last_val', {}).get('mean', 0.0):.4f}",
            f"{hmix_summary.get('hmix', {}).get(contract, {}).get('composite', {}).get('mean', 0.0):.4f}",
        ]
        for contract in promoted_contracts_screening
    ]
    confirmation_rows = [
        [
            row["pair_label"],
            f"{row['summary']['dense_mean']['mean']:.4f}",
            f"{row['summary']['last_val']['mean']:.4f}",
            f"{row['summary']['last5_val_mean']['mean']:.4f}",
            f"{row['summary']['best_to_last_drop']['mean']:.4f}",
            f"{row['summary']['composite']['mean']:.4f}",
        ]
        for row in confirmation_rankings
    ]
    holdout_rows: list[list[object]] = []
    for regime in HOLDOUT_REGIMES:
        for pair in top2_pairs_after_holdout:
            info = parse_pair(pair)
            holdout_rows.append(
                [
                    regime,
                    info["pair_label"],
                    f"{holdout_summary.get(regime, {}).get(pair, {}).get('dense_mean', {}).get('mean', 0.0):.4f}",
                    f"{holdout_summary.get(regime, {}).get(pair, {}).get('last_val', {}).get('mean', 0.0):.4f}",
                    f"{holdout_summary.get(regime, {}).get(pair, {}).get('composite', {}).get('mean', 0.0):.4f}",
                ]
            )
    rerun_rows: list[list[object]] = []
    for regime in rerun_regimes:
        for pair in top2_pairs_after_holdout:
            info = parse_pair(pair)
            rerun_rows.append(
                [
                    regime,
                    info["pair_label"],
                    f"{rerun_summary.get(regime, {}).get(pair, {}).get('composite', {}).get('mean', 0.0):.4f}",
                ]
            )
    extra_rows: list[list[object]] = []
    for pair in top2_pairs_after_holdout:
        info = parse_pair(pair)
        for regime in EXTRA_DEPTH_REGIMES:
            depth_payload = extra_depth_summary.get(pair, {}).get(regime, {})
            settle = depth_payload.get("depths", {}).get("settle", {})
            extra_rows.append(
                [
                    info["pair_label"],
                    regime,
                    f"{settle.get('best_dense_mean', 0.0):.4f}",
                    f"{settle.get('settle_rate', 0.0):.4f}",
                    f"{settle.get('steps_to_settle', 0.0):.2f}",
                    f"{settle.get('accept_on_settle_accuracy', 0.0):.4f}",
                ]
            )
    core_choice_rows = [
        [
            row["contract_label"],
            f"{row['lr_multiplier']:.2f}",
            f"{row['p_keep_prev']:.2f}",
            f"{row['summary']['composite']['mean']:.4f}",
        ]
        for row in core_rankings
    ]
    pilot_choice_rows = [
        [
            parse_pair(pair)["pair_label"],
            f"{choice['lr_multiplier']:.2f}",
            f"{choice['p_keep_prev']:.2f}",
            f"{choice['aux_anneal_final_multiplier']:.2f}",
            f"{choice['summary']['composite']['mean']:.4f}",
        ]
        for pair, choice in sorted(selected_settings.items())
    ]

    report = f"""# APSGNN v64: DS Contract Factorization

## What Changed From v63

v64 narrows the post-v63 question to DS factorization and stronger DS-like contracts. This round keeps the static selector bases fixed at `V` and `VT-0.5`, reuses the same 32-leaf APSGNN family, and tests whether the remaining instability is really in temporal credit assignment and training dynamics rather than selector weights.

## Budgets

- `P = 420`
- `S = 1134`
- `M = 2268`
- `L = 3024`
- visible GPUs used: `{summary['visible_gpu_count']}`
- rolling late-stage window: `{ROLLING_N}`

## Exact Regimes

- `Core`, `T1`, `T2a`, `Hmix` for the main matrices
- `T1r`, `T2b`, `T2c`, `Hmid` for holdout verification

## DS-Core Pilot Screen

{render_table(["Contract", "LR x", "p_keep_prev", "Composite"], core_choice_rows)}

Chosen DS-core contracts:

- best: `{core_best}`
- runner-up: `{core_runner_up}`

## Chosen Pair Settings

{render_table(["Pair", "LR x", "p_keep_prev", "Aux Final", "Composite"], pilot_choice_rows)}

## Exact Configs Used

{chr(10).join(f"- `{config_name}`" for config_name in summary["configs_used"])}

## Screening Summary

{render_table(["Pair", "Dense", "Last", "Last5", "Drop", "Composite"], screening_rows)}

### Contract Ranking

{render_table(["Contract", "Composite", "Std"], contract_rows)}

Promoted contracts to `Hmix` tiebreak: `{", ".join(promoted_contracts_screening)}`

## Hmix Contract Tiebreak

{render_table(["Contract", "Dense", "Last", "Composite"], hmix_rows)}

Top 2 contracts after `Hmix`: `{", ".join(top2_contracts_after_hmix)}`

## Confirmation Summary

{render_table(["Pair", "Dense", "Last", "Last5", "Drop", "Composite"], confirmation_rows)}

## Holdout Verification

{render_table(["Regime", "Pair", "Dense", "Last", "Composite"], holdout_rows)}

## Extra Compute / Settling

{render_table(["Pair", "Regime", "Settle Dense", "Settle Rate", "Steps", "Accept-on-settle"], extra_rows)}

## Fresh Reruns

Chosen rerun regimes: `{", ".join(rerun_regimes)}`

{render_table(["Regime", "Pair", "Composite"], rerun_rows)}

## Ambiguity Breaker

{json.dumps(ambiguity_summary, indent=2) if ambiguity_summary else "Not triggered yet."}

## Final Diagnosis

- outcome: `{outcome.get('outcome', 'unresolved')}`
- winner: `{outcome.get('winner', 'unresolved')}`

## Best Next Experiment

If the result is still unresolved, keep the best DS-like contract family and continue refining temporal credit assignment rather than reopening selector-family search.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
