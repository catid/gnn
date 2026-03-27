#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path
from typing import Any

import torch
import yaml

from apsgnn.probes import fit_linear_probe


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v87.json"
REPORT_PATH = REPORTS / "final_report_v87_collision_gated_retrieved_correction.md"
PACK_DEFS_PATH = REPORTS / "v87_pack_definitions.json"
BASE_LR = 2.0e-4
EXPECTED_TRAIN_STEPS = {"p": 300, "m": 900, "l": 1350}
RUN_RE = re.compile(
    r"v87-collision-(?P<regime>c[12])-(?P<condition>ambig|retrhead)-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|m|l)(?:-(?P<tag>(?!s\d+$)[^-]+))?(?:-s(?P<seed>\d+))?$"
)


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


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def latest_runs() -> list[tuple[dict[str, str], Path]]:
    pilot_runs: list[tuple[dict[str, str], Path]] = []
    latest: dict[tuple[str, str, str, str, str], tuple[dict[str, str], Path]] = {}
    for candidate in sorted(RUNS.glob("*-v87-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["schedule"] == "p":
            pilot_runs.append((meta, candidate))
            continue
        latest[(meta["regime"], meta["condition"], meta["pair"], meta["schedule"], meta["seed"])] = (
            meta,
            candidate,
        )
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


def eval_metric(run_dir: Path, kind: str, writers: int) -> dict[str, float]:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return {}
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def score_record(record: dict[str, Any]) -> float:
    return 0.45 * record["dense_mean"] + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    writers = list(config["task"].get("train_eval_writers", [config["task"]["writers_per_episode"]]))
    best_metrics = [eval_metric(run_dir, "best", k) for k in writers]
    last_metrics = [eval_metric(run_dir, "last", k) for k in writers]
    dense_mean = mean([item.get("query_accuracy", 0.0) for item in best_metrics])
    last_dense_mean = mean([item.get("query_accuracy", 0.0) for item in last_metrics])
    merged_metric = best_metrics[0] if best_metrics else {}
    record = {
        "run": run_dir.name,
        "regime": meta["regime"],
        "condition": meta["condition"],
        "pair": meta["pair"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"] or config["train"]["seed"]),
        "tag": meta["tag"],
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "dense_mean": dense_mean,
        "last_dense_mean": last_dense_mean,
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "query_delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
        "retrieval_top_mass": float(merged_metric.get("retrieval_top_mass", 0.0)),
        "retrieval_entropy": float(merged_metric.get("retrieval_entropy", 0.0)),
        "retrieval_cache_entries": float(merged_metric.get("retrieval_cache_entries", 0.0)),
        "retrieval_target_entry_hit_rate": float(merged_metric.get("retrieval_target_entry_hit_rate", 0.0)),
        "readout_enabled": bool(config["model"].get("cache_output_summary_readout", False)),
        "gate_feature_scale": float(config["model"].get("cache_output_gate_feature_scale", 0.0)),
    }
    record["pilot_score"] = score_record(record)
    return record


def group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "count": len(records),
        "dense_mean": mean_std([record["dense_mean"] for record in records]),
        "last_dense_mean": mean_std([record["last_dense_mean"] for record in records]),
        "last_val": mean_std([record["last_val"] for record in records]),
        "last5_val_mean": mean_std([record["last5_val_mean"] for record in records]),
        "pilot_score": mean_std([record["pilot_score"] for record in records]),
        "query_delivery_rate": mean_std([record["query_delivery_rate"] for record in records]),
        "query_first_hop_home_rate": mean_std([record["query_first_hop_home_rate"] for record in records]),
        "home_to_out_rate": mean_std([record["home_to_out_rate"] for record in records]),
        "retrieval_top_mass": mean_std([record["retrieval_top_mass"] for record in records]),
        "retrieval_entropy": mean_std([record["retrieval_entropy"] for record in records]),
        "retrieval_cache_entries": mean_std([record["retrieval_cache_entries"] for record in records]),
        "retrieval_target_entry_hit_rate": mean_std(
            [record["retrieval_target_entry_hit_rate"] for record in records]
        ),
    }


def choose_pilot(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    candidates = [r for r in records if r["schedule"] == "p" and r["condition"] == condition and r["tag"]]
    if not candidates:
        return {}
    best = max(candidates, key=lambda record: record["pilot_score"])
    return {
        "lr_multiplier": best["lr_multiplier"],
        "pilot_score": best["pilot_score"],
        "run": best["run"],
    }


def split_probe_tensors(payload: dict[str, torch.Tensor], key: str) -> dict[str, torch.Tensor] | None:
    if key not in payload or "labels" not in payload:
        return None
    x = payload[key].float()
    y = payload["labels"].long()
    count = int(y.numel())
    if count < 12:
        return None
    train_end = max(int(count * 0.6), 4)
    valid_end = max(int(count * 0.8), train_end + 2)
    valid_end = min(valid_end, count - 1)
    return {
        "train_x": x[:train_end],
        "train_y": y[:train_end],
        "valid_x": x[train_end:valid_end],
        "valid_y": y[train_end:valid_end],
        "test_x": x[valid_end:],
        "test_y": y[valid_end:],
    }


def probe_accuracy(path: Path, key: str) -> float:
    if not path.exists():
        return 0.0
    payload = torch.load(path, map_location="cpu")
    splits = split_probe_tensors(payload, key)
    if splits is None:
        return 0.0
    num_classes = int(torch.max(payload["labels"]).item()) + 1
    result = fit_linear_probe(
        train_x=splits["train_x"],
        train_y=splits["train_y"],
        valid_x=splits["valid_x"],
        valid_y=splits["valid_y"],
        test_x=splits["test_x"],
        test_y=splits["test_y"],
        num_classes=num_classes,
        steps=200,
        lr=0.1,
    )
    return float(result.test_accuracy)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for meta, run_dir in latest_runs():
        if not is_complete_run(run_dir, meta["schedule"]):
            continue
        records.append(summarize_run(run_dir, meta))

    pilot_choices = {
        "ambig": choose_pilot(records, "ambig"),
        "retrhead": choose_pilot(records, "retrhead"),
    }

    main_records = [r for r in records if r["schedule"] == "m"]
    rerun_records = [r for r in records if r["schedule"] == "l"]
    main_summary: dict[str, dict[str, Any]] = {}
    dense_gaps: dict[str, float] = {}
    for regime in ("c1", "c2"):
        ambig_records = [r for r in main_records if r["regime"] == regime and r["condition"] == "ambig"]
        retrhead_records = [r for r in main_records if r["regime"] == regime and r["condition"] == "retrhead"]
        main_summary[regime] = {
            "ambig": group_summary(ambig_records),
            "retrhead": group_summary(retrhead_records),
        }
        dense_gaps[regime] = (
            main_summary[regime]["retrhead"].get("dense_mean", {}).get("mean", 0.0)
            - main_summary[regime]["ambig"].get("dense_mean", {}).get("mean", 0.0)
        )

    strongest_regime = max(dense_gaps, key=lambda key: dense_gaps[key]) if dense_gaps else "c2"
    rerun_summary = {
        "regime": strongest_regime,
        "ambig": group_summary([r for r in rerun_records if r["regime"] == strongest_regime and r["condition"] == "ambig"]),
        "retrhead": group_summary(
            [r for r in rerun_records if r["regime"] == strongest_regime and r["condition"] == "retrhead"]
        ),
    }

    probe_audit: dict[str, dict[str, Any]] = {}
    for condition in ("ambig", "retrhead"):
        candidates = [r for r in main_records if r["regime"] == "c2" and r["condition"] == condition]
        if not candidates:
            probe_audit[condition] = {}
            continue
        best = max(candidates, key=lambda record: record["dense_mean"])
        probe_path = RUNS / best["run"] / "probe_best.pt"
        probe_audit[condition] = {
            "final_sink_state": probe_accuracy(probe_path, "final_sink_state"),
            "home_cache_mean_state": probe_accuracy(probe_path, "home_cache_mean_state"),
            "home_cache_max_state": probe_accuracy(probe_path, "home_cache_max_state"),
            "home_hidden_state": probe_accuracy(probe_path, "home_hidden_state"),
            "run": best["run"],
        }

    positive = (
        dense_gaps.get("c2", 0.0) > 0.0
        and rerun_summary["retrhead"].get("dense_mean", {}).get("mean", 0.0)
        >= rerun_summary["ambig"].get("dense_mean", {}).get("mean", 0.0)
    )

    summary = {
        "experiment": "v87_collision_gated_retrieved_correction",
        "budgets": {"p": 300, "m": 900, "l": 1350},
        "pack_defs_path": str(PACK_DEFS_PATH),
        "completed_run_count": len(records),
        "pilot_choices": pilot_choices,
        "main_summary": main_summary,
        "dense_gaps": dense_gaps,
        "strongest_regime": strongest_regime,
        "rerun_summary": rerun_summary,
        "probe_audit": probe_audit,
        "positive": bool(positive),
        "next_move": "promote this rescue to a broader collision follow-up" if positive else "do not promote this rescue as the main path",
        "runs": records,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report_lines = [
        "# Final Report v87 Collision-Gated Retrieved Correction",
        "",
        "## What Changed",
        "",
        "v87 tests one narrow architecture improvement beyond v75-v86: keep the ambiguity-aware output gate and add a zero-init retrieved-summary correction head that only contributes under real multi-entry collision.",
        "",
        "## Pilot Choices",
        "",
        f"- Ambiguity-aware LR multiplier: `{pilot_choices['ambig'].get('lr_multiplier', 0.0):.1f}`",
        f"- Collision-retrieved-head LR multiplier: `{pilot_choices['retrhead'].get('lr_multiplier', 0.0):.1f}`",
        "",
        "## Main Results",
        "",
        f"- `c1` ambig dense `{main_summary['c1']['ambig'].get('dense_mean', {}).get('mean', 0.0):.4f}`, retrhead dense `{main_summary['c1']['retrhead'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{dense_gaps.get('c1', 0.0):+.4f}`",
        f"- `c2` ambig dense `{main_summary['c2']['ambig'].get('dense_mean', {}).get('mean', 0.0):.4f}`, retrhead dense `{main_summary['c2']['retrhead'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{dense_gaps.get('c2', 0.0):+.4f}`",
        "",
        "## Fresh Rerun",
        "",
        f"- Regime: `{strongest_regime}`",
        f"- Ambiguity-aware dense: `{rerun_summary['ambig'].get('dense_mean', {}).get('mean', 0.0):.4f}`",
        f"- Collision-retrieved-head dense: `{rerun_summary['retrhead'].get('dense_mean', {}).get('mean', 0.0):.4f}`",
        "",
        "## Probe Audit on C2 Best Checkpoints",
        "",
        f"- Ambiguity-aware sink/cache/cache-max/home probe test acc: `{probe_audit.get('ambig', {}).get('final_sink_state', 0.0):.3f}` / `{probe_audit.get('ambig', {}).get('home_cache_mean_state', 0.0):.3f}` / `{probe_audit.get('ambig', {}).get('home_cache_max_state', 0.0):.3f}` / `{probe_audit.get('ambig', {}).get('home_hidden_state', 0.0):.3f}`",
        f"- Collision-retrieved-head sink/cache/cache-max/home probe test acc: `{probe_audit.get('retrhead', {}).get('final_sink_state', 0.0):.3f}` / `{probe_audit.get('retrhead', {}).get('home_cache_mean_state', 0.0):.3f}` / `{probe_audit.get('retrhead', {}).get('home_cache_max_state', 0.0):.3f}` / `{probe_audit.get('retrhead', {}).get('home_hidden_state', 0.0):.3f}`",
        "",
        "## Conclusion",
        "",
        f"- Positive: `{summary['positive']}`",
        f"- Next move: `{summary['next_move']}`",
    ]
    REPORT_PATH.write_text("\n".join(report_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
