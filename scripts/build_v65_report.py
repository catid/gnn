#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch
import yaml

from apsgnn.probes import bucketed_accuracy, fit_linear_probe, hard_slice_summary


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
SUMMARY_PATH = REPORTS / "summary_metrics_v65.json"
REPORT_PATH = REPORTS / "final_report_v65_architectural_headroom.md"
PACK_DEFS_PATH = REPORTS / "v65_pack_definitions.json"
BASE_LR = 2.0e-4
EXPECTED_TRAIN_STEPS = {"p": 300, "m": 1350, "l": 2160}
RUN_RE = re.compile(
    r"v65-(?P<pack>collision|delay)-(?P<regime>[^-]+)-(?P<cond_a>[^-]+)(?:-(?P<cond_b>[^-]+))?-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|m|l)(?:-(?P<tag>(?!s\d+$)[^-]+))?(?:-s(?P<seed>\d+))?$"
)
PAIR_LABELS = {
    "visitonly_d": "V/D",
    "visit_taskgrad_half_d": "VT-0.5/D",
    "visitonly_ds": "V/DS",
    "visit_taskgrad_half_ds": "VT-0.5/DS",
}


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


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    payload = match.groupdict(default="")
    payload["condition"] = (
        payload["cond_a"] if payload["pack"] == "delay" else f"{payload['cond_a']}-{payload['cond_b']}"
    )
    return payload


def latest_runs() -> list[tuple[dict[str, str], Path]]:
    pilot_runs: list[tuple[dict[str, str], Path]] = []
    latest: dict[tuple[str, str, str, str, str, str], tuple[dict[str, str], Path]] = {}
    for candidate in sorted(RUNS.glob("*-v65-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["schedule"] == "p":
            pilot_runs.append((meta, candidate))
            continue
        latest[(meta["pack"], meta["regime"], meta["condition"], meta["pair"], meta["schedule"], meta["seed"])] = (
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
    best_metrics = [eval_metric(run_dir, "best", writers_per_episode) for writers_per_episode in writers]
    last_metrics = [eval_metric(run_dir, "last", writers_per_episode) for writers_per_episode in writers]
    dense_mean = mean([item.get("query_accuracy", 0.0) for item in best_metrics])
    last_dense_mean = mean([item.get("query_accuracy", 0.0) for item in last_metrics])
    merged_metric = best_metrics[0] if best_metrics else {}
    record = {
        "run": run_dir.name,
        "pack": meta["pack"],
        "regime": meta["regime"],
        "condition": meta["condition"],
        "pair": meta["pair"],
        "pair_label": PAIR_LABELS.get(meta["pair"], meta["pair"]),
        "schedule": meta["schedule"],
        "seed": int(meta["seed"] or config["train"]["seed"]),
        "tag": meta["tag"],
        "config_name": (run_dir / "config.yaml").name,
        "train_steps": int(config["train"]["train_steps"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "cache_enabled": bool(config["model"].get("enable_cache", True)),
        "class_slice_enabled": bool(config["model"].get("use_reserved_class_slice", True)),
        "delay_override_mode": str(config["train"].get("delay_override_mode", "learned")),
        "required_delay_min": int(config["task"].get("required_delay_min", 0)),
        "required_delay_max": int(config["task"].get("required_delay_max", 0)),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "dense_mean": dense_mean,
        "last_dense_mean": last_dense_mean,
        "query_delivery_rate": float(last.get("val/query_delivery_rate", 0.0)),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/query_home_to_output_rate", 0.0)),
        "retrieval_top_mass": float(merged_metric.get("retrieval_top_mass", 0.0)),
        "retrieval_entropy": float(merged_metric.get("retrieval_entropy", 0.0)),
        "retrieval_cache_entries": float(merged_metric.get("retrieval_cache_entries", 0.0)),
        "retrieval_target_entry_hit_rate": float(merged_metric.get("retrieval_target_entry_hit_rate", 0.0)),
        "query_first_delay_mean": float(merged_metric.get("query_first_delay_mean", 0.0)),
        "query_first_delay_nonzero_rate": float(merged_metric.get("query_first_delay_nonzero_rate", 0.0)),
        "query_first_delay_match_rate": float(merged_metric.get("query_first_delay_match_rate", 0.0)),
    }
    record["pilot_score"] = score_record(record)
    return record


def select_best_lr(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    for record in records:
        if record["schedule"] != "p" or not record["tag"]:
            continue
        family = f"{record['pack']}|{record['regime']}|{record['condition']}|{record['pair']}"
        if family not in best or record["pilot_score"] > best[family]["pilot_score"]:
            best[family] = {
                "lr_multiplier": record["lr_multiplier"],
                "pilot_score": record["pilot_score"],
            }
    return best


def group_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    return {
        "count": len(records),
        "dense_mean": mean_std([record["dense_mean"] for record in records]),
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
        "query_first_delay_mean": mean_std([record["query_first_delay_mean"] for record in records]),
        "query_first_delay_nonzero_rate": mean_std(
            [record["query_first_delay_nonzero_rate"] for record in records]
        ),
        "query_first_delay_match_rate": mean_std(
            [record["query_first_delay_match_rate"] for record in records]
        ),
    }


def probe_rows(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    payload = torch.load(path, map_location="cpu")
    count = int(payload["labels"].size(0))
    rows: list[dict[str, float]] = []
    for index in range(count):
        rows.append(
            {
                "correct": float(payload.get("correct", torch.zeros(count))[index].item()),
                "delivered": float(payload.get("delivered", torch.zeros(count, dtype=torch.bool))[index].item()),
                "competing_entries": float(payload.get("home_competing_entries", torch.zeros(count))[index].item()),
                "ambiguity": float(1.0 - payload.get("home_retrieval_top_mass", torch.zeros(count))[index].item()),
                "required_delay": float(payload.get("required_delay", torch.zeros(count))[index].item()),
                "chosen_delay": float(payload.get("first_hop_delay", torch.zeros(count))[index].item()),
                "delay_ready": float(payload.get("home_delay_ready", torch.zeros(count, dtype=torch.bool))[index].item()),
                "target_entry_hit": float(payload.get("home_target_entry_hit", torch.zeros(count))[index].item()),
            }
        )
    return rows


def run_probe_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = torch.load(path, map_location="cpu")
    labels = payload["labels"].to(torch.long)

    def split_and_probe(features: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        mask = mask.to(torch.bool)
        if int(mask.sum().item()) < 12:
            return {}
        x = features[mask].to(torch.float32)
        y = labels[mask]
        n = x.size(0)
        a = max(int(n * 0.6), 1)
        b = max(int(n * 0.8), a + 1)
        result = fit_linear_probe(
            train_x=x[:a],
            train_y=y[:a],
            valid_x=x[a:b],
            valid_y=y[a:b],
            test_x=x[b:],
            test_y=y[b:],
            num_classes=32,
            steps=150,
            lr=0.05,
        )
        return {
            "train_accuracy": result.train_accuracy,
            "valid_accuracy": result.valid_accuracy,
            "test_accuracy": result.test_accuracy,
        }

    delivered = payload.get("delivered", torch.zeros(labels.size(0), dtype=torch.bool))
    home_mask = payload.get("home_entry_count", torch.zeros(labels.size(0))) > 0
    return {
        "final_sink_state": split_and_probe(payload["final_sink_state"], delivered),
        "home_cache_mean_state": split_and_probe(payload["home_cache_mean_state"], home_mask),
        "home_hidden_state": split_and_probe(payload["home_hidden_state"], home_mask),
    }


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "visible_gpu_count": visible_gpu_count(),
        "budgets": PACK_DEFS_PATH.exists() and read_json(PACK_DEFS_PATH).get("budgets", {}) or {},
        "completed_run_count": len(records),
        "pilot_choices": select_best_lr(records),
    }

    sanity_records = [
        record
        for record in records
        if record["schedule"] == "p"
        and record["pack"] == "collision"
        and record["regime"] == "c1"
        and record["condition"] == "cacheon-classon"
        and record["tag"] == ""
    ]
    summary["sanity_gate"] = {record["pair"]: record for record in sanity_records}
    contract_scores: dict[str, list[float]] = {"d": [], "ds": []}
    for record in sanity_records:
        contract_scores[record["pair"].rsplit("_", 1)[1]].append(record["pilot_score"])
    contract_means = {key: mean(values) for key, values in contract_scores.items()}
    chosen_contract = "d"
    if contract_means["ds"] > contract_means["d"] + 1.0e-6:
        chosen_contract = "ds"
    summary["chosen_contract_family"] = chosen_contract
    summary["incumbent_pair"] = f"visit_taskgrad_half_{chosen_contract}"
    summary["runner_up_pair"] = f"visitonly_{chosen_contract}"

    collision_records = [
        record
        for record in records
        if record["pack"] == "collision"
        and record["schedule"] == "m"
        and record["pair"] == summary["incumbent_pair"]
        and record["condition"] in {"cacheon-classon", "nocache-classon"}
    ]
    collision_summary: dict[str, Any] = {}
    for regime in ("c0", "c1", "c2"):
        on_records = [record for record in collision_records if record["regime"] == regime and record["condition"] == "cacheon-classon"]
        off_records = [record for record in collision_records if record["regime"] == regime and record["condition"] == "nocache-classon"]
        if on_records or off_records:
            on_summary = group_summary(on_records)
            off_summary = group_summary(off_records)
            collision_summary[regime] = {
                "cache_on": on_summary,
                "cache_off": off_summary,
                "dense_gap": on_summary.get("dense_mean", {}).get("mean", 0.0)
                - off_summary.get("dense_mean", {}).get("mean", 0.0),
            }
    summary["collision_pack"] = collision_summary
    collision_pass = any(
        collision_summary.get(regime, {}).get("dense_gap", 0.0) > 0.05
        and collision_summary.get(regime, {}).get("cache_on", {}).get("retrieval_target_entry_hit_rate", {}).get("mean", 0.0)
        > 0.25
        for regime in ("c1", "c2")
    )
    summary["collision_gate_passed"] = bool(collision_pass)

    class_slice_records = [
        record
        for record in records
        if record["pack"] == "collision"
        and record["schedule"] == "m"
        and record["pair"] == summary["incumbent_pair"]
        and record["regime"] in {"c1", "c2"}
        and record["condition"] in {"cacheon-classon", "cacheon-classoff"}
    ]
    class_slice_summary: dict[str, Any] = {}
    for regime in ("c1", "c2"):
        on_records = [record for record in class_slice_records if record["regime"] == regime and record["condition"] == "cacheon-classon"]
        off_records = [record for record in class_slice_records if record["regime"] == regime and record["condition"] == "cacheon-classoff"]
        if on_records or off_records:
            on_summary = group_summary(on_records)
            off_summary = group_summary(off_records)
            class_slice_summary[regime] = {
                "class_slice_on": on_summary,
                "class_slice_off": off_summary,
                "dense_drop": on_summary.get("dense_mean", {}).get("mean", 0.0)
                - off_summary.get("dense_mean", {}).get("mean", 0.0),
            }
    summary["class_slice_pack"] = class_slice_summary

    delay_records = [
        record
        for record in records
        if record["pack"] == "delay"
        and record["schedule"] == "m"
        and record["pair"] == summary["incumbent_pair"]
        and record["condition"] in {"learned", "zero"}
    ]
    delay_summary: dict[str, Any] = {}
    for regime in ("d0", "d1", "d2"):
        learned_records = [record for record in delay_records if record["regime"] == regime and record["condition"] == "learned"]
        zero_records = [record for record in delay_records if record["regime"] == regime and record["condition"] == "zero"]
        if learned_records or zero_records:
            learned_summary = group_summary(learned_records)
            zero_summary = group_summary(zero_records)
            delay_summary[regime] = {
                "learned": learned_summary,
                "forced_zero": zero_summary,
                "dense_gap": learned_summary.get("dense_mean", {}).get("mean", 0.0)
                - zero_summary.get("dense_mean", {}).get("mean", 0.0),
            }
    summary["delay_pack"] = delay_summary
    delay_pass = any(
        delay_summary.get(regime, {}).get("dense_gap", 0.0) > 0.05
        and delay_summary.get(regime, {}).get("learned", {}).get("query_first_delay_nonzero_rate", {}).get("mean", 0.0) > 0.25
        for regime in ("d1", "d2")
    )
    summary["delay_gate_passed"] = bool(delay_pass)

    delay_validation_records = [
        record
        for record in records
        if record["pack"] == "delay"
        and record["schedule"] == "p"
        and record["regime"] == "d1"
        and record["pair"] == summary["incumbent_pair"]
        and record["tag"] == ""
    ]
    summary["delay_validation"] = {}
    for condition in ("learned", "zero", "random", "fixed"):
        condition_records = [record for record in delay_validation_records if record["condition"] == condition]
        if condition_records:
            summary["delay_validation"][condition] = group_summary(condition_records)

    probe_audits: dict[str, Any] = {}
    hard_slices: dict[str, Any] = {}
    for record in records:
        probe_path = RUNS / record["run"] / "probe_best.pt"
        if not probe_path.exists():
            continue
        key = f"{record['pack']}:{record['regime']}:{record['condition']}:{record['pair']}"
        probe_audits[key] = run_probe_audit(probe_path)
        rows = probe_rows(probe_path)
        if record["pack"] == "collision":
            hard_slices[key] = {
                "summary": hard_slice_summary(
                    rows,
                    difficulty_key="competing_entries",
                    ambiguity_key="ambiguity",
                    correct_key="correct",
                    hard_difficulty_threshold=2.0,
                    hard_ambiguity_threshold=0.25,
                ),
                "bucketed_accuracy": bucketed_accuracy(rows, bucket_key="competing_entries", correct_key="correct"),
            }
        else:
            hard_slices[key] = {
                "summary": hard_slice_summary(
                    rows,
                    difficulty_key="required_delay",
                    ambiguity_key="required_delay",
                    correct_key="correct",
                    hard_difficulty_threshold=2.0,
                    hard_ambiguity_threshold=2.0,
                ),
                "bucketed_accuracy": bucketed_accuracy(rows, bucket_key="required_delay", correct_key="correct"),
            }
    summary["decodability_audit"] = probe_audits
    summary["hard_slice_audit"] = hard_slices

    optional_followup_records = [
        record for record in records if "adaptive" in record["condition"] or "branch" in record["condition"]
    ]
    summary["optional_followup"] = {
        "triggered": bool(optional_followup_records),
        "records": optional_followup_records,
    }

    if summary["collision_gate_passed"] or summary["delay_gate_passed"]:
        headroom = "still_has_headroom"
    else:
        headroom = "architectural_headroom_weakened"
    summary["headroom_conclusion"] = headroom
    return summary


def build_report(summary: dict[str, Any]) -> str:
    lines = [
        "# APSGNN v65: Architectural Headroom",
        "",
        "## What Changed From v64",
        "",
        "v65 stops spending the main budget on selector micro-tiebreaks and instead uses benchmark-pack discipline to test unresolved architectural debt: heavy-collision retrieval, reserved class-slice removal, and a benchmark where nonzero delay is causally necessary.",
        "",
        "## Chosen Contract Family",
        "",
        f"- chosen contract family: `{summary.get('chosen_contract_family', 'pending')}`",
        f"- incumbent pair: `{summary.get('incumbent_pair', 'pending')}`",
        f"- runner-up pair: `{summary.get('runner_up_pair', 'pending')}`",
        "",
        "## Collision Pack Summary",
        "",
    ]
    for regime, payload in summary.get("collision_pack", {}).items():
        lines.append(
            f"- `{regime}`: cache-on dense `{payload['cache_on'].get('dense_mean', {}).get('mean', 0.0):.4f}`, cache-off dense `{payload['cache_off'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{payload.get('dense_gap', 0.0):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Class-Slice Pack Summary",
            "",
        ]
    )
    class_slice_pack = summary.get("class_slice_pack", {})
    if not summary.get("collision_gate_passed", False):
        lines.append("- skipped: collision gate did not pass, so class-slice-off confirmation was not run.")
    for regime, payload in class_slice_pack.items():
        if payload.get("class_slice_off", {}).get("count", 0) <= 0:
            lines.append(
                f"- `{regime}`: class-on reference dense `{payload['class_slice_on'].get('dense_mean', {}).get('mean', 0.0):.4f}`; class-off not run"
            )
            continue
        lines.append(
            f"- `{regime}`: class-on dense `{payload['class_slice_on'].get('dense_mean', {}).get('mean', 0.0):.4f}`, class-off dense `{payload['class_slice_off'].get('dense_mean', {}).get('mean', 0.0):.4f}`, drop `{payload.get('dense_drop', 0.0):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Delay Pack Summary",
            "",
        ]
    )
    if summary.get("delay_validation"):
        lines.append("- validation controls:")
        for condition, payload in summary["delay_validation"].items():
            lines.append(
                f"  - `{condition}` dense `{payload.get('dense_mean', {}).get('mean', 0.0):.4f}`, first-hop nonzero `{payload.get('query_first_delay_nonzero_rate', {}).get('mean', 0.0):.4f}`"
            )
    for regime, payload in summary.get("delay_pack", {}).items():
        lines.append(
            f"- `{regime}`: learned dense `{payload['learned'].get('dense_mean', {}).get('mean', 0.0):.4f}`, forced-zero dense `{payload['forced_zero'].get('dense_mean', {}).get('mean', 0.0):.4f}`, gap `{payload.get('dense_gap', 0.0):.4f}`"
        )
    lines.extend(
        [
            "",
            "## Decodability Audit",
            "",
            f"- audited checkpoints: `{len(summary.get('decodability_audit', {}))}`",
            "",
            "## Final Diagnosis",
            "",
            f"- collision gate passed: `{summary.get('collision_gate_passed', False)}`",
            f"- delay gate passed: `{summary.get('delay_gate_passed', False)}`",
            f"- optional follow-up triggered: `{summary.get('optional_followup', {}).get('triggered', False)}`",
            f"- architectural headroom conclusion: `{summary.get('headroom_conclusion', 'pending')}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    records = []
    for meta, run_dir in latest_runs():
        if not is_complete_run(run_dir, meta["schedule"]):
            continue
        records.append(summarize_run(run_dir, meta))
    summary = build_summary(records)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
