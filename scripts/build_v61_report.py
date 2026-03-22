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
SUMMARY_PATH = REPORTS / "summary_metrics_v61.json"
REPORT_PATH = REPORTS / "final_report_v61_selector_operationalization.md"

BASE_LR = 2.0e-4
ROLLING_N = 5
EXPECTED_TRAIN_STEPS = {"p": 420, "s": 1260, "m": 2520, "l": 3360}

ARM_INFO = {
    "visitonly": {"family": "visitonly", "category": "static", "label": "V"},
    "visit_taskgrad_half": {"family": "visit_taskgrad_half", "category": "static", "label": "VT-0.5"},
    "gate_writers_le2": {"family": "g_writers", "category": "gate", "label": "G_writers<=2"},
    "gate_writers_le3": {"family": "g_writers", "category": "gate", "label": "G_writers<=3"},
    "gate_ingress_pool1": {"family": "g_ingress", "category": "gate", "label": "G_ingress(pool1)"},
    "gate_ingress_pool1_or_tightttl": {
        "family": "g_ingress",
        "category": "gate",
        "label": "G_ingress(pool1|tightttl)",
    },
    "gate_meta_a": {"family": "g_meta", "category": "gate", "label": "G_meta(A)"},
    "gate_meta_b": {"family": "g_meta", "category": "gate", "label": "G_meta(B)"},
    "gate_online_a": {"family": "g_online", "category": "gate", "label": "G_online(A)"},
    "gate_online_b": {"family": "g_online", "category": "gate", "label": "G_online(B)"},
}

ARM_SETTINGS = {
    "visitonly": "Static `z(task_visits)`",
    "visit_taskgrad_half": "Static `z(task_visits) + 0.5*z(task_grad)`",
    "gate_writers_le2": "Use `VT-0.5` when writers <= 2, else `V`",
    "gate_writers_le3": "Use `VT-0.5` when writers <= 3, else `V`",
    "gate_ingress_pool1": "Use `VT-0.5` when `start_node_pool_size == 1`, else `V`",
    "gate_ingress_pool1_or_tightttl": "Use `VT-0.5` when `start_node_pool_size == 1` or TTL is tight, else `V`",
    "gate_meta_a": "Metadata linear gate A on writers / ingress / tight TTL",
    "gate_meta_b": "Metadata linear gate B on writers / ingress / tight TTL",
    "gate_online_a": "Early-stage online gate A using visit entropy / Gini after stage 4",
    "gate_online_b": "Early-stage online gate B using visit entropy / Gini after stage 4",
}

FAMILY_LABELS = {
    "visitonly": "V",
    "visit_taskgrad_half": "VT-0.5",
    "g_writers": "G_writers",
    "g_ingress": "G_ingress",
    "g_meta": "G_meta",
    "g_online": "G_online",
}

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

KNOWN_REGIMES = ["core", "t1", "t1r", "t2a", "t2b", "t2c"]
REPRESENTATIVE_REGIMES = ["core", "t1", "t2a"]
HOLDOUT_REGIMES = ["hmid", "hmix"]

RUN_RE = re.compile(
    r"v61-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
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


def latest_v61_runs() -> list[tuple[dict[str, str], Path]]:
    latest: dict[tuple[str, str, str, str, str], Path] = {}
    for candidate in sorted(RUNS.glob("*-v61-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        key = (meta["regime"], meta["arm"], meta["schedule"], meta["tag"], meta["seed"])
        latest[key] = candidate
    return [(parse_run_name(path.name) or {}, path) for path in latest.values()]


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


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def summarize_run(run_dir: Path, meta: dict[str, str]) -> dict[str, Any]:
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(ROLLING_N, len(vals)) :]
    writers = REGIME_WRITERS[meta["regime"]]
    dense_writers = writers[1:] if len(writers) > 1 else writers
    record: dict[str, Any] = {
        "run": run_dir.name,
        "regime": meta["regime"],
        "arm": meta["arm"],
        "family": ARM_INFO[meta["arm"]]["family"],
        "family_label": FAMILY_LABELS[ARM_INFO[meta["arm"]]["family"]],
        "category": ARM_INFO[meta["arm"]]["category"],
        "arm_label": ARM_INFO[meta["arm"]]["label"],
        "schedule": meta["schedule"],
        "seed": int(meta["seed"]),
        "tag": meta["tag"],
        "config_name": f"configs/v61_{meta['regime']}_{meta['arm']}_32_{meta['schedule']}.yaml",
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": round(float(config["train"]["lr"]) / BASE_LR, 4),
        "train_steps": int(config["train"]["train_steps"]),
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[f"k{writer}"] for writer in dense_writers])
    record["last_dense_mean"] = mean([record[f"last_k{writer}"] for writer in dense_writers])
    record["composite"] = 0.45 * record["dense_mean"] + 0.35 * record["last_val"] + 0.20 * record["last5_val_mean"]
    record["score"] = record["last_val"] + record["dense_mean"]
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
        "score": mean_std([record["score"] for record in records]),
        "lr_multiplier": mean_std([record["lr_multiplier"] for record in records]),
        "runs": [record["run"] for record in records],
    }
    regime = records[0]["regime"]
    writers = REGIME_WRITERS[regime]
    for writer in writers:
        if all(f"k{writer}" in record for record in records):
            out[f"k{writer}"] = mean_std([record[f"k{writer}"] for record in records])
        if all(f"last_k{writer}" in record for record in records):
            out[f"last_k{writer}"] = mean_std([record[f"last_k{writer}"] for record in records])
    return out


def candidate_key(record: dict[str, Any]) -> str:
    return f"{record['arm']}@lr{record['lr_multiplier']:.1f}"


def rank_candidates(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for record in records:
        grouped.setdefault(record["family"], {}).setdefault(candidate_key(record), []).append(record)
    ranked: dict[str, dict[str, Any]] = {}
    for family, candidates in grouped.items():
        summaries = []
        for key, candidate_records in candidates.items():
            summaries.append(
                {
                    "candidate": key,
                    "arm": candidate_records[0]["arm"],
                    "label": candidate_records[0]["arm_label"],
                    "lr_multiplier": candidate_records[0]["lr_multiplier"],
                    "summary": summarize_group(candidate_records),
                }
            )
        summaries.sort(key=lambda row: row["summary"]["composite"]["mean"], reverse=True)
        ranked[family] = {
            "label": FAMILY_LABELS[family],
            "candidates": summaries,
            "best": summaries[0] if summaries else {},
        }
    return ranked


def summarize_by_regime_and_arm(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for regime in sorted({record["regime"] for record in records}):
        out[regime] = {}
        for arm in sorted({record["arm"] for record in records if record["regime"] == regime}):
            out[regime][arm] = summarize_group(
                [record for record in records if record["regime"] == regime and record["arm"] == arm]
            )
    return out


def summarize_by_family(records: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for family in sorted({record["family"] for record in records}):
        family_records = [record for record in records if record["family"] == family]
        arm_names = sorted({record["arm"] for record in family_records})
        out[family] = {
            "label": FAMILY_LABELS[family],
            "arms": arm_names,
            "summary": summarize_group(family_records),
        }
    return out


def choose_best_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    gate_records = [record for record in records if record["category"] == "gate"]
    families = summarize_by_family(gate_records)
    ranked = sorted(
        (
            {
                "family": family,
                "label": payload["label"],
                "arms": payload["arms"],
                "summary": payload["summary"],
            }
            for family, payload in families.items()
        ),
        key=lambda row: row["summary"]["composite"]["mean"],
        reverse=True,
    )
    return ranked[0] if ranked else {}


def exploitation_rule_scores(exploitation_summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    v_total = 0.0
    vt_total = 0.0
    keyed_total = 0.0
    for regime in KNOWN_REGIMES:
        v_score = exploitation_summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
        vt_score = (
            exploitation_summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
        )
        v_total += v_score
        vt_total += vt_score
        keyed_total += vt_score if regime in {"core", "t2a"} else v_score
    return {
        "single_selector_v": v_total,
        "single_selector_vt_half": vt_total,
        "known_keyed_rule": keyed_total,
    }


def regime_winner_rows(exploitation_summary: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for regime in KNOWN_REGIMES:
        v_comp = exploitation_summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
        vt_comp = (
            exploitation_summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
        )
        winner = "V" if v_comp >= vt_comp else "VT-0.5"
        rows.append([regime, winner, f"{v_comp:.4f}", f"{vt_comp:.4f}"])
    return rows


def holdout_totals(holdout_summary: dict[str, dict[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = {}
    for regime_payload in holdout_summary.values():
        for arm, payload in regime_payload.items():
            totals.setdefault(arm, 0.0)
            totals[arm] += payload.get("composite", {}).get("mean", 0.0)
    return totals


def holdout_winner_rows(holdout_summary: dict[str, dict[str, Any]]) -> list[list[str]]:
    rows: list[list[str]] = []
    for regime in HOLDOUT_REGIMES:
        regime_payload = holdout_summary.get(regime, {})
        if not regime_payload:
            continue
        ranked = sorted(
            (
                (ARM_INFO[arm]["label"], payload.get("composite", {}).get("mean", 0.0))
                for arm, payload in regime_payload.items()
            ),
            key=lambda row: row[1],
            reverse=True,
        )
        winner, winner_score = ranked[0]
        runner_score = ranked[1][1] if len(ranked) > 1 else 0.0
        rows.append([regime, winner, f"{winner_score:.4f}", f"{winner_score - runner_score:.4f}"])
    return rows


def make_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return ""
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(rendered)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    for meta, run_dir in latest_v61_runs():
        if not meta or not is_complete_run(run_dir, meta["schedule"]):
            continue
        all_records.append(summarize_run(run_dir, meta))
    all_records.sort(key=lambda record: (record["schedule"], record["regime"], record["arm"], record["seed"], record["tag"]))

    pilot_records = [record for record in all_records if record["schedule"] == "p" and record["regime"] in {"core", "t1"}]
    exploitation_records = [
        record
        for record in all_records
        if record["schedule"] == "m" and record["arm"] in {"visitonly", "visit_taskgrad_half"} and record["regime"] in KNOWN_REGIMES
    ]
    exploration_records = [
        record
        for record in all_records
        if record["schedule"] == "s" and record["category"] == "gate" and record["regime"] in REPRESENTATIVE_REGIMES
    ]
    holdout_records = [
        record for record in all_records if record["schedule"] == "l" and record["regime"] in HOLDOUT_REGIMES
    ]

    pilot_rankings = rank_candidates(pilot_records)
    exploitation_summary = summarize_by_regime_and_arm(exploitation_records)
    exploration_summary = summarize_by_family(exploration_records)
    holdout_summary = summarize_by_regime_and_arm(holdout_records)
    best_gate = choose_best_gate(exploration_records)
    known_rule_scores = exploitation_rule_scores(exploitation_summary)
    holdout_score_totals = holdout_totals(holdout_summary)

    config_names = sorted({record["config_name"] for record in all_records})
    gpu_count = visible_gpu_count()

    payload = {
        "rollup_window": ROLLING_N,
        "visible_gpu_count": gpu_count,
        "records": all_records,
        "pilot_rankings": pilot_rankings,
        "exploitation_summary": exploitation_summary,
        "exploration_summary": exploration_summary,
        "best_gate": best_gate,
        "holdout_summary": holdout_summary,
        "known_rule_scores": known_rule_scores,
        "config_names": config_names,
    }
    SUMMARY_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    calibration_rows = []
    for family, payload_family in pilot_rankings.items():
        best = payload_family.get("best", {})
        summary = best.get("summary", {})
        calibration_rows.append(
            [
                payload_family["label"],
                best.get("label", ""),
                f"{best.get('lr_multiplier', 0.0):.1f}",
                f"{summary.get('composite', {}).get('mean', 0.0):.4f}",
                f"{summary.get('dense_mean', {}).get('mean', 0.0):.4f}",
                f"{summary.get('last_val', {}).get('mean', 0.0):.4f}",
            ]
        )
    calibration_table = make_table(
        ["Family", "Chosen Variant", "LR x", "Pilot Composite", "Dense", "Last"],
        calibration_rows,
    )

    completed_rows = [
        [
            record["schedule"].upper(),
            record["regime"],
            record["arm_label"],
            record["seed"],
            f"{record['lr_multiplier']:.1f}",
            f"{record['best_val']:.4f}",
            f"{record['last_val']:.4f}",
            f"{record['dense_mean']:.4f}",
            f"{record['composite']:.4f}",
        ]
        for record in all_records
    ]
    completed_table = make_table(
        ["Sched", "Regime", "Arm", "Seed", "LR x", "Best", "Last", "Dense", "Composite"],
        completed_rows,
    )

    exploitation_rows = []
    for regime in KNOWN_REGIMES:
        for arm in ["visitonly", "visit_taskgrad_half"]:
            summary = exploitation_summary.get(regime, {}).get(arm, {})
            if not summary:
                continue
            exploitation_rows.append(
                [
                    regime,
                    ARM_INFO[arm]["label"],
                    f"{summary['best_val']['mean']:.4f}",
                    f"{summary['last_val']['mean']:.4f}",
                    f"{summary['dense_mean']['mean']:.4f}",
                    f"{summary['composite']['mean']:.4f}",
                ]
            )
    exploitation_table = make_table(
        ["Regime", "Arm", "Best", "Last", "Dense", "Composite"],
        exploitation_rows,
    )
    exploitation_winner_table = make_table(
        ["Regime", "Winner", "V Composite", "VT-0.5 Composite"],
        regime_winner_rows(exploitation_summary),
    )

    exploration_rows = []
    for family, summary in exploration_summary.items():
        exploration_rows.append(
            [
                summary["label"],
                ", ".join(ARM_INFO[arm]["label"] for arm in summary["arms"]),
                f"{summary['summary']['best_val']['mean']:.4f}",
                f"{summary['summary']['last_val']['mean']:.4f}",
                f"{summary['summary']['dense_mean']['mean']:.4f}",
                f"{summary['summary']['composite']['mean']:.4f}",
            ]
        )
    exploration_table = make_table(
        ["Gate Family", "Chosen Variant", "Best", "Last", "Dense", "Composite"],
        exploration_rows,
    )

    holdout_rows = []
    for regime in HOLDOUT_REGIMES:
        for arm, summary in holdout_summary.get(regime, {}).items():
            holdout_rows.append(
                [
                    regime,
                    ARM_INFO[arm]["label"],
                    f"{summary['best_val']['mean']:.4f}",
                    f"{summary['last_val']['mean']:.4f}",
                    f"{summary['dense_mean']['mean']:.4f}",
                    f"{summary['composite']['mean']:.4f}",
                ]
            )
    holdout_table = make_table(
        ["Holdout", "Arm", "Best", "Last", "Dense", "Composite"],
        holdout_rows,
    )
    holdout_winners_table = make_table(
        ["Holdout", "Winner", "Composite", "Margin"],
        holdout_winner_rows(holdout_summary),
    )

    chosen_static = "V" if known_rule_scores["single_selector_v"] >= known_rule_scores["single_selector_vt_half"] else "VT-0.5"
    specialist_note = []
    for regime in ["core", "t2a"]:
        v_comp = exploitation_summary.get(regime, {}).get("visitonly", {}).get("composite", {}).get("mean", 0.0)
        vt_comp = exploitation_summary.get(regime, {}).get("visit_taskgrad_half", {}).get("composite", {}).get("mean", 0.0)
        specialist_note.append(f"`{regime}` -> {'VT-0.5' if vt_comp >= v_comp else 'V'}")
    holdout_gate_text = "No holdout runs were found yet."
    if holdout_summary and best_gate:
        gate_arm = best_gate["arms"][0]
        gate_total = holdout_score_totals.get(gate_arm, 0.0)
        v_total = holdout_score_totals.get("visitonly", 0.0)
        vt_total = holdout_score_totals.get("visit_taskgrad_half", 0.0)
        ranked = sorted(
            [
                ("V", v_total),
                ("VT-0.5", vt_total),
                (ARM_INFO[gate_arm]["label"], gate_total),
            ],
            key=lambda row: row[1],
            reverse=True,
        )
        holdout_gate_text = (
            f"Holdout composite totals: `{ranked[0][0]} {ranked[0][1]:.4f}`, "
            f"`{ranked[1][0]} {ranked[1][1]:.4f}`, `{ranked[2][0]} {ranked[2][1]:.4f}`."
        )

    chosen_gate_rows = []
    for family, payload_family in pilot_rankings.items():
        best = payload_family.get("best", {})
        arm = best.get("arm", "")
        chosen_gate_rows.append(
            [
                payload_family["label"],
                best.get("label", ""),
                ARM_SETTINGS.get(arm, ""),
                f"{best.get('lr_multiplier', 0.0):.1f}",
            ]
        )
    chosen_gate_table = make_table(["Family", "Chosen Arm", "Chosen Setting", "LR x"], chosen_gate_rows)

    REPORT_PATH.write_text(
        "\n".join(
            [
                "# APSGNN v61: Selector Operationalization",
                "",
                "## What Changed From v60",
                "",
                "v61 keeps the v60 static conclusion under direct matched re-check, then uses the other half of the budget to test four non-oracle gates that choose between `V` and `VT-0.5` without using the regime label directly.",
                "",
                "This round is split 50/50 between exploitation and exploration so the static v60 answer gets a clean re-check while adaptive gates still receive a fair enough screening matrix and fresh holdout verification before being discarded.",
                "",
                "## Budgets",
                "",
                "- `P = 420`",
                "- `S = 1260`",
                "- `M = 2520`",
                "- `L = 3360`",
                f"- visible GPUs used: `{gpu_count}`",
                f"- rolling late-stage window: `{ROLLING_N}` evals",
                "",
                "## Exact Regimes",
                "",
                "- Known: `Core`, `T1`, `T1r`, `T2a`, `T2b`, `T2c`",
                "- Holdouts: `Hmid`, `Hmix`",
                "- Holdout configs are distinct from the known six because they use writer density `3` and eval densities `3/7/11`, with `Hmix` also setting `start_node_pool_size = 1`.",
                "",
                "## Calibration Summary",
                "",
                calibration_table,
                "",
                "Chosen settings after pilot:",
                "",
                chosen_gate_table,
                "",
                "## Exact Configs Used",
                "",
                *[f"- `{name}`" for name in config_names],
                "",
                "## Completed Runs",
                "",
                completed_table,
                "",
                "## Exploitation Summary",
                "",
                exploitation_table,
                "",
                "Known-regime composite totals:",
                f"- `V everywhere`: `{known_rule_scores['single_selector_v']:.6f}`",
                f"- `VT-0.5 everywhere`: `{known_rule_scores['single_selector_vt_half']:.6f}`",
                f"- `Known keyed rule`: `{known_rule_scores['known_keyed_rule']:.6f}`",
                "",
                "Known-regime winners:",
                "",
                exploitation_winner_table,
                "",
                "## Exploration Summary",
                "",
                exploration_table,
                "",
                "## Holdout Verification",
                "",
                holdout_table,
                "",
                holdout_winners_table,
                "",
                "## Best Gate",
                "",
                (
                    f"Top gate from screening: `{best_gate.get('label', '')}` "
                    f"with composite `{best_gate.get('summary', {}).get('composite', {}).get('mean', 0.0):.6f}`."
                    if best_gate
                    else "No gate runs were found."
                ),
                "",
                holdout_gate_text,
                "",
                "## Final Diagnosis",
                "",
                f"- Best single static selector on the known-regime `M` matrix so far: `{chosen_static}`.",
                f"- Home / ingress-stress specialist check: {', '.join(specialist_note)}.",
                (
                    f"- Best non-oracle gate kept alive through holdouts: `{best_gate.get('label', '')}` "
                    f"using `{ARM_SETTINGS.get(best_gate.get('arms', [''])[0], '')}`."
                    if best_gate
                    else "- No adaptive gate survived screening."
                ),
                "- Interpret the final recommendation from the combination of known-regime totals and the holdout verification totals above.",
                "",
                f"- Summary JSON: [summary_metrics_v61.json]({SUMMARY_PATH})",
                f"- Report: [final_report_v61_selector_operationalization.md]({REPORT_PATH})",
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
