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
SUMMARY_PATH = REPORTS / "summary_metrics_v26.json"
REPORT_PATH = REPORTS / "final_report_v26_50_50_selector_campaign.md"
LR_CHOICES_PATH = REPORTS / "v26_lr_choices.json"
BASE_LR = 2.0e-4

SELECTORS = [
    "visitonly",
    "visit_taskgrad_half",
    "visit_taskgrad_025",
    "visit_taskgrad_0375",
    "visit_taskgrad_0625",
    "visit_taskgrad_half_querygrad_0125",
    "stageadaptive_vt",
    "querygradonly",
]

EXPLORATION_SELECTORS = [
    "visit_taskgrad_025",
    "visit_taskgrad_0375",
    "visit_taskgrad_0625",
    "visit_taskgrad_half_querygrad_0125",
    "stageadaptive_vt",
    "querygradonly",
]

DISPLAY = {
    "visitonly": "V",
    "visit_taskgrad_half": "VT-0.5",
    "visit_taskgrad_025": "VT-0.25",
    "visit_taskgrad_0375": "VT-0.375",
    "visit_taskgrad_0625": "VT-0.625",
    "visit_taskgrad_half_querygrad_0125": "VT-0.5+Q0.125",
    "stageadaptive_vt": "StageAdaptive-VT",
    "querygradonly": "Q",
}

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
    "t2b": [4, 8, 12, 14],
    "t2c": [6, 10, 14, 16],
}

SCHEDULE_LABELS = {
    "p": "Pilot-P",
    "s": "Screen-S",
    "m": "Confirm-M",
    "l": "Stress-L",
}

PHASE_LABELS = {
    "core_p": "Core-P",
    "core_s": "Core-S",
    "core_m": "Core-M",
    "core_l": "Core-L",
    "t1_s": "T1-S",
    "t1_m": "T1-M",
    "t1_l": "T1-L",
    "t2a_l": "T2a-L",
    "t2b_l": "T2b-L",
    "t2c_l": "T2c-L",
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


def latest_runs(prefix: str) -> list[tuple[Path, int]]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [(latest_by_seed[seed], seed) for seed in sorted(latest_by_seed)]


def extract_eval(run_dir: Path, kind: str, writers: int) -> float:
    path = run_dir / f"eval_{kind}_k{writers}.json"
    if not path.exists():
        return 0.0
    payload = read_json(path)
    metrics = payload.get("metrics", payload)
    return float(metrics.get("query_accuracy", 0.0))


def screening_composite(record: dict[str, Any]) -> float:
    return 0.45 * float(record["dense_mean"]) + 0.35 * float(record["last_val"]) + 0.20 * float(record["last5_val_mean"])


def summarize_run(run_dir: Path, selector: str, regime: str, schedule: str, seed: int) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    if not vals:
        raise RuntimeError(f"No validation rows in {run_dir / 'metrics.jsonl'}")
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    config = read_json(run_dir / "config.yaml") if (run_dir / "config.yaml").suffix == ".json" else None
    if config is None:
        import yaml

        config = yaml.safe_load((run_dir / "config.yaml").read_text())
    writers = REGIME_WRITERS[regime]
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    record: dict[str, Any] = {
        "selector": selector,
        "selector_label": DISPLAY[selector],
        "regime": regime,
        "schedule": schedule,
        "phase": f"{regime}_{schedule}",
        "seed": seed,
        "run": run_dir.name,
        "lr": float(config["train"]["lr"]),
        "lr_multiplier": float(config["train"]["lr"]) / BASE_LR,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "query_first_hop_home_rate": float(last.get("val/query_first_hop_home_rate", 0.0)),
        "delivery_rate": float(last.get("val/delivery_rate", 0.0)),
        "home_to_out_rate": float(last.get("val/home_to_out_rate", 0.0)),
    }
    for writer in writers:
        record[f"k{writer}"] = extract_eval(run_dir, "best", writer)
        record[f"last_k{writer}"] = extract_eval(run_dir, "last", writer)
    record["dense_mean"] = mean([record[key] for key in dense_keys])
    record["last_dense_mean"] = mean([record[f"last_{key}"] for key in dense_keys])

    coverage = read_json(run_dir / "coverage_summary.json")
    stages = coverage.get("stages", [])
    if stages:
        first = stages[0]
        for step in ("10", "50", "100", "200"):
            record[f"task_visit_{step}"] = float(first.get("task_visit_coverage_at", {}).get(step, 0.0))
            record[f"task_grad_{step}"] = float(first.get("task_grad_coverage_at", {}).get(step, 0.0))
        record["time_to_full_visit"] = float(first.get("time_to_full_visit", 0.0))
        record["time_to_full_grad"] = float(first.get("time_to_full_grad", 0.0))
    record["screen_composite"] = screening_composite(record)
    return record


def phase_records(selector: str, regime: str, schedule: str, *, pilot_lr_tags: list[str] | None = None) -> list[dict[str, Any]]:
    prefixes: list[str]
    if regime == "core" and schedule == "p" and pilot_lr_tags:
        prefixes = [f"v26-core-{selector}-32-p-{tag}" for tag in pilot_lr_tags]
    else:
        prefixes = [f"v26-{regime}-{selector}-32-{schedule}"]
    records: list[dict[str, Any]] = []
    for prefix in prefixes:
        for run_dir, seed in latest_runs(prefix):
            if (run_dir / "coverage_summary.json").exists() and (run_dir / "metrics.jsonl").exists():
                records.append(summarize_run(run_dir, selector, regime, schedule, seed))
    return records


def summarize_phase(records: list[dict[str, Any]], regime: str) -> dict[str, dict[str, Any]]:
    writers = REGIME_WRITERS[regime]
    dense_keys = [f"k{writer}" for writer in writers[1:]]
    out: dict[str, dict[str, Any]] = {}
    for selector in sorted({record["selector"] for record in records}):
        selector_records = [record for record in records if record["selector"] == selector]
        out[selector] = {
            "count": len(selector_records),
            "best_val": mean_std([record["best_val"] for record in selector_records]),
            "last_val": mean_std([record["last_val"] for record in selector_records]),
            "last5_val_mean": mean_std([record["last5_val_mean"] for record in selector_records]),
            "best_to_last_drop": mean_std([record["best_to_last_drop"] for record in selector_records]),
            "dense_mean": mean_std([record["dense_mean"] for record in selector_records]),
            "last_dense_mean": mean_std([record["last_dense_mean"] for record in selector_records]),
            "screen_composite": mean_std([record["screen_composite"] for record in selector_records]),
            "query_first_hop_home_rate": mean_std([record["query_first_hop_home_rate"] for record in selector_records]),
            "delivery_rate": mean_std([record["delivery_rate"] for record in selector_records]),
            "home_to_out_rate": mean_std([record["home_to_out_rate"] for record in selector_records]),
            "lr_multiplier": mean_std([record["lr_multiplier"] for record in selector_records]),
        }
        for key in dense_keys:
            out[selector][key] = mean_std([record[key] for record in selector_records])
    return out


def rank_summary(summary: dict[str, dict[str, Any]], selectors: list[str] | None = None) -> list[str]:
    candidates = selectors if selectors is not None else list(summary)
    candidates = [selector for selector in candidates if selector in summary]
    return sorted(candidates, key=lambda selector: summary[selector]["screen_composite"]["mean"], reverse=True)


def choose_lr_multipliers(pilot_records: list[dict[str, Any]]) -> dict[str, float]:
    by_selector: dict[str, list[dict[str, Any]]] = {}
    for record in pilot_records:
        by_selector.setdefault(record["selector"], []).append(record)
    choices: dict[str, float] = {}
    for selector, records in by_selector.items():
        ranked = sorted(
            records,
            key=lambda item: (item["screen_composite"], -abs(float(item["lr_multiplier"]) - 1.0)),
            reverse=True,
        )
        choices[selector] = float(ranked[0]["lr_multiplier"])
    return choices


def combined_rank(phase_a: dict[str, dict[str, Any]], phase_b: dict[str, dict[str, Any]], selectors: list[str]) -> list[str]:
    def score(selector: str) -> float:
        return (
            phase_a.get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
            + phase_b.get(selector, {}).get("screen_composite", {}).get("mean", 0.0)
        )

    return sorted(selectors, key=score, reverse=True)


def fmt_stat(payload: dict[str, float]) -> str:
    return f"{payload['mean']:.4f} ± {payload['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def run_table(records: list[dict[str, Any]], extra_keys: list[str]) -> str:
    headers = ["Phase", "Sel", "Seed", "LRx", "Best", "Last", "Last5", "Dense", *extra_keys]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["phase"], item["selector"], item["seed"], item["lr_multiplier"])):
        rows.append(
            [
                PHASE_LABELS.get(record["phase"], record["phase"]),
                record["selector_label"],
                str(record["seed"]),
                f"{record['lr_multiplier']:.3f}",
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['last5_val_mean']:.4f}",
                f"{record['dense_mean']:.4f}",
                *[f"{record.get(key, 0.0):.4f}" for key in extra_keys],
            ]
        )
    return markdown_table(headers, rows)


def plot_summary(summary: dict[str, dict[str, Any]], selectors: list[str], title: str, path: Path) -> None:
    selectors = [selector for selector in selectors if selector in summary]
    if not selectors:
        return
    labels = [DISPLAY[selector] for selector in selectors]
    metrics = [("screen_composite", "Composite"), ("dense_mean", "Dense"), ("last_val", "Last")]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4.8 * len(metrics), 4.0))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        means = [summary[selector][metric]["mean"] for selector in selectors]
        stds = [summary[selector][metric]["std"] for selector in selectors]
        ax.bar(labels, means, yerr=stds, capsize=4)
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    REPORTS.mkdir(exist_ok=True)
    pilot_lr_tags = ["lr080", "lr100", "lr060"]
    all_records: list[dict[str, Any]] = []
    phase_summaries: dict[str, dict[str, Any]] = {}

    pilot_records: list[dict[str, Any]] = []
    for selector in SELECTORS:
        pilot_records.extend(phase_records(selector, "core", "p", pilot_lr_tags=pilot_lr_tags))
    if pilot_records:
        all_records.extend(pilot_records)
        phase_summaries["core_p"] = summarize_phase(pilot_records, "core")
        lr_choices = choose_lr_multipliers(pilot_records)
        LR_CHOICES_PATH.write_text(json.dumps(lr_choices, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        lr_choices = {selector: 1.0 for selector in SELECTORS}

    for regime in ("core", "t1", "t2a", "t2b", "t2c"):
        for schedule in ("s", "m", "l"):
            records: list[dict[str, Any]] = []
            for selector in SELECTORS:
                records.extend(phase_records(selector, regime, schedule))
            if not records:
                continue
            phase = f"{regime}_{schedule}"
            all_records.extend(records)
            phase_summaries[phase] = summarize_phase(records, regime)

    core_screen_rank = rank_summary(phase_summaries.get("core_s", {}), EXPLORATION_SELECTORS)
    promoted_t1 = core_screen_rank[:3]
    overall_exploration_rank = combined_rank(
        phase_summaries.get("core_s", {}),
        phase_summaries.get("t1_s", {}),
        promoted_t1,
    )
    promoted_m = overall_exploration_rank[:2]

    exploit_candidates = ["visitonly", "visit_taskgrad_half"]
    exploit_score = {}
    for selector in exploit_candidates:
        exploit_score[selector] = (
            phase_summaries.get("core_m", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("core_m", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t1_m", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_m", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2a_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2a_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2b_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2b_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2c_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2c_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
        )
    exploitation_winner = max(exploit_score, key=exploit_score.get) if any(exploit_score.values()) else "visit_taskgrad_half"

    final_candidates = list(dict.fromkeys([exploitation_winner, *promoted_m]))
    final_scores = {}
    for selector in final_candidates:
        final_scores[selector] = (
            phase_summaries.get("core_m", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("core_m", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t1_m", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_m", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("core_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("core_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t1_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t1_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2a_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2a_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2b_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2b_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
            + phase_summaries.get("t2c_l", {}).get(selector, {}).get("dense_mean", {}).get("mean", 0.0)
            + phase_summaries.get("t2c_l", {}).get(selector, {}).get("last_val", {}).get("mean", 0.0)
        )
    final_rank = sorted(final_scores, key=final_scores.get, reverse=True)

    plot_summary(
        phase_summaries.get("core_s", {}),
        core_screen_rank,
        "v26 Core-S Exploration",
        REPORTS / "v26_exploration_core_screen.png",
    )
    plot_summary(
        phase_summaries.get("t1_s", {}),
        promoted_t1,
        "v26 T1-S Exploration",
        REPORTS / "v26_exploration_t1_screen.png",
    )
    plot_summary(
        phase_summaries.get("core_m", {}),
        exploit_candidates,
        "v26 Core-M Exploitation",
        REPORTS / "v26_exploitation_core_m.png",
    )
    plot_summary(
        phase_summaries.get("t1_m", {}),
        exploit_candidates,
        "v26 T1-M Exploitation",
        REPORTS / "v26_exploitation_t1_m.png",
    )

    summary = {
        "pilot_lr_choices": lr_choices,
        "core_screen_rank": core_screen_rank,
        "promoted_t1": promoted_t1,
        "overall_exploration_rank": overall_exploration_rank,
        "promoted_m": promoted_m,
        "exploitation_winner": exploitation_winner,
        "final_rank": final_rank,
        "phase_summaries": phase_summaries,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pilot_rows = [
        [
            DISPLAY[selector],
            f"{lr_choices.get(selector, 1.0):.3f}",
            fmt_stat(phase_summaries["core_p"][selector]["screen_composite"]) if "core_p" in phase_summaries and selector in phase_summaries["core_p"] else "-",
        ]
        for selector in SELECTORS
        if selector in lr_choices
    ]
    screening_rows = [
        [
            DISPLAY[selector],
            fmt_stat(phase_summaries["core_s"][selector]["screen_composite"]),
            fmt_stat(phase_summaries["core_s"][selector]["dense_mean"]),
            fmt_stat(phase_summaries["core_s"][selector]["last_val"]),
        ]
        for selector in core_screen_rank
        if "core_s" in phase_summaries and selector in phase_summaries["core_s"]
    ]
    report = f"""# APSGNN v26 50/50 Selector Campaign

Visible GPUs used: 2

## Calibration and LR choices

Budgets:
- `P=450` with stage steps `[20, 20, 30, 30, 40, 50, 260]`
- `S=1350` with stage steps `[60, 60, 75, 75, 90, 120, 870]`
- `M=2550` with stage steps `[90, 90, 120, 120, 150, 180, 1800]`
- `L=3570` with stage steps `[120, 120, 150, 150, 180, 220, 2630]`

Pilot LR multipliers:

{markdown_table(["Selector", "Chosen LRx", "Pilot Composite"], pilot_rows) if pilot_rows else "No pilot runs recorded yet."}

## Exploration screening

{markdown_table(["Selector", "Core-S Composite", "Core-S Dense", "Core-S Last"], screening_rows) if screening_rows else "No exploration screening runs recorded yet."}

Promoted to `T1-S`: `{", ".join(DISPLAY[s] for s in promoted_t1) if promoted_t1 else "-"}`  
Promoted to `M`: `{", ".join(DISPLAY[s] for s in promoted_m) if promoted_m else "-"}`  
Current exploitation winner: `{DISPLAY.get(exploitation_winner, exploitation_winner)}`  
Current final rank: `{", ".join(DISPLAY[s] for s in final_rank) if final_rank else "-"}`

## Completed runs

{run_table(all_records, ["query_first_hop_home_rate", "delivery_rate", "home_to_out_rate"]) if all_records else "No runs recorded yet."}
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
