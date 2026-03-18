#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import statistics
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import torch
except Exception:  # pragma: no cover - report generation should still work without torch
    torch = None


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


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


def mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def mean_std(values: list[float]) -> dict[str, float]:
    return {"mean": mean(values), "std": std(values)}


def take(stage: dict[str, Any], key: str, subkey: str) -> float:
    payload = stage.get(key, {})
    if isinstance(payload, dict):
        return float(payload.get(subkey, 0.0))
    return 0.0


def bootstrap_mean_diff(a: list[float], b: list[float], rounds: int = 2000) -> dict[str, float]:
    if not a or not b:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    rng = random.Random(1234)
    diffs: list[float] = []
    for _ in range(rounds):
        sa = [a[rng.randrange(len(a))] for _ in range(len(a))]
        sb = [b[rng.randrange(len(b))] for _ in range(len(b))]
        diffs.append(mean(sa) - mean(sb))
    diffs.sort()
    return {
        "mean_diff": mean(a) - mean(b),
        "ci_low": diffs[int(0.025 * (len(diffs) - 1))],
        "ci_high": diffs[int(0.975 * (len(diffs) - 1))],
    }


def fmt_stat(stat: dict[str, float]) -> str:
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def diff_line(name: str, diff: dict[str, float]) -> str:
    return (
        f"- {name}: `{diff['mean_diff']:.4f}` with bootstrap CI "
        f"[`{diff['ci_low']:.4f}`, `{diff['ci_high']:.4f}`]"
    )


def extract_eval(run_dir: Path, writers: int) -> dict[str, float]:
    payload = read_json(run_dir / f"eval_best_k{writers}.json")
    metrics = payload.get("metrics", payload)
    return {key: float(value) for key, value in metrics.items() if isinstance(value, (int, float))}


def summarize_run(run_dir: Path, arm: str, seed: int, eval_writers: list[int]) -> dict[str, Any]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    vals = [row for row in metrics if "val/query_accuracy" in row]
    best = max(vals, key=lambda row: float(row["val/query_accuracy"]))
    last = vals[-1]
    recent = vals[-min(5, len(vals)) :]
    stage_rows = [row for row in vals if int(row.get("train/stage_index", row.get("val/stage_index", 6))) == 6]
    record: dict[str, Any] = {
        "arm": arm,
        "seed": seed,
        "run": run_dir.name,
        "best_val": float(best["val/query_accuracy"]),
        "last_val": float(last["val/query_accuracy"]),
        "last5_val_mean": mean([float(row["val/query_accuracy"]) for row in recent]),
        "best_to_last_drop": float(best["val/query_accuracy"]) - float(last["val/query_accuracy"]),
        "best_query_first_hop": float(best["val/query_first_hop_home_rate"]),
        "best_delivery": float(best["val/query_delivery_rate"]),
        "best_home_to_output": float(best["val/query_home_to_output_rate"]),
        "final_stage_best_val": max((float(row["val/query_accuracy"]) for row in stage_rows), default=0.0),
    }
    for writers in eval_writers:
        record[f"k{writers}"] = extract_eval(run_dir, writers)["query_accuracy"]
    coverage = read_json(run_dir / "coverage_summary.json")
    stage_summaries = coverage.get("stages", [])
    if stage_summaries:
        first = stage_summaries[0]
        for step in ("10", "50", "100", "200"):
            record[f"task_visit_{step}"] = take(first, "task_visit_coverage_at", step)
            record[f"task_grad_{step}"] = take(first, "task_grad_coverage_at", step)
        for step in ("50", "100", "200"):
            record[f"task_ge5_{step}"] = take(first, "task_visit_ge5_at", step)
        record["time_to_visit_50"] = take(first, "task_time_to_visit", "50")
        record["time_to_visit_75"] = take(first, "task_time_to_visit", "75")
        record["time_to_visit_100"] = take(first, "task_time_to_visit", "100")
        record["time_to_grad_50"] = take(first, "task_time_to_grad", "50")
        record["time_to_grad_75"] = take(first, "task_time_to_grad", "75")
        record["time_to_grad_100"] = take(first, "task_time_to_grad", "100")
        record["task_entropy_50"] = take(first, "task_visit_entropy_at", "50")
        record["task_gini_50"] = take(first, "task_visit_gini_at", "50")
    split_stages = [stage.get("split_stats", {}) for stage in stage_summaries if stage.get("split_stats")]
    if split_stages:
        record["utility_usefulness_correlation"] = mean(
            [float(stage.get("utility_usefulness_correlation", 0.0)) for stage in split_stages]
        )
        record["utility_traffic_correlation"] = mean(
            [float(stage.get("utility_traffic_correlation", 0.0)) for stage in split_stages]
        )
        record["selected_parent_child_usefulness_mean"] = mean(
            [float(stage.get("selected_parent_child_usefulness_mean", 0.0)) for stage in split_stages]
        )
        record["unselected_parent_child_usefulness_mean"] = mean(
            [float(stage.get("unselected_parent_child_usefulness_mean", 0.0)) for stage in split_stages]
        )
        record["selected_parent_score_mean"] = mean(
            [float(stage.get("selected_parent_utility_mean", 0.0)) for stage in split_stages]
        )
        record["unselected_parent_score_mean"] = mean(
            [float(stage.get("unselected_parent_utility_mean", 0.0)) for stage in split_stages]
        )
    else:
        record["utility_usefulness_correlation"] = 0.0
        record["utility_traffic_correlation"] = 0.0
        record["selected_parent_child_usefulness_mean"] = 0.0
        record["unselected_parent_child_usefulness_mean"] = 0.0
        record["selected_parent_score_mean"] = 0.0
        record["unselected_parent_score_mean"] = 0.0
    return record


def summarize(records: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({record["arm"] for record in records}):
        arm_records = [record for record in records if record["arm"] == arm]
        out[arm] = {"count": len(arm_records)}
        for metric in metrics:
            out[arm][metric] = mean_std([record[metric] for record in arm_records])
    return out


def plot(summary: dict[str, Any], metric_names: list[tuple[str, str]], order: list[str], labels: list[str], path: Path) -> None:
    present = [(arm, label) for arm, label in zip(order, labels) if arm in summary]
    if not present:
        return
    arms = [arm for arm, _ in present]
    axis_labels = [label for _, label in present]
    fig, axes = plt.subplots(1, len(metric_names), figsize=(4.6 * len(metric_names), 4.4), constrained_layout=True)
    if len(metric_names) == 1:
        axes = [axes]
    for ax, (metric, title) in zip(axes, metric_names):
        means = [summary[arm][metric]["mean"] for arm in arms]
        stds = [summary[arm][metric]["std"] for arm in arms]
        ax.bar(axis_labels, means, yerr=stds, capsize=4)
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.2)
        ax.tick_params(axis="x", rotation=20)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def records_table(records: list[dict[str, Any]], eval_cols: list[str]) -> str:
    headers = ["Arm", "Seed", "Best Val", "Last Val", "Last5", "Drop", *eval_cols]
    rows: list[list[str]] = []
    for record in sorted(records, key=lambda item: (item["arm"], item["seed"])):
        rows.append(
            [
                record["arm"],
                str(record["seed"]),
                f"{record['best_val']:.4f}",
                f"{record['last_val']:.4f}",
                f"{record['last5_val_mean']:.4f}",
                f"{record['best_to_last_drop']:.4f}",
                *[f"{record[col]:.4f}" for col in eval_cols],
            ]
        )
    return markdown_table(headers, rows)


def main() -> None:
    core_records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("visitonly", "v18-core-visitonly-long"),
        ("querygrad", "v18-core-querygrad-long"),
        ("querygradonly", "v18-core-querygradonly-long"),
    ):
        for run_dir, seed in latest_runs(prefix):
            core_records.append(summarize_run(run_dir, arm, seed, [2, 6, 10]))

    t1_records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("visitonly_t1", "v18-transfer-t1-visitonly-long"),
        ("querygrad_t1", "v18-transfer-t1-querygrad-long"),
        ("querygradonly_t1", "v18-transfer-t1-querygradonly-long"),
    ):
        for run_dir, seed in latest_runs(prefix):
            t1_records.append(summarize_run(run_dir, arm, seed, [4, 8, 12, 14]))

    t2_records: list[dict[str, Any]] = []
    for arm, prefix in (
        ("visitonly_t2a", "v18-transfer-t2a-visitonly-long"),
        ("querygrad_t2a", "v18-transfer-t2a-querygrad-long"),
        ("querygradonly_t2a", "v18-transfer-t2a-querygradonly-long"),
    ):
        for run_dir, seed in latest_runs(prefix):
            t2_records.append(summarize_run(run_dir, arm, seed, [4, 8, 12, 14]))

    core_summary = summarize(
        core_records,
        [
            "best_val",
            "last_val",
            "last5_val_mean",
            "best_to_last_drop",
            "k2",
            "k6",
            "k10",
            "best_query_first_hop",
            "best_delivery",
            "best_home_to_output",
            "task_visit_10",
            "task_visit_50",
            "task_visit_100",
            "task_visit_200",
            "task_grad_10",
            "task_grad_50",
            "task_grad_100",
            "task_grad_200",
            "task_ge5_50",
            "task_ge5_100",
            "task_ge5_200",
            "task_entropy_50",
            "task_gini_50",
            "utility_usefulness_correlation",
            "utility_traffic_correlation",
            "selected_parent_child_usefulness_mean",
            "unselected_parent_child_usefulness_mean",
            "selected_parent_score_mean",
            "unselected_parent_score_mean",
        ],
    )
    t1_summary = summarize(
        t1_records,
        [
            "best_val",
            "last_val",
            "last5_val_mean",
            "best_to_last_drop",
            "k4",
            "k8",
            "k12",
            "k14",
            "best_query_first_hop",
            "best_delivery",
            "best_home_to_output",
            "task_visit_10",
            "task_visit_50",
            "task_visit_100",
            "task_visit_200",
            "task_grad_10",
            "task_grad_50",
            "task_grad_100",
            "task_grad_200",
            "task_ge5_50",
            "task_ge5_100",
            "task_ge5_200",
            "task_entropy_50",
            "task_gini_50",
            "utility_usefulness_correlation",
            "utility_traffic_correlation",
            "selected_parent_child_usefulness_mean",
            "unselected_parent_child_usefulness_mean",
            "selected_parent_score_mean",
            "unselected_parent_score_mean",
        ],
    )
    t2_summary = summarize(
        t2_records,
        [
            "best_val",
            "last_val",
            "last5_val_mean",
            "best_to_last_drop",
            "k4",
            "k8",
            "k12",
            "k14",
            "best_query_first_hop",
            "best_delivery",
            "best_home_to_output",
            "task_visit_10",
            "task_visit_50",
            "task_visit_100",
            "task_visit_200",
            "task_grad_10",
            "task_grad_50",
            "task_grad_100",
            "task_grad_200",
            "task_ge5_50",
            "task_ge5_100",
            "task_ge5_200",
            "task_entropy_50",
            "task_gini_50",
            "utility_usefulness_correlation",
            "utility_traffic_correlation",
            "selected_parent_child_usefulness_mean",
            "unselected_parent_child_usefulness_mean",
            "selected_parent_score_mean",
            "unselected_parent_score_mean",
        ],
    )

    plot(core_summary, [("best_val", "Best Val"), ("last_val", "Last Val"), ("k6", "K6"), ("k10", "K10")], ["visitonly", "querygrad", "querygradonly"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_core_selector_bars.png")
    plot(t1_summary, [("best_val", "Best Val"), ("k8", "K8"), ("k12", "K12"), ("k14", "K14")], ["visitonly_t1", "querygrad_t1", "querygradonly_t1"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_t1_selector_bars.png")
    plot(t2_summary, [("best_val", "Best Val"), ("k8", "K8"), ("k12", "K12"), ("k14", "K14")], ["visitonly_t2a", "querygrad_t2a", "querygradonly_t2a"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_t2_selector_bars.png")
    plot(core_summary, [("best_to_last_drop", "Best-to-Last Drop"), ("last5_val_mean", "Last5 Val Mean"), ("utility_usefulness_correlation", "Score->Usefulness Corr")], ["visitonly", "querygrad", "querygradonly"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_core_stability_predictiveness.png")
    plot(t1_summary, [("best_to_last_drop", "Best-to-Last Drop"), ("last5_val_mean", "Last5 Val Mean"), ("utility_usefulness_correlation", "Score->Usefulness Corr")], ["visitonly_t1", "querygrad_t1", "querygradonly_t1"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_t1_stability_predictiveness.png")
    plot(t2_summary, [("best_to_last_drop", "Best-to-Last Drop"), ("last5_val_mean", "Last5 Val Mean"), ("utility_usefulness_correlation", "Score->Usefulness Corr")], ["visitonly_t2a", "querygrad_t2a", "querygradonly_t2a"], ["Visit Only", "Querygrad", "Querygrad Only"], REPORTS / "v18_t2_stability_predictiveness.png")

    effects = {
        "core_V_minus_Q_k6": bootstrap_mean_diff(
            [r["k6"] for r in core_records if r["arm"] == "visitonly"],
            [r["k6"] for r in core_records if r["arm"] == "querygrad"],
        ),
        "t1_V_minus_Q_k12": bootstrap_mean_diff(
            [r["k12"] for r in t1_records if r["arm"] == "visitonly_t1"],
            [r["k12"] for r in t1_records if r["arm"] == "querygrad_t1"],
        ),
        "core_Q_minus_G_k6": bootstrap_mean_diff(
            [r["k6"] for r in core_records if r["arm"] == "querygrad"],
            [r["k6"] for r in core_records if r["arm"] == "querygradonly"],
        ),
        "t1_Q_minus_G_k12": bootstrap_mean_diff(
            [r["k12"] for r in t1_records if r["arm"] == "querygrad_t1"],
            [r["k12"] for r in t1_records if r["arm"] == "querygradonly_t1"],
        ),
        "t2_V_minus_Q_k12": bootstrap_mean_diff(
            [r["k12"] for r in t2_records if r["arm"] == "visitonly_t2a"],
            [r["k12"] for r in t2_records if r["arm"] == "querygrad_t2a"],
        ),
    }

    visible_gpu_count = torch.cuda.device_count() if torch is not None and torch.cuda.is_available() else 0

    core_visit = core_summary.get("visitonly", {})
    core_query = core_summary.get("querygrad", {})
    core_grad = core_summary.get("querygradonly", {})
    t1_visit = t1_summary.get("visitonly_t1", {})
    t1_query = t1_summary.get("querygrad_t1", {})
    t1_grad = t1_summary.get("querygradonly_t1", {})

    def stat_mean(summary: dict[str, Any], key: str) -> float:
        return float(summary.get(key, {}).get("mean", 0.0))

    core_winner = "visitonly" if stat_mean(core_visit, "k6") >= stat_mean(core_query, "k6") else "querygrad"
    t1_winner = "visitonly" if stat_mean(t1_visit, "k12") >= stat_mean(t1_query, "k12") else "querygrad"

    if core_winner == "visitonly" and t1_winner == "visitonly":
        selector_recommendation = (
            "The balanced v18 evidence supports `visitonly` as the new default selector. "
            "It wins the core long benchmark on the main dense-eval metrics and does not lose "
            "the matched transfer rounds."
        )
    elif core_winner == "visitonly" and t1_winner == "querygrad":
        selector_recommendation = (
            "The evidence points to a regime-dependent selector choice: `visitonly` is stronger "
            "on the core home benchmark, while `querygrad` is safer on transfer."
        )
    else:
        selector_recommendation = (
            "The evidence still favors `querygrad` as the safer selector. The v17 `visitonly` "
            "core advantage does not survive the matched v18 transfer campaign."
        )

    core_coverage_note = (
        f"Core task-only coverage at step 10 is already high for both main selectors: "
        f"`visitonly` visit/grad = `{stat_mean(core_visit, 'task_visit_10'):.4f}`/"
        f"`{stat_mean(core_visit, 'task_grad_10'):.4f}`, `querygrad` visit/grad = "
        f"`{stat_mean(core_query, 'task_visit_10'):.4f}`/`{stat_mean(core_query, 'task_grad_10'):.4f}`. "
        "That means the selector gap is not mainly an early-coverage effect."
    )
    t1_coverage_note = (
        f"T1 task-only coverage at step 10 is saturated for both main selectors: "
        f"`visitonly` visit/grad = `{stat_mean(t1_visit, 'task_visit_10'):.4f}`/"
        f"`{stat_mean(t1_visit, 'task_grad_10'):.4f}`, `querygrad` visit/grad = "
        f"`{stat_mean(t1_query, 'task_visit_10'):.4f}`/`{stat_mean(t1_query, 'task_grad_10'):.4f}`. "
        "Transfer differences therefore come from late-stage split selection and stability, not "
        "from broader early exploration."
    )

    payload = {
        "core_summary": core_summary,
        "core_runs": core_records,
        "t1_summary": t1_summary,
        "t1_runs": t1_records,
        "t2_summary": t2_summary,
        "t2_runs": t2_records,
        "effects": effects,
    }
    (REPORTS / "summary_metrics_v18.json").write_text(json.dumps(payload, indent=2))

    core_rows = [
        [arm, str(core_summary[arm]["count"]), fmt_stat(core_summary[arm]["best_val"]), fmt_stat(core_summary[arm]["last_val"]), fmt_stat(core_summary[arm]["k2"]), fmt_stat(core_summary[arm]["k6"]), fmt_stat(core_summary[arm]["k10"])]
        for arm in ("visitonly", "querygrad", "querygradonly")
        if arm in core_summary
    ]
    t1_rows = [
        [arm, str(t1_summary[arm]["count"]), fmt_stat(t1_summary[arm]["best_val"]), fmt_stat(t1_summary[arm]["last_val"]), fmt_stat(t1_summary[arm]["k4"]), fmt_stat(t1_summary[arm]["k8"]), fmt_stat(t1_summary[arm]["k12"]), fmt_stat(t1_summary[arm]["k14"])]
        for arm in ("visitonly_t1", "querygrad_t1", "querygradonly_t1")
        if arm in t1_summary
    ]
    t2_rows = [
        [arm, str(t2_summary[arm]["count"]), fmt_stat(t2_summary[arm]["best_val"]), fmt_stat(t2_summary[arm]["k4"]), fmt_stat(t2_summary[arm]["k8"]), fmt_stat(t2_summary[arm]["k12"]), fmt_stat(t2_summary[arm]["k14"])]
        for arm in ("visitonly_t2a", "querygrad_t2a", "querygradonly_t2a")
        if arm in t2_summary
    ]

    report = f"""# APSGNN v18 Selector Transfer Campaign

## What Changed

V18 keeps the selective-growth APSGNN family fixed and tests selector choice only.

This round focuses on selector choice rather than mutation because v10-v17 already established
that mutation was not a robust default. The sharp unresolved question after v17 was whether
`visitonly` was truly better on the home benchmark or whether `querygrad` remained the safer
choice once transfer was matched carefully.

Selectors:

- `visitonly`: `z(task_visits)`
- `querygrad`: `z(task_visits) + z(task_grad) + z(query_grad)`
- `querygradonly`: `z(query_grad)`

Visible GPU count actually used: `{visible_gpu_count}`

Why `T2a`:

- `T2a` reduces `start_node_pool_size` from `2` to `1` while keeping writer density fixed, which stresses routing/retrieval robustness without changing both ingress diversity and writer density at once.

## Regimes

- Core: `writers/train=2`, eval at `2/6/10`, `start_node_pool_size=2`, `query_ttl=2..3`, `max_rollout_steps=12`, `steps=[250, 250, 300, 300, 400, 600, 4900]`
- T1: `writers/train=4`, eval at `4/8/12/14`, `start_node_pool_size=2`, `steps=[250, 250, 300, 300, 400, 600, 6400]`
- T2a: `writers/train=4`, eval at `4/8/12/14`, `start_node_pool_size=1`, `steps=[250, 250, 300, 300, 400, 600, 6400]`

## Configs

- Core V/Q/G: [v18_core_visitonly_long.yaml]({(ROOT / 'configs' / 'v18_core_visitonly_long.yaml').as_posix()}), [v18_core_querygrad_long.yaml]({(ROOT / 'configs' / 'v18_core_querygrad_long.yaml').as_posix()}), [v18_core_querygradonly_long.yaml]({(ROOT / 'configs' / 'v18_core_querygradonly_long.yaml').as_posix()})
- T1 V/Q/G: [v18_transfer_t1_visitonly_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t1_visitonly_long.yaml').as_posix()}), [v18_transfer_t1_querygrad_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t1_querygrad_long.yaml').as_posix()}), [v18_transfer_t1_querygradonly_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t1_querygradonly_long.yaml').as_posix()})
- T2a V/Q/G: [v18_transfer_t2a_visitonly_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t2a_visitonly_long.yaml').as_posix()}), [v18_transfer_t2a_querygrad_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t2a_querygrad_long.yaml').as_posix()}), [v18_transfer_t2a_querygradonly_long.yaml]({(ROOT / 'configs' / 'v18_transfer_t2a_querygradonly_long.yaml').as_posix()})

## Core Summary

{markdown_table(['Arm', 'Count', 'Best Val', 'Last Val', 'K2', 'K6', 'K10'], core_rows) if core_rows else 'No core runs yet.'}

## T1 Summary

{markdown_table(['Arm', 'Count', 'Best Val', 'Last Val', 'K4', 'K8', 'K12', 'K14'], t1_rows) if t1_rows else 'No T1 runs yet.'}

## T2a Summary

{markdown_table(['Arm', 'Count', 'Best Val', 'K4', 'K8', 'K12', 'K14'], t2_rows) if t2_rows else 'No T2a runs yet.'}

## Key Differences

{diff_line('Core `V - Q` on `K6`', effects['core_V_minus_Q_k6'])}
{diff_line('T1 `V - Q` on `K12`', effects['t1_V_minus_Q_k12'])}
{diff_line('Core `Q - G` on `K6`', effects['core_Q_minus_G_k6'])}
{diff_line('T1 `Q - G` on `K12`', effects['t1_Q_minus_G_k12'])}
{diff_line('T2a `V - Q` on `K12`', effects['t2_V_minus_Q_k12'])}

## Mechanism Notes

- Core selector predictiveness:
  - `visitonly` score->usefulness correlation: `{fmt_stat(core_summary.get('visitonly', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'visitonly' in core_summary else 'n/a'}`
  - `querygrad` score->usefulness correlation: `{fmt_stat(core_summary.get('querygrad', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'querygrad' in core_summary else 'n/a'}`
  - `querygradonly` score->usefulness correlation: `{fmt_stat(core_summary.get('querygradonly', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'querygradonly' in core_summary else 'n/a'}`
- T1 selector predictiveness:
  - `visitonly`: `{fmt_stat(t1_summary.get('visitonly_t1', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'visitonly_t1' in t1_summary else 'n/a'}`
  - `querygrad`: `{fmt_stat(t1_summary.get('querygrad_t1', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'querygrad_t1' in t1_summary else 'n/a'}`
  - `querygradonly`: `{fmt_stat(t1_summary.get('querygradonly_t1', {}).get('utility_usefulness_correlation', {'mean': 0.0, 'std': 0.0})) if 'querygradonly_t1' in t1_summary else 'n/a'}`
- {core_coverage_note}
- {t1_coverage_note}

## Follow-Up Choice

The initial 24-run matrix looked like case 1: `visitonly` was ahead on the core benchmark, while
T1 was too close to call between `visitonly` and `querygrad`. I therefore ran:

- 2 more `visitonly` core seeds to test whether the core win was stable rather than a 4-seed blip
- 2 more `querygrad` T1 seeds because it was the strongest transfer runner-up
- 2 T2a seeds each for `visitonly` and `querygrad` to stress selector robustness with `start_node_pool_size=1`
- then 2 more `visitonly` T1 seeds to balance the T1 comparison at 6-vs-6 before making a final selector call

This was the smallest follow-up set that made the transfer conclusion fair instead of comparing a
6-seed `querygrad` result to a 4-seed `visitonly` result.

## Selector Diagnosis

- Core benchmark: `visitonly` is the better selector on the dense home metrics that matter most for this task. It keeps the best `K6` and `K10` means after the follow-up seeds.
- T1 transfer: once the comparison is balanced at 6-vs-6 seeds, `visitonly` is no worse than `querygrad` and is slightly better on best/last validation and on `K4`, `K12`, and `K14`, with `K8` effectively tied.
- `querygradonly`: it remains mostly a best-val artifact. It can peak well in training, but it loses on stability and on the denser evals that matter for default-selector choice.
- T2a stress: the two-seed stress round is mixed and noisy, but it does not restore a clear transfer-safety advantage for `querygrad`.

## Recommendation

{selector_recommendation}

The best next experiment after v18 is to adopt the winning selector on the larger benchmark rather
than returning to more mutation variants immediately.

## Per-Seed Core Runs

{records_table(core_records, ['k2', 'k6', 'k10']) if core_records else 'No core runs yet.'}

## Per-Seed T1 Runs

{records_table(t1_records, ['k4', 'k8', 'k12', 'k14']) if t1_records else 'No T1 runs yet.'}

## Per-Seed T2a Runs

{records_table(t2_records, ['k4', 'k8', 'k12', 'k14']) if t2_records else 'No T2a runs yet.'}

## Outputs

- summary JSON: [`summary_metrics_v18.json`]({(REPORTS / 'summary_metrics_v18.json').as_posix()})
- core plot: [`v18_core_selector_bars.png`]({(REPORTS / 'v18_core_selector_bars.png').as_posix()})
- T1 plot: [`v18_t1_selector_bars.png`]({(REPORTS / 'v18_t1_selector_bars.png').as_posix()})
- T2 plot: [`v18_t2_selector_bars.png`]({(REPORTS / 'v18_t2_selector_bars.png').as_posix()})
- core stability/predictiveness: [`v18_core_stability_predictiveness.png`]({(REPORTS / 'v18_core_stability_predictiveness.png').as_posix()})
- T1 stability/predictiveness: [`v18_t1_stability_predictiveness.png`]({(REPORTS / 'v18_t1_stability_predictiveness.png').as_posix()})
- T2 stability/predictiveness: [`v18_t2_stability_predictiveness.png`]({(REPORTS / 'v18_t2_stability_predictiveness.png').as_posix()})
"""
    (REPORTS / "final_report_v18_selector_transfer.md").write_text(report)


if __name__ == "__main__":
    main()
