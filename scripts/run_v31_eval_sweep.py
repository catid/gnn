#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
SCHEDULE_TRAIN_STEPS = {"l": 3570, "xl": 4590}


def latest_runs(prefix: str) -> list[Path]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def is_complete_substantive_run(run_dir: Path, expected_train_steps: int) -> bool:
    config_path = run_dir / "config.yaml"
    metrics_path = run_dir / "metrics.jsonl"
    if not config_path.exists() or not metrics_path.exists() or not (run_dir / "last.pt").exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    actual_train_steps = int(config.get("train", {}).get("train_steps", 0))
    if actual_train_steps != expected_train_steps:
        return False
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= expected_train_steps


def eval_checkpoint(run_dir: Path, checkpoint: str, writers: int, batches: int) -> None:
    checkpoint_path = run_dir / f"{checkpoint}.pt"
    if not checkpoint_path.exists():
        return
    out_path = run_dir / f"eval_{checkpoint}_k{writers}.json"
    subprocess.run(
        [
            "python",
            "-m",
            "apsgnn.eval",
            "--config",
            str(run_dir / "config.yaml"),
            "--checkpoint",
            str(checkpoint_path),
            "--writers-per-episode",
            str(writers),
            "--batches",
            str(batches),
            "--output",
            str(out_path),
        ],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="+", required=True)
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--selectors", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    writer_map = {
        "core": [2, 6, 10],
        "t1": [4, 8, 12, 14],
        "t2a": [4, 8, 12, 14],
    }
    checkpoints = ["best"] if args.best_only else ["best", "last"]

    for regime in args.regimes:
        for schedule in args.schedules:
            expected_train_steps = SCHEDULE_TRAIN_STEPS[schedule]
            for selector in args.selectors:
                prefix = f"v31-{regime}-{selector}-32-{schedule}"
                for run_dir in latest_runs(prefix):
                    if not is_complete_substantive_run(run_dir, expected_train_steps):
                        continue
                    for checkpoint in checkpoints:
                        if checkpoint == "best" and not (run_dir / "best.pt").exists():
                            continue
                        for writers in writer_map[regime]:
                            eval_checkpoint(run_dir, checkpoint, writers, args.batches)


if __name__ == "__main__":
    main()
