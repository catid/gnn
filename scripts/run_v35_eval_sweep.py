#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
    "t1r": [4, 8, 12, 14],
}

EXPECTED_TRAIN_STEPS = 4590


def latest_runs(prefix: str) -> list[Path]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def is_complete_substantive_run(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    if not metrics_path.exists() or not config_path.exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != EXPECTED_TRAIN_STEPS:
        return False
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= EXPECTED_TRAIN_STEPS


def run_eval(run_dir: Path, writers: int, kind: str, batches: int) -> None:
    checkpoint = run_dir / f"{kind}.pt"
    if not checkpoint.exists():
        return
    output = run_dir / f"eval_{kind}_k{writers}.json"
    subprocess.run(
        [
            "python",
            "-m",
            "apsgnn.eval",
            "--config",
            str(run_dir / "config.yaml"),
            "--checkpoint",
            str(checkpoint),
            "--writers-per-episode",
            str(writers),
            "--batches",
            str(batches),
            "--output",
            str(output),
        ],
        check=True,
        cwd=ROOT,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="+", required=True)
    parser.add_argument("--selectors", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    kinds = ["best"] if args.best_only else ["best", "last"]
    for regime in args.regimes:
        for selector in args.selectors:
            prefix = f"v35-{regime}-{selector}-32-xl"
            for run_dir in latest_runs(prefix):
                if not is_complete_substantive_run(run_dir):
                    continue
                for kind in kinds:
                    for writers in REGIME_WRITERS[regime]:
                        run_eval(run_dir, writers, kind, args.batches)


if __name__ == "__main__":
    main()
