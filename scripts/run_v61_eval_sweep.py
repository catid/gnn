#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

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

EXPECTED_TRAIN_STEPS = {"p": 420, "s": 1260, "m": 2520, "l": 3360}

RUN_RE = re.compile(
    r"v61-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def latest_runs(regime: str, arm: str, schedule: str) -> list[Path]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob("*-v61-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["regime"] != regime or meta["arm"] != arm or meta["schedule"] != schedule:
            continue
        latest_by_seed[int(meta["seed"])] = candidate
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def is_complete_substantive_run(run_dir: Path, schedule: str) -> bool:
    metrics_path = run_dir / "metrics.jsonl"
    config_path = run_dir / "config.yaml"
    last_path = run_dir / "last.pt"
    if not metrics_path.exists() or not config_path.exists() or not last_path.exists():
        return False
    config = yaml.safe_load(config_path.read_text())
    if int(config.get("train", {}).get("train_steps", 0)) != EXPECTED_TRAIN_STEPS[schedule]:
        return False
    max_step = 0
    for line in metrics_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        max_step = max(max_step, int(row.get("step", 0)))
    return max_step >= EXPECTED_TRAIN_STEPS[schedule]


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
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=96)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    kinds = ["best"] if args.best_only else ["best", "last"]
    for regime in args.regimes:
        for schedule in args.schedules:
            for arm in args.arms:
                for run_dir in latest_runs(regime, arm, schedule):
                    if not is_complete_substantive_run(run_dir, schedule):
                        continue
                    for kind in kinds:
                        for writers in REGIME_WRITERS[regime]:
                            run_eval(run_dir, writers, kind, args.batches)


if __name__ == "__main__":
    main()
