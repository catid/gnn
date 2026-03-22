#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

REGIME_WRITERS = {
    "core": [2, 6, 10],
    "t1": [4, 8, 12, 14],
    "t2a": [4, 8, 12, 14],
}


def latest_runs(prefix: str) -> list[Path]:
    latest_by_seed: dict[int, Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{prefix}-s*")):
        seed = int(candidate.name.rsplit("-s", 1)[1])
        latest_by_seed[seed] = candidate
    return [latest_by_seed[seed] for seed in sorted(latest_by_seed)]


def run_eval(run_dir: Path, writers: int, kind: str, batches: int) -> None:
    checkpoint = run_dir / f"{kind}.pt"
    if not checkpoint.exists():
        return
    output = run_dir / f"eval_{kind}_k{writers}.json"
    cmd = [
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
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="+", required=True)
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--selectors", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    kinds = ["best"] if args.best_only else ["best", "last"]
    for regime in args.regimes:
        for schedule in args.schedules:
            for selector in args.selectors:
                prefix = f"v27-{regime}-{selector}-32-{schedule}"
                for run_dir in latest_runs(prefix):
                    for kind in kinds:
                        for writers in REGIME_WRITERS[regime]:
                            run_eval(run_dir, writers, kind, args.batches)


if __name__ == "__main__":
    main()
