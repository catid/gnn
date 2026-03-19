#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"

SELECTORS = [
    "visitonly",
    "visit_taskgrad",
    "visit_querygrad",
    "full_querygrad",
    "querygradonly",
    "visit_taskgrad_half",
    "visit_querygrad_half",
]

WRITERS_BY_REGIME = {
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


def run_eval(config: str, checkpoint: Path, writers: int, batches: int, tag: str, output: Path) -> None:
    cmd = [
        "python",
        "-m",
        "apsgnn.eval",
        "--config",
        config,
        "--checkpoint",
        str(checkpoint),
        "--writers-per-episode",
        str(writers),
        "--batches",
        str(batches),
        "--tag",
        tag,
        "--output",
        str(output),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval sweeps for v21 selector funnel runs.")
    parser.add_argument("--regimes", nargs="+", choices=sorted(WRITERS_BY_REGIME), required=True)
    parser.add_argument("--schedules", nargs="+", choices=["s", "m", "l"], required=True)
    parser.add_argument("--selectors", nargs="+", default=SELECTORS, choices=SELECTORS)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    for regime in args.regimes:
        for schedule in args.schedules:
            if regime == "t2a" and schedule != "l":
                continue
            for selector in args.selectors:
                prefix = f"v21-{regime}-{selector}-32-{schedule}"
                config = f"configs/v21_{regime}_{selector}_32_{schedule}.yaml"
                for run_dir in latest_runs(prefix):
                    checkpoints = [("best", run_dir / "best.pt")]
                    if not args.best_only:
                        checkpoints.append(("last", run_dir / "last.pt"))
                    for kind, checkpoint in checkpoints:
                        if not checkpoint.exists():
                            continue
                        for writers in WRITERS_BY_REGIME[regime]:
                            tag = f"{kind}_k{writers}"
                            output = run_dir / f"eval_{tag}.json"
                            if output.exists():
                                continue
                            run_eval(config, checkpoint, writers, args.batches, tag, output)


if __name__ == "__main__":
    main()
