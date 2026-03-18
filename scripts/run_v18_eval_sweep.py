#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


ARM_SPECS = {
    "V_core": {
        "prefix": "v18-core-visitonly-long",
        "config": "configs/v18_core_visitonly_long.yaml",
        "writers": [2, 6, 10],
    },
    "Q_core": {
        "prefix": "v18-core-querygrad-long",
        "config": "configs/v18_core_querygrad_long.yaml",
        "writers": [2, 6, 10],
    },
    "G_core": {
        "prefix": "v18-core-querygradonly-long",
        "config": "configs/v18_core_querygradonly_long.yaml",
        "writers": [2, 6, 10],
    },
    "V_t1": {
        "prefix": "v18-transfer-t1-visitonly-long",
        "config": "configs/v18_transfer_t1_visitonly_long.yaml",
        "writers": [4, 8, 12, 14],
    },
    "Q_t1": {
        "prefix": "v18-transfer-t1-querygrad-long",
        "config": "configs/v18_transfer_t1_querygrad_long.yaml",
        "writers": [4, 8, 12, 14],
    },
    "G_t1": {
        "prefix": "v18-transfer-t1-querygradonly-long",
        "config": "configs/v18_transfer_t1_querygradonly_long.yaml",
        "writers": [4, 8, 12, 14],
    },
    "V_t2": {
        "prefix": "v18-transfer-t2a-visitonly-long",
        "config": "configs/v18_transfer_t2a_visitonly_long.yaml",
        "writers": [4, 8, 12, 14],
    },
    "Q_t2": {
        "prefix": "v18-transfer-t2a-querygrad-long",
        "config": "configs/v18_transfer_t2a_querygrad_long.yaml",
        "writers": [4, 8, 12, 14],
    },
    "G_t2": {
        "prefix": "v18-transfer-t2a-querygradonly-long",
        "config": "configs/v18_transfer_t2a_querygradonly_long.yaml",
        "writers": [4, 8, 12, 14],
    },
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
    parser = argparse.ArgumentParser(description="Run eval sweep for v18 runs.")
    parser.add_argument("--arms", nargs="*", default=sorted(ARM_SPECS))
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    for arm in args.arms:
        spec = ARM_SPECS[arm]
        run_dirs = latest_runs(spec["prefix"])
        for run_dir in run_dirs:
            checkpoints = [("best", run_dir / "best.pt")]
            if not args.best_only:
                checkpoints.append(("last", run_dir / "last.pt"))
            for kind, checkpoint in checkpoints:
                if not checkpoint.exists():
                    continue
                for writers in spec["writers"]:
                    tag = f"{kind}_k{writers}"
                    output = run_dir / f"eval_{tag}.json"
                    if output.exists():
                        continue
                    run_eval(spec["config"], checkpoint, writers, args.batches, tag, output)


if __name__ == "__main__":
    main()
