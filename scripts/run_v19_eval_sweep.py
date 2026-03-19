#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


ARM_SPECS = {
    "V_core": {
        "prefix": "v19-core-visitonly-scale",
        "config": "configs/v19_core_visitonly_scale.yaml",
        "writers": [4, 8, 12, 16],
    },
    "VT_core": {
        "prefix": "v19-core-visit-taskgrad-scale",
        "config": "configs/v19_core_visit_taskgrad_scale.yaml",
        "writers": [4, 8, 12, 16],
    },
    "VQ_core": {
        "prefix": "v19-core-visit-querygrad-scale",
        "config": "configs/v19_core_visit_querygrad_scale.yaml",
        "writers": [4, 8, 12, 16],
    },
    "VTQ_core": {
        "prefix": "v19-core-full-querygrad-scale",
        "config": "configs/v19_core_full_querygrad_scale.yaml",
        "writers": [4, 8, 12, 16],
    },
    "Q_core": {
        "prefix": "v19-core-querygradonly-scale",
        "config": "configs/v19_core_querygradonly_scale.yaml",
        "writers": [4, 8, 12, 16],
    },
    "V_t1": {
        "prefix": "v19-transfer-t1-visitonly-scale",
        "config": "configs/v19_transfer_t1_visitonly_scale.yaml",
        "writers": [6, 10, 14, 18],
    },
    "VT_t1": {
        "prefix": "v19-transfer-t1-visit-taskgrad-scale",
        "config": "configs/v19_transfer_t1_visit_taskgrad_scale.yaml",
        "writers": [6, 10, 14, 18],
    },
    "VQ_t1": {
        "prefix": "v19-transfer-t1-visit-querygrad-scale",
        "config": "configs/v19_transfer_t1_visit_querygrad_scale.yaml",
        "writers": [6, 10, 14, 18],
    },
    "VTQ_t1": {
        "prefix": "v19-transfer-t1-full-querygrad-scale",
        "config": "configs/v19_transfer_t1_full_querygrad_scale.yaml",
        "writers": [6, 10, 14, 18],
    },
    "Q_t1": {
        "prefix": "v19-transfer-t1-querygradonly-scale",
        "config": "configs/v19_transfer_t1_querygradonly_scale.yaml",
        "writers": [6, 10, 14, 18],
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
    parser = argparse.ArgumentParser(description="Run eval sweep for v19 runs.")
    parser.add_argument("--arms", nargs="*", default=sorted(ARM_SPECS))
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    args = parser.parse_args()

    for arm in args.arms:
        spec = ARM_SPECS[arm]
        for run_dir in latest_runs(spec["prefix"]):
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
