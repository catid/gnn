#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "summary_metrics_v9.json"


ARM_TO_CONFIG = {
    "B": "configs/v9_staged_static_selective_long.yaml",
    "C": "configs/v9_clone_selective_long.yaml",
    "U": "configs/v9_utility_selective_long.yaml",
    "UM": "configs/v9_utility_mutate_long.yaml",
    "US": "configs/v9_utility_nosuccess_long.yaml",
    "UG": "configs/v9_utility_nograd_long.yaml",
    "U_followup": "configs/v9_utility_selective_longplus.yaml",
    "UM_followup": "configs/v9_utility_mutate_longplus.yaml",
    "US_followup": "configs/v9_utility_nosuccess_longplus.yaml",
    "UG_followup": "configs/v9_utility_nograd_longplus.yaml",
    "transfer_U": "configs/v9_transfer_h1_utility_longplus.yaml",
    "transfer_UM": "configs/v9_transfer_h1_utility_mutate_longplus.yaml",
}


ARM_TO_WRITERS = {
    "B": [2, 6, 10],
    "C": [2, 6, 10],
    "U": [2, 6, 10],
    "UM": [2, 6, 10],
    "US": [2, 6, 10],
    "UG": [2, 6, 10],
    "U_followup": [2, 6, 10],
    "UM_followup": [2, 6, 10],
    "US_followup": [2, 6, 10],
    "UG_followup": [2, 6, 10],
    "transfer_U": [4, 8, 12, 14],
    "transfer_UM": [4, 8, 12, 14],
}


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
    parser = argparse.ArgumentParser(description="Run eval sweep for v9 runs.")
    parser.add_argument("--summary", default=str(SUMMARY_PATH))
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--include-transfer", action="store_true")
    args = parser.parse_args()

    summary = json.loads(Path(args.summary).read_text())
    runs = summary["runs"]

    for run in runs:
        arm = run["arm"]
        if not args.include_transfer and arm.startswith("transfer_"):
            continue
        config = ARM_TO_CONFIG.get(arm)
        writers_list = ARM_TO_WRITERS.get(arm)
        if config is None or writers_list is None:
            continue
        run_dir = Path(run["run_dir"])
        checkpoints = [("best", run_dir / "best.pt")]
        if not args.best_only:
            checkpoints.append(("last", run_dir / "last.pt"))
        for kind, checkpoint in checkpoints:
            if not checkpoint.exists():
                continue
            for writers in writers_list:
                tag = f"{kind}_k{writers}"
                output = run_dir / f"eval_{tag}.json"
                if output.exists():
                    continue
                run_eval(config, checkpoint, writers, args.batches, tag, output)


if __name__ == "__main__":
    main()
