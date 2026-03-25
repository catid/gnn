#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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

EXPECTED_TRAIN_STEPS = {"p": 300, "s": 810, "m": 1620, "l": 2268}

RUN_RE = re.compile(
    r"v63ee-(?P<regime>[^-]+)-(?P<arm>[a-z0-9_]+)-32-(?P<schedule>p|s|m|l)(?:-(?P<tag>[^-]+))?-s(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def latest_runs(regime: str, arm: str, schedule: str, tags: set[str] | None = None) -> list[Path]:
    latest_by_seed: dict[tuple[int, str], Path] = {}
    for candidate in sorted(RUNS.glob("*-v63ee-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["regime"] != regime or meta["arm"] != arm or meta["schedule"] != schedule:
            continue
        if tags is not None and meta["tag"] not in tags:
            continue
        latest_by_seed[(int(meta["seed"]), meta["tag"])] = candidate
    return [latest_by_seed[key] for key in sorted(latest_by_seed)]


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
        if line.strip():
            max_step = max(max_step, int(json.loads(line).get("step", 0)))
    return max_step >= EXPECTED_TRAIN_STEPS[schedule]


def run_eval(
    run_dir: Path,
    *,
    writers: int,
    kind: str,
    batches: int,
    rollout_steps: int | None,
    tag: str | None,
    start_pool: int | None = None,
    ttl_min: int | None = None,
    ttl_max: int | None = None,
) -> None:
    checkpoint = run_dir / f"{kind}.pt"
    if not checkpoint.exists():
        return
    suffix = f"_{tag}" if tag else ""
    output = run_dir / f"eval_{kind}{suffix}_k{writers}.json"
    if output.exists():
        return
    cmd = [
        "python3",
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
        "--tag",
        tag or kind,
    ]
    if rollout_steps is not None:
        cmd.extend(["--rollout-steps", str(rollout_steps)])
    if start_pool is not None:
        cmd.extend(["--start-node-pool-size", str(start_pool)])
    if ttl_min is not None:
        cmd.extend(["--query-ttl-min", str(ttl_min)])
    if ttl_max is not None:
        cmd.extend(["--query-ttl-max", str(ttl_max)])
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regimes", nargs="+", required=True)
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--batches", type=int, default=96)
    parser.add_argument("--best-only", action="store_true")
    parser.add_argument("--extra-depth", action="store_true")
    parser.add_argument("--settle-cap", type=int, default=30)
    parser.add_argument("--tags", nargs="*", default=None)
    parser.add_argument("--surface", action="store_true")
    parser.add_argument("--surface-writers", nargs="*", type=int, default=[2, 3, 4, 5, 6])
    parser.add_argument("--surface-start-pools", nargs="*", type=int, default=[1, 2])
    parser.add_argument("--surface-ttl-pairs", nargs="*", default=["2,2", "2,3"])
    args = parser.parse_args()

    kinds = ["best"] if args.best_only else ["best", "last"]
    tag_set = None if args.tags is None else set(args.tags)

    for regime in args.regimes:
        for schedule in args.schedules:
            for arm in args.arms:
                for run_dir in latest_runs(regime, arm, schedule, tag_set):
                    if not is_complete_substantive_run(run_dir, schedule):
                        continue
                    config = yaml.safe_load((run_dir / "config.yaml").read_text())
                    base_depth = int(config.get("task", {}).get("max_rollout_steps", 0))
                    depth_specs = [(None, None)]
                    if args.extra_depth:
                        depth_specs = [
                            ("standard", base_depth),
                            ("plus50", int(math.ceil(base_depth * 1.5))),
                            ("plus100", int(math.ceil(base_depth * 2.0))),
                            ("settle", int(args.settle_cap)),
                        ]
                    for kind in kinds:
                        for writers in REGIME_WRITERS[regime]:
                            for tag, rollout_steps in depth_specs:
                                run_eval(
                                    run_dir,
                                    writers=writers,
                                    kind=kind,
                                    batches=args.batches,
                                    rollout_steps=rollout_steps,
                                    tag=tag,
                                )
                        if args.surface:
                            for writers in args.surface_writers:
                                for start_pool in args.surface_start_pools:
                                    for ttl_pair in args.surface_ttl_pairs:
                                        ttl_min_str, ttl_max_str = ttl_pair.split(",", 1)
                                        ttl_min = int(ttl_min_str)
                                        ttl_max = int(ttl_max_str)
                                        surface_tag = f"surface_w{writers}_p{start_pool}_t{ttl_min}{ttl_max}"
                                        run_eval(
                                            run_dir,
                                            writers=writers,
                                            kind=kind,
                                            batches=args.batches,
                                            rollout_steps=None,
                                            tag=surface_tag,
                                            start_pool=start_pool,
                                            ttl_min=ttl_min,
                                            ttl_max=ttl_max,
                                        )


if __name__ == "__main__":
    main()
