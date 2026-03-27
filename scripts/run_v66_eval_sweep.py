#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from apsgnn.config import load_config
from apsgnn.eval import run_evaluation
from apsgnn.growth import GrowthSchedule, GrowthTopology
from apsgnn.model import APSGNNModel
from apsgnn.tasks import GrowthMemoryRoutingTask, MemoryRoutingTask, SanityRoutingTask
from apsgnn.utils import save_json, seed_everything


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PREFIX = "v66"
EXPECTED_TRAIN_STEPS = {"p": 300, "m": 1350, "l": 2160}
RUN_RE = re.compile(
    r"v66-(?P<pack>collision|delay|followup)-(?P<regime>[^-]+)-(?P<condition>[^-]+)-(?P<pair>[a-z0-9_]+)-32-(?P<schedule>p|m|l)(?:-(?P<tag>(?!s\d+$)[^-]+))?-s(?P<seed>\d+)$"
)


def parse_run_name(name: str) -> dict[str, str] | None:
    match = RUN_RE.search(name)
    if match is None:
        return None
    return match.groupdict(default="")


def latest_runs(pack: str, regime: str, pair: str, schedule: str, condition: str | None = None) -> list[Path]:
    latest: dict[tuple[int, str], Path] = {}
    for candidate in sorted(RUNS.glob(f"*-{PREFIX}-*")):
        if not candidate.is_dir():
            continue
        meta = parse_run_name(candidate.name)
        if meta is None:
            continue
        if meta["pack"] != pack or meta["regime"] != regime or meta["pair"] != pair or meta["schedule"] != schedule:
            continue
        if condition is not None and meta["condition"] != condition:
            continue
        latest[(int(meta["seed"]), meta["tag"])] = candidate
    return [latest[key] for key in sorted(latest)]


def is_complete_run(run_dir: Path, schedule: str) -> bool:
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


def checkpoint_topology(checkpoint: dict[str, Any]) -> GrowthTopology | None:
    payload = checkpoint.get("growth_topology")
    return GrowthTopology.from_dict(payload) if payload is not None else None


def make_task(config):
    if config.task.name == "memory_growth":
        return GrowthMemoryRoutingTask(config)
    if config.task.name == "memory":
        return MemoryRoutingTask(config)
    return SanityRoutingTask(config)


def eval_suffix(*, bypass_mode: str) -> str:
    if bypass_mode and bypass_mode != "none":
        return f"_bypass-{bypass_mode}"
    return ""


def apply_eval_overrides(model: APSGNNModel, *, bypass_mode: str) -> None:
    if bypass_mode and bypass_mode != "none":
        model.config.model.cache_read_bypass_mode = bypass_mode
        if model.cache_retriever is not None:
            model.cache_retriever.bypass_mode = bypass_mode


@torch.no_grad()
def collect_probe_payload(
    *,
    run_dir: Path,
    kind: str,
    batches: int,
    bypass_mode: str,
) -> dict[str, Any]:
    config = load_config(run_dir / "config.yaml")
    checkpoint = torch.load(run_dir / f"{kind}.pt", map_location="cpu")
    topology = checkpoint_topology(checkpoint)
    growth_schedule = GrowthSchedule.from_config(config)
    eval_active_compute_nodes = topology.active_compute_nodes if topology is not None else growth_schedule.final_active_compute_nodes
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    seed_everything(config.train.seed)
    task = make_task(config)
    model = APSGNNModel(config).to(device)
    apply_eval_overrides(model, bypass_mode=bypass_mode)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Probe checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()
    model.set_first_hop_teacher_force_ratio(0.0)
    model.set_probe_collection(True)
    model.set_growth_context(
        active_compute_nodes=eval_active_compute_nodes,
        bootstrap_active=False,
        active_node_ids=None if topology is None else topology.active_node_tensor(),
        clockwise_successor_lookup=None if topology is None else topology.successor_lookup(),
    )

    collected: dict[str, list[torch.Tensor]] = {}
    for offset in range(batches):
        seed = config.train.seed + 2_000_000 + offset
        if config.task.name == "memory_growth":
            batch = task.generate(
                batch_size=config.train.batch_size_per_gpu,
                seed=seed,
                writers_per_episode=config.task.writers_per_episode,
                active_compute_nodes=eval_active_compute_nodes,
                bootstrap_mode=False,
                topology=topology,
            ).to(device)
        elif config.task.name == "memory":
            batch = task.generate(
                batch_size=config.train.batch_size_per_gpu,
                seed=seed,
                writers_per_episode=config.task.writers_per_episode,
            ).to(device)
        else:
            batch = task.generate(batch_size=config.train.batch_size_per_gpu, seed=seed).to(device)
        output = model(batch)
        probe_states = output.get("probe_states") or {}
        for key, value in probe_states.items():
            if isinstance(value, torch.Tensor):
                collected.setdefault(key, []).append(value.detach().cpu())

    payload = {key: torch.cat(values, dim=0) for key, values in collected.items() if values}
    if "final_sink_state" in payload and "labels" in payload and "delivered" in payload:
        delivered = payload["delivered"].to(torch.bool)
        correct = torch.zeros(payload["labels"].size(0), dtype=torch.float32)
        if delivered.any():
            logits = model._output_logits(payload["final_sink_state"][delivered].to(device)).detach().cpu()
            correct[delivered] = (logits.argmax(dim=-1) == payload["labels"][delivered]).to(torch.float32)
        payload["correct"] = correct
    return {key: value for key, value in payload.items()}


def run_eval(run_dir: Path, *, kind: str, batches: int, bypass_mode: str) -> None:
    config = load_config(run_dir / "config.yaml")
    checkpoint = torch.load(run_dir / f"{kind}.pt", map_location="cpu")
    topology = checkpoint_topology(checkpoint)
    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    model = APSGNNModel(config).to(device)
    apply_eval_overrides(model, bypass_mode=bypass_mode)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Eval checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    writers_list = list(config.task.train_eval_writers or [config.task.writers_per_episode])
    suffix = eval_suffix(bypass_mode=bypass_mode)
    for writers in writers_list:
        output_path = run_dir / f"eval_{kind}_k{writers}{suffix}.json"
        if output_path.exists():
            continue
        metrics = run_evaluation(
            model=model,
            config_path=run_dir / "config.yaml",
            device=device,
            batches=batches,
            rank=0,
            writers_per_episode=writers,
            topology=topology,
            desc=f"{run_dir.name}:{kind}:k{writers}{suffix}",
        )
        save_json({"metrics": metrics}, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v66 eval sweeps and probe collection.")
    parser.add_argument("--pack", choices=["collision", "delay", "followup"], required=True)
    parser.add_argument("--regimes", nargs="+", required=True)
    parser.add_argument("--pairs", nargs="+", required=True)
    parser.add_argument("--schedules", nargs="+", required=True)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--batches", type=int, default=96)
    parser.add_argument("--collect-probes", action="store_true")
    parser.add_argument("--override-cache-bypass-mode", default="none")
    args = parser.parse_args()

    for regime in args.regimes:
        for pair in args.pairs:
            for schedule in args.schedules:
                conditions = args.conditions or [None]
                for condition in conditions:
                    for run_dir in latest_runs(args.pack, regime, pair, schedule, condition):
                        if not is_complete_run(run_dir, schedule):
                            continue
                        for kind in ("best", "last"):
                            checkpoint = run_dir / f"{kind}.pt"
                            if not checkpoint.exists():
                                continue
                            run_eval(
                                run_dir,
                                kind=kind,
                                batches=args.batches,
                                bypass_mode=args.override_cache_bypass_mode,
                            )
                            if args.collect_probes:
                                suffix = eval_suffix(bypass_mode=args.override_cache_bypass_mode)
                                probe_path = run_dir / f"probe_{kind}{suffix}.pt"
                                if not probe_path.exists():
                                    torch.save(
                                        collect_probe_payload(
                                            run_dir=run_dir,
                                            kind=kind,
                                            batches=args.batches,
                                            bypass_mode=args.override_cache_bypass_mode,
                                        ),
                                        probe_path,
                                    )


if __name__ == "__main__":
    main()
