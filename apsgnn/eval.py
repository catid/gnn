from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from apsgnn.config import dump_config, load_config
from apsgnn.ddp_utils import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from apsgnn.model import APSGNNModel
from apsgnn.tasks import MemoryRoutingTask, SanityRoutingTask
from apsgnn.utils import ensure_dir, environment_info, save_json, seed_everything


def is_first_hop_router_checkpoint_key(key: str) -> bool:
    return key.startswith("first_hop_router") or key.startswith("first_hop_router_ln")


def accumulate_metric_sums(
    accumulator: dict[str, torch.Tensor],
    metric_sums: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for key, value in metric_sums.items():
        value = value.detach().to(dtype=torch.float64)
        if key in accumulator:
            accumulator[key] = accumulator[key] + value
        else:
            accumulator[key] = value.clone()
    return accumulator


def reduce_metric_sums(metric_sums: dict[str, torch.Tensor], device: torch.device) -> dict[str, float]:
    if not metric_sums:
        return {}
    keys = sorted(metric_sums)
    stacked = torch.stack([metric_sums[key].to(device=device, dtype=torch.float64) for key in keys], dim=0)
    if is_distributed():
        torch.distributed.all_reduce(stacked)
    return {key: stacked[index].item() for index, key in enumerate(keys)}


def finalize_metrics(metric_sums: dict[str, float]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    sample_count = max(metric_sums.get("query_delivery_count", 0.0), 1.0)
    metrics["loss"] = metric_sums.get("loss_total", 0.0) / sample_count
    metrics["query_accuracy"] = (
        metric_sums.get("query_accuracy_hit", 0.0) / max(metric_sums.get("query_accuracy_count", 0.0), 1.0)
    )
    metrics["query_delivery_rate"] = metric_sums.get("query_delivery_hit", 0.0) / sample_count
    metrics["writer_first_hop_home_rate"] = (
        metric_sums.get("writer_home_hit", 0.0) / max(metric_sums.get("writer_home_count", 0.0), 1.0)
    )
    metrics["query_first_hop_home_rate"] = (
        metric_sums.get("query_home_hit", 0.0) / max(metric_sums.get("query_home_count", 0.0), 1.0)
    )
    metrics["query_home_to_output_rate"] = (
        metric_sums.get("query_home_output_hit", 0.0)
        / max(metric_sums.get("query_home_output_count", 0.0), 1.0)
    )
    metrics["first_hop_teacher_force_ratio"] = (
        metric_sums.get("first_hop_teacher_force_sum", 0.0)
        / max(metric_sums.get("first_hop_teacher_force_count", 0.0), 1.0)
    )
    metrics["average_hops"] = metric_sums.get("avg_hops_sum", 0.0) / max(metric_sums.get("avg_hops_count", 0.0), 1.0)
    metrics["average_delay"] = metric_sums.get("delay_sum", 0.0) / max(metric_sums.get("delay_count", 0.0), 1.0)
    if "cache_mean_sum" in metric_sums:
        metrics["cache_mean_occupancy"] = metric_sums.get("cache_mean_sum", 0.0) / max(
            metric_sums.get("cache_mean_count", 0.0),
            1.0,
        )
        metrics["cache_max_occupancy"] = metric_sums.get("cache_max_sum", 0.0) / max(
            metric_sums.get("cache_max_count", 0.0),
            1.0,
        )
    metrics["packets_processed"] = metric_sums.get("packets_processed_sum", 0.0)
    return metrics


@torch.no_grad()
def run_evaluation(
    model: APSGNNModel | DDP,
    config_path: str | Path,
    device: torch.device,
    batches: int,
    rank: int,
    writers_per_episode: int | None = None,
    desc: str = "eval",
) -> dict[str, float]:
    config = load_config(config_path)
    task_name = config.task.name
    if task_name == "memory":
        task = MemoryRoutingTask(config)
    else:
        task = SanityRoutingTask(config)

    was_training = model.training
    model.eval()
    unwrapped = model.module if isinstance(model, DDP) else model
    unwrapped.set_first_hop_teacher_force_ratio(0.0)
    metric_sums: dict[str, torch.Tensor] = {}
    iterator = range(batches)
    if is_main_process():
        iterator = tqdm(iterator, desc=desc, leave=False)

    for offset in iterator:
        seed = config.train.seed + 1_000_000 + rank * 100_000 + offset
        if task_name == "memory":
            batch = task.generate(
                batch_size=config.train.batch_size_per_gpu,
                seed=seed,
                writers_per_episode=writers_per_episode,
            ).to(device)
        else:
            batch = task.generate(batch_size=config.train.batch_size_per_gpu, seed=seed).to(device)
        output = model(batch)
        metric_sums = accumulate_metric_sums(metric_sums, output["metric_sums"])

    reduced = reduce_metric_sums(metric_sums, device=device)
    metrics = finalize_metrics(reduced)
    if was_training:
        model.train()
    return metrics


def _load_checkpoint(model: APSGNNModel, checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)
    disallowed_missing = [key for key in missing_keys if not is_first_hop_router_checkpoint_key(key)]
    disallowed_unexpected = [key for key in unexpected_keys if not is_first_hop_router_checkpoint_key(key)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}",
        )
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate APSGNN checkpoints.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate.")
    parser.add_argument("--writers-per-episode", type=int, default=None, help="Override writers_per_episode.")
    parser.add_argument("--batches", type=int, default=None, help="Override validation batch count.")
    parser.add_argument("--tag", default="eval", help="Output tag.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    rank, _, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    config = load_config(args.config)
    seed_everything(config.train.seed + rank)

    model = APSGNNModel(config).to(device)
    checkpoint = _load_checkpoint(model, args.checkpoint, device)
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)

    batches = args.batches or config.train.val_batches
    metrics = run_evaluation(
        model=model,
        config_path=args.config,
        device=device,
        batches=batches,
        rank=rank,
        writers_per_episode=args.writers_per_episode,
        desc=args.tag,
    )
    metrics["checkpoint_step"] = int(checkpoint.get("step", -1))
    metrics["writers_per_episode"] = args.writers_per_episode or config.task.writers_per_episode

    if is_main_process():
        output_path = Path(args.output) if args.output else ensure_dir(Path(args.checkpoint).parent) / f"{args.tag}.json"
        payload = {
            "config": config.to_dict(),
            "environment": environment_info(),
            "metrics": metrics,
        }
        save_json(payload, output_path)
        print(json.dumps(metrics, indent=2, sort_keys=True))

    cleanup_distributed()


if __name__ == "__main__":
    main()
