from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from apsgnn.config import ExperimentConfig, dump_config, load_config
from apsgnn.ddp_utils import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from apsgnn.eval import accumulate_metric_sums, finalize_metrics, reduce_metric_sums, run_evaluation
from apsgnn.model import APSGNNModel
from apsgnn.tasks import MemoryRoutingTask, SanityRoutingTask
from apsgnn.utils import (
    MetricsWriter,
    count_parameters,
    environment_info,
    make_run_dir,
    plot_metrics,
    save_json,
    save_run_metadata,
    seed_everything,
)


def unwrap_model(model: APSGNNModel | DDP) -> APSGNNModel:
    return model.module if isinstance(model, DDP) else model


def first_hop_teacher_force_ratio(config_step: int, config: ExperimentConfig) -> float:
    anneal_steps = max(config.train.first_hop_teacher_force_anneal_steps, 1)
    if anneal_steps == 1:
        return float(config.train.first_hop_teacher_force_end)
    progress = min(max((config_step - 1) / (anneal_steps - 1), 0.0), 1.0)
    start = config.train.first_hop_teacher_force_start
    end = config.train.first_hop_teacher_force_end
    return float(start + (end - start) * progress)


def is_first_hop_router_checkpoint_key(key: str) -> bool:
    return key.startswith("first_hop_router") or key.startswith("first_hop_router_ln")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train APSGNN experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override runtime.run_name.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override train steps.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run warmup + throughput benchmark only.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from.")
    return parser


def create_run_dir(root: str, run_name: str) -> Path:
    if is_distributed():
        object_list = [str(make_run_dir(root, run_name)) if is_main_process() else None]
        torch.distributed.broadcast_object_list(object_list, src=0)
        return Path(object_list[0])
    return make_run_dir(root, run_name)


def save_checkpoint(
    run_dir: Path,
    model: APSGNNModel | DDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_metric: float,
    name: str,
) -> Path:
    unwrapped = model.module if isinstance(model, DDP) else model
    path = run_dir / f"{name}.pt"
    torch.save(
        {
            "step": step,
            "best_metric": best_metric,
            "model": unwrapped.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path


def maybe_load_checkpoint(
    model: APSGNNModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, float]:
    if checkpoint_path is None:
        return 0, -math.inf
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    disallowed_missing = [key for key in missing_keys if not is_first_hop_router_checkpoint_key(key)]
    disallowed_unexpected = [key for key in unexpected_keys if not is_first_hop_router_checkpoint_key(key)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}",
        )
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"]), float(checkpoint.get("best_metric", -math.inf))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    rank, _, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    config = load_config(args.config)
    if args.run_name is not None:
        config.runtime.run_name = args.run_name
    if args.train_steps is not None:
        config.train.train_steps = args.train_steps

    seed_everything(config.train.seed + rank)
    task_name = config.task.name
    task = MemoryRoutingTask(config) if task_name == "memory" else SanityRoutingTask(config)

    run_dir = create_run_dir(config.runtime.output_root, config.runtime.run_name)
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, run_dir / "config.yaml")
        save_run_metadata(run_dir, config)

    model = APSGNNModel(config).to(device)
    use_autocast = device.type == "cuda" and config.train.bf16 and torch.cuda.is_bf16_supported()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    start_step, best_metric = maybe_load_checkpoint(model, optimizer, args.checkpoint, device)

    if config.train.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]
    if is_distributed():
        model = DDP(model, device_ids=[local_rank], broadcast_buffers=False, find_unused_parameters=True)

    if is_main_process():
        save_json(
            {
                "parameters_trainable": count_parameters(model.module if isinstance(model, DDP) else model),
                "environment": environment_info(),
            },
            run_dir / "summary.json",
        )

    metrics_writer = MetricsWriter(run_dir) if is_main_process() else None
    interval_metric_sums: dict[str, torch.Tensor] = {}
    interval_start_time = time.perf_counter()
    progress = range(start_step + 1, config.train.train_steps + 1)
    if is_main_process() and not args.benchmark_only:
        progress = tqdm(progress, desc=config.runtime.run_name)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for step in progress:
        model.train()
        unwrap_model(model).set_first_hop_teacher_force_ratio(first_hop_teacher_force_ratio(step, config))
        optimizer.zero_grad(set_to_none=True)
        batch_seed = config.train.seed + rank * 100_000 + step
        if task_name == "memory":
            batch = task.generate(config.train.batch_size_per_gpu, batch_seed).to(device)
        else:
            batch = task.generate(config.train.batch_size_per_gpu, batch_seed).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            output = model(batch)
            loss = output["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()

        interval_metric_sums = accumulate_metric_sums(interval_metric_sums, output["metric_sums"])

        should_log = step % config.train.log_interval == 0 or step == config.train.train_steps
        should_eval = (
            (not args.benchmark_only)
            and (step % config.train.eval_interval == 0 or step == config.train.train_steps)
        )

        row: dict[str, float | int | None] | None = None
        if should_log:
            reduced = reduce_metric_sums(interval_metric_sums, device=device)
            train_metrics = finalize_metrics(reduced)
            elapsed = time.perf_counter() - interval_start_time
            global_packets = reduced.get("packets_processed_sum", 0.0)
            train_metrics["packets_per_second"] = global_packets / max(elapsed, 1.0e-6)
            if torch.cuda.is_available():
                memory_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                train_metrics["max_memory_gb"] = memory_gb
                torch.cuda.reset_peak_memory_stats(device)
            if is_main_process():
                row = {"step": step, "train/loss": train_metrics["loss"]}
                for key, value in train_metrics.items():
                    row[f"train/{key}"] = float(value)
            interval_metric_sums = {}
            interval_start_time = time.perf_counter()

        if should_eval:
            val_metrics = run_evaluation(
                model=model,
                config_path=args.config,
                device=device,
                batches=config.train.val_batches,
                rank=rank,
                writers_per_episode=config.task.writers_per_episode if task_name == "memory" else None,
                desc="val",
            )
            if is_main_process():
                row = row or {"step": step}
                row["val/loss"] = float(val_metrics["loss"])
                for key, value in val_metrics.items():
                    row[f"val/{key}"] = float(value)
                primary_metric = val_metrics["query_accuracy"] if task_name == "memory" else val_metrics["query_delivery_rate"]
                if primary_metric >= best_metric:
                    best_metric = primary_metric
                    best_path = save_checkpoint(run_dir, model, optimizer, step, best_metric, "best")
                    row["best_checkpoint"] = str(best_path)

        if is_main_process() and row is not None:
            metrics_writer.append(row)
            if not args.benchmark_only:
                print(row)

        if is_main_process() and not args.benchmark_only and (
            step % config.train.save_interval == 0 or step == config.train.train_steps
        ):
            save_checkpoint(run_dir, model, optimizer, step, best_metric, "last")

    if args.benchmark_only and is_main_process():
        csv_path = metrics_writer.flush_csv()
        plot_metrics(csv_path, run_dir / "benchmark")
        print(f"Benchmark run written to {run_dir}")
    elif is_main_process():
        csv_path = metrics_writer.flush_csv()
        plot_metrics(csv_path, run_dir / "training")
        save_checkpoint(run_dir, model, optimizer, config.train.train_steps, best_metric, "last")
        print(f"Training run written to {run_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
