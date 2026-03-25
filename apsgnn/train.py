from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from apsgnn.config import ExperimentConfig, dump_config, load_config, selector_decision_for_stage
from apsgnn.ddp_utils import cleanup_distributed, is_distributed, is_main_process, setup_distributed
from apsgnn.eval import accumulate_metric_sums, finalize_metrics, reduce_metric_sums, run_evaluation
from apsgnn.growth import (
    CoverageTracker,
    GrowthSchedule,
    GrowthTopology,
    build_initial_topology,
    mutation_stagnation_info,
    transition_model_for_growth,
    transition_topology_for_growth,
)
from apsgnn.model import APSGNNModel
from apsgnn.tasks import GrowthMemoryRoutingTask, MemoryRoutingTask, SanityRoutingTask
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


def training_rollout_steps(config_step: int, config: ExperimentConfig) -> int:
    full_steps = int(config.task.max_rollout_steps)
    shallow_steps = int(config.train.contract_shallow_rollout_steps)
    shallow_fraction = float(config.train.contract_shallow_train_fraction)
    rand_fraction = float(getattr(config.train, "contract_rand_depth_train_fraction", 0.0))
    rand_multipliers = [float(value) for value in getattr(config.train, "contract_rand_depth_multipliers", [])]
    if rand_multipliers and rand_fraction > 0.0:
        cutoff = max(int(config.train.train_steps * rand_fraction), 1)
        if int(config_step) <= cutoff:
            choice_index = (
                int(config.train.seed) * 1_315_423_911 + int(config_step) * 2_654_435_761
            ) % len(rand_multipliers)
            multiplier = rand_multipliers[choice_index]
            return max(1, int(round(full_steps * multiplier)))
    if shallow_steps <= 0 or shallow_fraction <= 0.0:
        return full_steps
    cutoff = max(int(config.train.train_steps * shallow_fraction), 1)
    if int(config_step) <= cutoff:
        return max(1, min(shallow_steps, full_steps))
    return full_steps


def is_first_hop_router_checkpoint_key(key: str) -> bool:
    return key.startswith("first_hop_router") or key.startswith("first_hop_router_ln")


def is_cache_retriever_checkpoint_key(key: str) -> bool:
    return key.startswith("cache_retriever")


def load_model_weights(
    model: APSGNNModel,
    checkpoint_path: str,
    device: torch.device,
    *,
    allow_cache_retriever_mismatch: bool,
) -> tuple[dict[str, object], list[str], list[str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint["model"]
    missing_keys, unexpected_keys = model.load_state_dict(state, strict=False)

    def allowed(key: str) -> bool:
        if is_first_hop_router_checkpoint_key(key):
            return True
        if allow_cache_retriever_mismatch and is_cache_retriever_checkpoint_key(key):
            return True
        return False

    disallowed_missing = [key for key in missing_keys if not allowed(key)]
    disallowed_unexpected = [key for key in unexpected_keys if not allowed(key)]
    if disallowed_missing or disallowed_unexpected:
        raise RuntimeError(
            f"Checkpoint load mismatch: missing={missing_keys}, unexpected={unexpected_keys}",
        )
    return checkpoint, missing_keys, unexpected_keys


def maybe_initialize_from_checkpoint(
    model: APSGNNModel,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if checkpoint_path is None or checkpoint_path == "":
        return
    load_model_weights(
        model,
        checkpoint_path,
        device,
        allow_cache_retriever_mismatch=True,
    )


def freeze_first_hop_router(model: APSGNNModel) -> None:
    for module_name in ("first_hop_router", "first_hop_router_ln"):
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train APSGNN experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--run-name", default=None, help="Override runtime.run_name.")
    parser.add_argument("--train-steps", type=int, default=None, help="Override train steps.")
    parser.add_argument("--seed", type=int, default=None, help="Override train seed.")
    parser.add_argument("--lr-scale", type=float, default=None, help="Optional learning-rate multiplier.")
    parser.add_argument("--benchmark-only", action="store_true", help="Run warmup + throughput benchmark only.")
    parser.add_argument("--checkpoint", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--init-checkpoint", default=None, help="Optional model-only warm-start checkpoint.")
    parser.add_argument(
        "--contract-penultimate-keep-prob",
        type=float,
        default=None,
        help="Optional override for stochastic penultimate temporal-credit keep probability.",
    )
    parser.add_argument(
        "--late-stage-stability-weight",
        type=float,
        default=None,
        help="Optional override for the late-stage route-stability penalty weight.",
    )
    parser.add_argument(
        "--contract-aux-anneal-final-multiplier",
        type=float,
        default=None,
        help="Optional override for the final routing-auxiliary anneal multiplier.",
    )
    parser.add_argument(
        "--contract-aux-anneal-start-fraction",
        type=float,
        default=None,
        help="Optional override for the fraction of training after which routing auxiliaries begin annealing.",
    )
    parser.add_argument(
        "--contract-rand-depth-train-fraction",
        type=float,
        default=None,
        help="Optional override for the fraction of training that uses randomized rollout depth.",
    )
    parser.add_argument(
        "--contract-rand-depth-multipliers",
        default=None,
        help="Optional comma-separated override for randomized rollout-depth multipliers.",
    )
    parser.add_argument(
        "--slow-commit-interval",
        type=int,
        default=None,
        help="Optional override for delayed cache commit interval. <=1 disables delayed commit.",
    )
    parser.add_argument(
        "--selector-gate-online-stage-index-min",
        type=int,
        default=None,
        help="Optional override for the minimum stage index where the online gate may switch selectors.",
    )
    parser.add_argument(
        "--selector-gate-online-entropy-high-threshold",
        type=float,
        default=None,
        help="Optional override for the online-gate entropy threshold.",
    )
    parser.add_argument(
        "--selector-gate-online-gini-high-threshold",
        type=float,
        default=None,
        help="Optional override for the online-gate Gini threshold.",
    )
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
    *,
    growth_topology: GrowthTopology | None = None,
) -> Path:
    unwrapped = model.module if isinstance(model, DDP) else model
    path = run_dir / f"{name}.pt"
    payload = {
        "step": step,
        "best_metric": best_metric,
        "model": unwrapped.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    if growth_topology is not None:
        payload["growth_topology"] = growth_topology.to_dict()
    torch.save(payload, path)
    return path


def maybe_load_checkpoint(
    model: APSGNNModel,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | None,
    device: torch.device,
) -> tuple[int, float, GrowthTopology | None]:
    if checkpoint_path is None:
        return 0, -math.inf, None
    checkpoint, _, _ = load_model_weights(
        model,
        checkpoint_path,
        device,
        allow_cache_retriever_mismatch=False,
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    topology_payload = checkpoint.get("growth_topology")
    topology = GrowthTopology.from_dict(topology_payload) if topology_payload is not None else None
    return int(checkpoint["step"]), float(checkpoint.get("best_metric", -math.inf)), topology


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
    if args.seed is not None:
        config.train.seed = args.seed
    if args.lr_scale is not None:
        config.train.lr = float(config.train.lr) * float(args.lr_scale)
    if args.init_checkpoint is not None:
        config.train.init_checkpoint = args.init_checkpoint
    if args.contract_penultimate_keep_prob is not None:
        config.train.contract_penultimate_keep_prob = float(args.contract_penultimate_keep_prob)
    if args.late_stage_stability_weight is not None:
        config.train.late_stage_stability_weight = float(args.late_stage_stability_weight)
    if args.contract_aux_anneal_final_multiplier is not None:
        config.train.contract_aux_anneal_final_multiplier = float(args.contract_aux_anneal_final_multiplier)
    if args.contract_aux_anneal_start_fraction is not None:
        config.train.contract_aux_anneal_start_fraction = float(args.contract_aux_anneal_start_fraction)
    if args.contract_rand_depth_train_fraction is not None:
        config.train.contract_rand_depth_train_fraction = float(args.contract_rand_depth_train_fraction)
    if args.contract_rand_depth_multipliers is not None:
        config.train.contract_rand_depth_multipliers = [
            float(part.strip())
            for part in str(args.contract_rand_depth_multipliers).split(",")
            if part.strip()
        ]
    if args.slow_commit_interval is not None:
        config.train.slow_commit_interval = int(args.slow_commit_interval)
    if args.selector_gate_online_stage_index_min is not None:
        config.growth.selector_gate_online_stage_index_min = int(args.selector_gate_online_stage_index_min)
    if args.selector_gate_online_entropy_high_threshold is not None:
        config.growth.selector_gate_online_entropy_high_threshold = float(args.selector_gate_online_entropy_high_threshold)
    if args.selector_gate_online_gini_high_threshold is not None:
        config.growth.selector_gate_online_gini_high_threshold = float(args.selector_gate_online_gini_high_threshold)

    seed_everything(config.train.seed + rank)
    task_name = config.task.name
    if task_name == "memory_growth":
        task = GrowthMemoryRoutingTask(config)
    elif task_name == "memory":
        task = MemoryRoutingTask(config)
    else:
        task = SanityRoutingTask(config)
    growth_schedule = GrowthSchedule.from_config(config)

    run_dir = create_run_dir(config.runtime.output_root, config.runtime.run_name)
    if is_main_process():
        run_dir.mkdir(parents=True, exist_ok=True)
        dump_config(config, run_dir / "config.yaml")
        save_run_metadata(run_dir, config)

    model = APSGNNModel(config).to(device)
    if args.checkpoint is None:
        maybe_initialize_from_checkpoint(model, config.train.init_checkpoint or None, device)
    if config.train.freeze_first_hop_router:
        freeze_first_hop_router(model)
    use_autocast = device.type == "cuda" and config.train.bf16 and torch.cuda.is_bf16_supported()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.train.lr,
        weight_decay=config.train.weight_decay,
    )
    start_step, best_metric, resumed_topology = maybe_load_checkpoint(model, optimizer, args.checkpoint, device)

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
    coverage_tracker = (
        CoverageTracker(
            num_compute_nodes=config.model.num_compute_nodes,
            gradient_norm_threshold=config.growth.gradient_norm_threshold,
            utility_ema_decay=config.growth.utility_ema_decay,
            utility_tail_fraction=config.growth.utility_tail_fraction,
        )
        if config.growth.enabled
        else None
    )
    selective_topology = config.growth.enabled and config.growth.topology_mode == "selective"
    topology: GrowthTopology | None = resumed_topology
    if selective_topology and topology is None:
        if start_step > 0:
            raise RuntimeError("Selective growth resume requires checkpoint topology metadata.")
        topology = build_initial_topology(config, growth_schedule.stages[0].active_compute_nodes)
    interval_metric_sums: dict[str, torch.Tensor] = {}
    interval_start_time = time.perf_counter()
    current_stage = None
    stage_val_history: dict[int, list[float]] = {}
    progress = range(start_step + 1, config.train.train_steps + 1)
    if is_main_process() and not args.benchmark_only:
        progress = tqdm(progress, desc=config.runtime.run_name)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    for step in progress:
        stage = growth_schedule.stage_for_step(step)
        if current_stage is None or stage.index != current_stage.index:
            split_stats = None
            if current_stage is not None and stage.active_compute_nodes > current_stage.active_compute_nodes:
                stage_transition_stats = mutation_stagnation_info(
                    stage_val_history.get(current_stage.index, []),
                    window=config.growth.mutation_stagnation_window,
                    delta=config.growth.mutation_stagnation_delta,
                )
                selector_snapshot = coverage_tracker.current_snapshot() if coverage_tracker is not None else {}
                selector_decision = selector_decision_for_stage(
                    config.growth,
                    stage.index,
                    task=config.task,
                    current_snapshot=selector_snapshot,
                )
                selector_weights = selector_decision["weights"]
                if selective_topology:
                    assert topology is not None
                    utility_components = (
                        coverage_tracker.selection_components(
                            topology,
                            utility_alpha=config.growth.utility_success_alpha,
                            utility_visit_weight=selector_weights["utility_visit_weight"],
                            utility_grad_weight=selector_weights["utility_grad_weight"],
                            utility_query_visit_weight=selector_weights["utility_query_visit_weight"],
                            utility_query_grad_weight=selector_weights["utility_query_grad_weight"],
                        )
                        if coverage_tracker is not None
                        else {}
                    )
                    topology, topology_stats = transition_topology_for_growth(
                        topology,
                        stage.active_compute_nodes,
                        split_parent_policy=config.growth.split_parent_policy,
                        utility_components=utility_components,
                        utility_alpha=config.growth.utility_success_alpha,
                        utility_visit_weight=selector_weights["utility_visit_weight"],
                        utility_grad_weight=selector_weights["utility_grad_weight"],
                        utility_query_visit_weight=selector_weights["utility_query_visit_weight"],
                        utility_query_grad_weight=selector_weights["utility_query_grad_weight"],
                        seed=config.train.seed + stage.index,
                        future_active_counts=[
                            future_stage.active_compute_nodes
                            for future_stage in growth_schedule.stages[stage.index + 1 :]
                        ],
                    )
                    topology_stats.update(
                        {
                            "selector_stage_index": int(stage.index),
                            "selector_weights": selector_weights,
                            "selector_decision": selector_decision,
                            "selector_snapshot": selector_snapshot,
                            "stage_stagnated": stage_transition_stats["stagnated"],
                            "stage_stagnation_window": stage_transition_stats["window"],
                            "stage_stagnation_delta": stage_transition_stats["delta"],
                            "stage_val_tail": stage_transition_stats["tail_values"],
                            "stage_val_range": stage_transition_stats["range"],
                        }
                    )
                    split_stats = transition_model_for_growth(
                        unwrap_model(model),
                        current_stage.active_compute_nodes,
                        stage.active_compute_nodes,
                        transition_mode=config.growth.transition_mode,
                        split_mode=config.growth.split_mode,
                        mutation_scale=config.growth.split_mutation_scale,
                        seed=config.train.seed + stage.index,
                        selective_parent_child_pairs=topology_stats.get("sibling_pairs", []),
                        transition_stats=topology_stats,
                        next_stage_index=stage.index,
                        mutation_stage_index_min=config.growth.mutation_stage_index_min,
                        mutation_selected_fraction=config.growth.mutation_selected_fraction,
                        mutation_score_margin=config.growth.mutation_score_margin,
                        mutation_min_visit_z=config.growth.mutation_min_visit_z,
                        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
                        mutation_require_stagnation=config.growth.mutation_require_stagnation,
                    )
                else:
                    split_stats = transition_model_for_growth(
                        unwrap_model(model),
                        current_stage.active_compute_nodes,
                        stage.active_compute_nodes,
                        transition_mode=config.growth.transition_mode,
                        split_mode=config.growth.split_mode,
                        mutation_scale=config.growth.split_mutation_scale,
                        seed=config.train.seed + stage.index,
                        next_stage_index=stage.index,
                        mutation_stage_index_min=config.growth.mutation_stage_index_min,
                        mutation_selected_fraction=config.growth.mutation_selected_fraction,
                        mutation_score_margin=config.growth.mutation_score_margin,
                        mutation_min_visit_z=config.growth.mutation_min_visit_z,
                        mutation_min_query_grad_z=config.growth.mutation_min_query_grad_z,
                        mutation_require_stagnation=config.growth.mutation_require_stagnation,
                        transition_stats={
                            "selector_stage_index": int(stage.index),
                            "selector_weights": selector_weights,
                            "selector_decision": selector_decision,
                            "selector_snapshot": selector_snapshot,
                            "stage_stagnated": stage_transition_stats["stagnated"],
                            "stage_stagnation_window": stage_transition_stats["window"],
                            "stage_stagnation_delta": stage_transition_stats["delta"],
                            "stage_val_tail": stage_transition_stats["tail_values"],
                            "stage_val_range": stage_transition_stats["range"],
                        },
                    )
                if is_distributed():
                    torch.distributed.barrier()
            elif selective_topology and topology is None:
                topology = build_initial_topology(config, stage.active_compute_nodes)
            current_stage = stage
            if coverage_tracker is not None:
                coverage_tracker.start_stage(stage, split_stats=split_stats, topology=topology)

        model.train()
        unwrapped_model = unwrap_model(model)
        unwrapped_model.set_first_hop_teacher_force_ratio(first_hop_teacher_force_ratio(step, config))
        unwrapped_model.set_training_progress(step, config.train.train_steps)
        current_rollout_steps = training_rollout_steps(step, config)
        unwrapped_model.set_rollout_steps_override(current_rollout_steps)
        unwrapped_model.set_growth_context(
            active_compute_nodes=stage.active_compute_nodes,
            bootstrap_active=stage.bootstrap_active(step),
            active_node_ids=None if topology is None else topology.active_node_tensor(),
            clockwise_successor_lookup=None if topology is None else topology.successor_lookup(),
        )
        optimizer.zero_grad(set_to_none=True)
        batch_seed = config.train.seed + rank * 100_000 + step
        if task_name == "memory_growth":
            batch = task.generate(
                config.train.batch_size_per_gpu,
                batch_seed,
                active_compute_nodes=stage.active_compute_nodes,
                bootstrap_mode=stage.bootstrap_active(step),
                topology=topology,
            ).to(device)
        elif task_name == "memory":
            batch = task.generate(config.train.batch_size_per_gpu, batch_seed).to(device)
        else:
            batch = task.generate(config.train.batch_size_per_gpu, batch_seed).to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            output = model(batch)
            loss = output["loss"]
        loss.backward()

        coverage_snapshot: dict[str, float | int] = {}
        if config.growth.enabled:
            diagnostics = output.get("diagnostics", {})
            stacked_signals = torch.stack(
                [
                    diagnostics.get(
                        "all_visit_counts",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "task_visit_counts",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "query_visit_counts",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "bootstrap_visit_counts",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "success_visit_counts",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "all_gradient_signal",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "task_gradient_signal",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "query_gradient_signal",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                    diagnostics.get(
                        "bootstrap_gradient_signal",
                        torch.zeros(config.model.num_compute_nodes, device=device, dtype=torch.float32),
                    ).to(device=device, dtype=torch.float64),
                ],
                dim=0,
            )
            if is_distributed():
                torch.distributed.all_reduce(stacked_signals)
            if coverage_tracker is not None:
                coverage_snapshot = coverage_tracker.update(
                    step=step,
                    stage=stage,
                    all_visit_counts=stacked_signals[0],
                    task_visit_counts=stacked_signals[1],
                    query_visit_counts=stacked_signals[2],
                    bootstrap_visit_counts=stacked_signals[3],
                    success_visit_counts=stacked_signals[4],
                    all_gradient_signal=stacked_signals[5],
                    task_gradient_signal=stacked_signals[6],
                    query_gradient_signal=stacked_signals[7],
                    bootstrap_gradient_signal=stacked_signals[8],
                )

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
                if coverage_tracker is not None:
                    for key, value in coverage_tracker.current_snapshot().items():
                        row[f"train/{key}"] = float(value)
                    row["train/stage_local_step"] = int(coverage_snapshot.get("stage_local_step", stage.local_step(step)))
                    row["train/stage_bootstrap_active"] = float(stage.bootstrap_active(step))
                row["train/effective_rollout_steps"] = int(current_rollout_steps)
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
                topology=topology,
                rollout_steps=None,
            )
            stage_val_history.setdefault(stage.index, []).append(
                float(val_metrics.get("query_accuracy", val_metrics.get("query_delivery_rate", 0.0)))
            )
            if is_main_process():
                row = row or {"step": step}
                row["val/loss"] = float(val_metrics["loss"])
                for key, value in val_metrics.items():
                    row[f"val/{key}"] = float(value)
                primary_metric = val_metrics["query_accuracy"] if task_name == "memory" else val_metrics["query_delivery_rate"]
                allow_best_update = (
                    not config.growth.best_metric_final_stage_only
                    or stage.active_compute_nodes == growth_schedule.final_active_compute_nodes
                )
                if allow_best_update and primary_metric >= best_metric:
                    best_metric = primary_metric
                    best_path = save_checkpoint(
                        run_dir,
                        model,
                        optimizer,
                        step,
                        best_metric,
                        "best",
                        growth_topology=topology,
                    )
                    row["best_checkpoint"] = str(best_path)

        if is_main_process() and row is not None:
            metrics_writer.append(row)
            if not args.benchmark_only:
                print(row)

        if is_main_process() and not args.benchmark_only and (
            step % config.train.save_interval == 0 or step == config.train.train_steps
        ):
            save_checkpoint(run_dir, model, optimizer, step, best_metric, "last", growth_topology=topology)

    if args.benchmark_only and is_main_process():
        csv_path = metrics_writer.flush_csv()
        plot_metrics(csv_path, run_dir / "benchmark")
        if coverage_tracker is not None:
            save_json(coverage_tracker.to_dict(), run_dir / "coverage_summary.json")
        print(f"Benchmark run written to {run_dir}")
    elif is_main_process():
        csv_path = metrics_writer.flush_csv()
        plot_metrics(csv_path, run_dir / "training")
        save_checkpoint(
            run_dir,
            model,
            optimizer,
            config.train.train_steps,
            best_metric,
            "last",
            growth_topology=topology,
        )
        if coverage_tracker is not None:
            save_json(coverage_tracker.to_dict(), run_dir / "coverage_summary.json")
        print(f"Training run written to {run_dir}")

    cleanup_distributed()


if __name__ == "__main__":
    main()
