from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass
class ModelConfig:
    nodes_total: int = 32
    d_model: int = 128
    address_dim: int = 128
    nhead: int = 4
    delay_bins: int = 8
    max_ttl: int = 8
    cache_capacity: int = 64
    enable_cache: bool = True
    route_temperature: float = 0.7
    delay_temperature: float = 0.7
    use_first_hop_key_hint: bool = True
    use_learned_first_hop_router: bool = False
    first_hop_router_variant: str = "legacy"
    first_hop_router_hidden_dim: int = 256
    first_hop_router_layers: int = 2
    first_hop_router_use_residual: bool = True
    first_hop_router_residual_scale: float = 0.25
    first_hop_router_separate_heads: bool = False
    first_hop_router_aux_type: str = "none"
    cache_read_variant: str = "explicit"
    cache_read_hidden_dim: int = 256
    cache_read_layers: int = 2
    use_reserved_class_slice: bool = True
    first_hop_hint_residual_scale: float = 0.25
    readout_class_scale: float = 8.0
    num_classes: int = 32
    key_dim: int = 32
    packet_roles: int = 3
    mlp_ratio: int = 4
    dropout: float = 0.0

    @property
    def num_compute_nodes(self) -> int:
        return self.nodes_total - 1


@dataclass
class TaskConfig:
    name: str = "memory"
    writers_per_episode: int = 6
    writer_inject_step: int = 0
    query_inject_step: int = 2
    start_node_pool_size: int = 0
    home_node_pool_size: int = 0
    query_ttl_min: int = 3
    query_ttl_max: int = 6
    max_rollout_steps: int = 10
    delay_mode: str = "none"
    required_delay_min: int = 0
    required_delay_max: int = 0
    sanity_min_ttl: int = 2
    sanity_max_ttl: int = 4
    train_eval_writers: list[int] = field(default_factory=lambda: [6, 10])
    hash_seed: int = 17


@dataclass
class TrainConfig:
    seed: int = 1234
    batch_size_per_gpu: int = 16
    train_steps: int = 3000
    eval_interval: int = 250
    log_interval: int = 25
    save_interval: int = 250
    val_batches: int = 40
    lr: float = 2.0e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    compile_model: bool = False
    aux_writer_weight: float = 1.0
    aux_query_weight: float = 1.0
    aux_home_out_weight: float = 1.0
    address_aux_weight: float = 2.0
    delay_reg_weight: float = 0.05
    gravity_weight: float = 0.01
    missing_output_penalty: float = 0.5
    sanity_route_weight: float = 1.0
    sanity_delay_weight: float = 0.05
    first_hop_teacher_force_start: float = 0.0
    first_hop_teacher_force_end: float = 0.0
    first_hop_teacher_force_anneal_steps: int = 1
    first_hop_router_aux_weight: float = 0.0
    benchmark_warmup_steps: int = 10
    benchmark_steps: int = 100
    init_checkpoint: str = ""
    freeze_first_hop_router: bool = False
    contract_kind: str = "baseline"
    contract_detach_temporal_state: bool = False
    contract_query_supervision: str = "first_output"
    contract_penultimate_keep_prob: float = 0.0
    contract_shallow_train_fraction: float = 0.0
    contract_shallow_rollout_steps: int = 0
    contract_aux_anneal_start_fraction: float = 1.0
    contract_aux_anneal_final_multiplier: float = 1.0
    contract_rand_depth_train_fraction: float = 0.0
    contract_rand_depth_multipliers: list[float] = field(default_factory=list)
    late_stage_stability_weight: float = 0.0
    late_stage_stability_start_fraction: float = 0.5
    late_stage_stability_after_home_only: bool = True
    slow_commit_interval: int = 0
    adaptive_compute_penalty_weight: float = 0.0
    delay_override_mode: str = "learned"
    delay_override_value: int = 0


@dataclass
class RuntimeConfig:
    output_root: str = "runs"
    report_root: str = "reports"
    run_name: str = "default"
    notes: str = ""


@dataclass
class GrowthConfig:
    enabled: bool = False
    stage_active_counts: list[int] = field(default_factory=list)
    stage_steps: list[int] = field(default_factory=list)
    transition_mode: str = "split"
    topology_mode: str = "uniform"
    bootstrap_steps: int = 0
    clock_prior_bias: float = 0.0
    delay_zero_bias: float = 0.0
    bootstrap_clock_prior_bias: float = 0.0
    bootstrap_delay_zero_bias: float = 0.0
    bootstrap_route_weight: float = 1.0
    bootstrap_delay_weight: float = 0.05
    bootstrap_force_clockwise: bool = True
    bootstrap_force_delay_zero: bool = True
    split_mode: str = "none"
    split_parent_policy: str = "balanced"
    split_mutation_scale: float = 0.02
    mutation_stage_index_min: int = 0
    mutation_selected_fraction: float = 1.0
    mutation_score_margin: float = -1.0e9
    mutation_min_visit_z: float = -1.0e9
    mutation_min_query_grad_z: float = -1.0e9
    mutation_require_stagnation: bool = False
    mutation_stagnation_window: int = 2
    mutation_stagnation_delta: float = 0.0
    gradient_norm_threshold: float = 1.0e-12
    utility_ema_decay: float = 0.95
    utility_visit_weight: float = 1.0
    utility_grad_weight: float = 1.0
    utility_success_alpha: float = 0.75
    utility_query_visit_weight: float = 0.0
    utility_query_grad_weight: float = 0.0
    adaptive_selector_stage_index_min: int = -1
    adaptive_utility_visit_weight: float = 1.0
    adaptive_utility_grad_weight: float = 1.0
    adaptive_utility_query_visit_weight: float = 0.0
    adaptive_utility_query_grad_weight: float = 0.0
    selector_gate_kind: str = "none"
    selector_gate_label: str = ""
    selector_gate_writers_threshold: int = -1
    selector_gate_ingress_start_node_pool_threshold: int = -1
    selector_gate_ingress_allow_tight_ttl: bool = False
    selector_gate_meta_writer_weight: float = 0.0
    selector_gate_meta_ingress_bonus: float = 0.0
    selector_gate_meta_tight_ttl_bonus: float = 0.0
    selector_gate_meta_bias: float = 0.0
    selector_gate_meta_threshold: float = 0.0
    selector_gate_online_stage_index_min: int = -1
    selector_gate_online_entropy_high_threshold: float = 1.0e9
    selector_gate_online_gini_high_threshold: float = 1.0e9
    utility_tail_fraction: float = 1.0
    best_metric_final_stage_only: bool = False


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    growth: GrowthConfig = field(default_factory=GrowthConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ConfigT = TypeVar("ConfigT")


def _selector_weight_payload(
    *,
    visit_weight: float,
    grad_weight: float,
    query_visit_weight: float,
    query_grad_weight: float,
) -> dict[str, float]:
    return {
        "utility_visit_weight": float(visit_weight),
        "utility_grad_weight": float(grad_weight),
        "utility_query_visit_weight": float(query_visit_weight),
        "utility_query_grad_weight": float(query_grad_weight),
    }


def _selector_label_for_weights(weights: dict[str, float]) -> str:
    if (
        float(weights["utility_visit_weight"]) == 1.0
        and float(weights["utility_grad_weight"]) == 0.5
        and float(weights["utility_query_visit_weight"]) == 0.0
        and float(weights["utility_query_grad_weight"]) == 0.0
    ):
        return "VT-0.5"
    if (
        float(weights["utility_visit_weight"]) == 1.0
        and float(weights["utility_grad_weight"]) == 0.0
        and float(weights["utility_query_visit_weight"]) == 0.0
        and float(weights["utility_query_grad_weight"]) == 0.0
    ):
        return "V"
    return "custom"


def _static_selector_weights(growth: GrowthConfig) -> dict[str, float]:
    return _selector_weight_payload(
        visit_weight=growth.utility_visit_weight,
        grad_weight=growth.utility_grad_weight,
        query_visit_weight=growth.utility_query_visit_weight,
        query_grad_weight=growth.utility_query_grad_weight,
    )


def _adaptive_selector_weights(growth: GrowthConfig) -> dict[str, float]:
    return _selector_weight_payload(
        visit_weight=growth.adaptive_utility_visit_weight,
        grad_weight=growth.adaptive_utility_grad_weight,
        query_visit_weight=growth.adaptive_utility_query_visit_weight,
        query_grad_weight=growth.adaptive_utility_query_grad_weight,
    )


def _vt_half_weights() -> dict[str, float]:
    return _selector_weight_payload(
        visit_weight=1.0,
        grad_weight=0.5,
        query_visit_weight=0.0,
        query_grad_weight=0.0,
    )


def _v_weights() -> dict[str, float]:
    return _selector_weight_payload(
        visit_weight=1.0,
        grad_weight=0.0,
        query_visit_weight=0.0,
        query_grad_weight=0.0,
    )


def _tight_ttl(task: TaskConfig) -> bool:
    return int(task.query_ttl_min) == int(task.query_ttl_max)


def selector_decision_for_stage(
    growth: GrowthConfig,
    next_stage_index: int,
    *,
    task: TaskConfig | None = None,
    current_snapshot: dict[str, float] | None = None,
) -> dict[str, Any]:
    task = task or TaskConfig()
    snapshot = current_snapshot or {}
    gate_kind = str(growth.selector_gate_kind or "none")

    if gate_kind != "none":
        use_vt_half = False
        gate_metrics: dict[str, float | int] = {"next_stage_index": int(next_stage_index)}
        if gate_kind == "writers":
            threshold = int(growth.selector_gate_writers_threshold)
            writers = int(task.writers_per_episode)
            use_vt_half = threshold >= 0 and writers <= threshold
            gate_metrics.update({"writers_per_episode": writers, "writers_threshold": threshold})
        elif gate_kind == "ingress":
            threshold = int(growth.selector_gate_ingress_start_node_pool_threshold)
            pool_size = int(task.start_node_pool_size)
            tight_ttl = int(_tight_ttl(task))
            use_vt_half = threshold >= 0 and pool_size <= threshold
            if not use_vt_half and bool(growth.selector_gate_ingress_allow_tight_ttl):
                use_vt_half = bool(tight_ttl)
            gate_metrics.update(
                {
                    "start_node_pool_size": pool_size,
                    "ingress_pool_threshold": threshold,
                    "tight_ttl": tight_ttl,
                    "tight_ttl_allowed": int(bool(growth.selector_gate_ingress_allow_tight_ttl)),
                }
            )
        elif gate_kind == "meta":
            writers = float(task.writers_per_episode)
            pool_is_single = 1.0 if int(task.start_node_pool_size) == 1 else 0.0
            tight_ttl = 1.0 if _tight_ttl(task) else 0.0
            meta_score = (
                float(growth.selector_gate_meta_writer_weight) * writers
                + float(growth.selector_gate_meta_ingress_bonus) * pool_is_single
                + float(growth.selector_gate_meta_tight_ttl_bonus) * tight_ttl
                + float(growth.selector_gate_meta_bias)
            )
            threshold = float(growth.selector_gate_meta_threshold)
            use_vt_half = meta_score <= threshold
            gate_metrics.update(
                {
                    "writers_per_episode": writers,
                    "start_node_pool_size": int(task.start_node_pool_size),
                    "tight_ttl": int(tight_ttl),
                    "meta_score": meta_score,
                    "meta_threshold": threshold,
                }
            )
        elif gate_kind == "online":
            min_stage = int(growth.selector_gate_online_stage_index_min)
            entropy_threshold = float(growth.selector_gate_online_entropy_high_threshold)
            gini_threshold = float(growth.selector_gate_online_gini_high_threshold)
            entropy = float(snapshot.get("task_visit_entropy", 0.0))
            gini = float(snapshot.get("task_visit_gini", 0.0))
            use_vt_half = (
                min_stage >= 0
                and int(next_stage_index) >= min_stage
                and (entropy >= entropy_threshold or gini >= gini_threshold)
            )
            gate_metrics.update(
                {
                    "task_visit_entropy": entropy,
                    "task_visit_gini": gini,
                    "online_stage_index_min": min_stage,
                    "online_entropy_high_threshold": entropy_threshold,
                    "online_gini_high_threshold": gini_threshold,
                }
            )
        else:
            raise ValueError(f"Unknown selector gate kind: {gate_kind}")

        weights = _vt_half_weights() if use_vt_half else _v_weights()
        return {
            "mode": "gate",
            "gate_kind": gate_kind,
            "gate_label": str(growth.selector_gate_label or gate_kind),
            "selected_selector": "visit_taskgrad_half" if use_vt_half else "visitonly",
            "selected_selector_label": "VT-0.5" if use_vt_half else "V",
            "used_vt_half": int(use_vt_half),
            "weights": weights,
            "gate_metrics": gate_metrics,
        }

    use_adaptive = (
        int(growth.adaptive_selector_stage_index_min) >= 0
        and int(next_stage_index) >= int(growth.adaptive_selector_stage_index_min)
    )
    weights = _adaptive_selector_weights(growth) if use_adaptive else _static_selector_weights(growth)
    return {
        "mode": "adaptive" if use_adaptive else "static",
        "gate_kind": "none",
        "gate_label": "",
        "selected_selector": _selector_label_for_weights(weights),
        "selected_selector_label": _selector_label_for_weights(weights),
        "used_vt_half": int(_selector_label_for_weights(weights) == "VT-0.5"),
        "weights": weights,
        "gate_metrics": {"next_stage_index": int(next_stage_index)},
    }


def selector_weights_for_stage(
    growth: GrowthConfig,
    next_stage_index: int,
    *,
    task: TaskConfig | None = None,
    current_snapshot: dict[str, float] | None = None,
) -> dict[str, float]:
    return selector_decision_for_stage(
        growth,
        next_stage_index,
        task=task,
        current_snapshot=current_snapshot,
    )["weights"]


def _merge_dataclass(instance: ConfigT, updates: dict[str, Any]) -> ConfigT:
    for dataclass_field in fields(instance):
        key = dataclass_field.name
        if key not in updates:
            continue
        value = getattr(instance, key)
        incoming = updates[key]
        if is_dataclass(value):
            _merge_dataclass(value, incoming)
        else:
            setattr(instance, key, incoming)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    config = ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        updates = yaml.safe_load(handle) or {}
    return _merge_dataclass(config, updates)


def dump_config(config: ExperimentConfig, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
