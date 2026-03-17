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
    query_ttl_min: int = 3
    query_ttl_max: int = 6
    max_rollout_steps: int = 10
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
    split_mutation_scale: float = 0.02
    gradient_norm_threshold: float = 1.0e-12
    utility_ema_decay: float = 0.95


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
