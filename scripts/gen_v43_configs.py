#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"


MODEL_BASE = {
    "nodes_total": 33,
    "max_ttl": 32,
    "d_model": 128,
    "address_dim": 128,
    "nhead": 4,
    "delay_bins": 8,
    "cache_capacity": 48,
    "use_first_hop_key_hint": False,
    "use_learned_first_hop_router": True,
    "first_hop_router_variant": "key_mlp_ce",
    "first_hop_router_hidden_dim": 256,
    "first_hop_router_layers": 3,
    "first_hop_router_use_residual": False,
    "first_hop_router_separate_heads": True,
    "first_hop_router_aux_type": "none",
    "cache_read_variant": "learned_implicit",
    "cache_read_hidden_dim": 256,
    "cache_read_layers": 3,
}

TRAIN_BASE = {
    "batch_size_per_gpu": 1,
    "val_batches": 24,
    "lr": 2.0e-4,
    "aux_writer_weight": 2.0,
    "aux_query_weight": 2.0,
    "first_hop_teacher_force_start": 1.0,
    "first_hop_teacher_force_end": 0.0,
}

GROWTH_BASE = {
    "enabled": True,
    "transition_mode": "split",
    "topology_mode": "selective",
    "bootstrap_steps": 75,
    "clock_prior_bias": 0.35,
    "delay_zero_bias": 0.25,
    "bootstrap_clock_prior_bias": 6.0,
    "bootstrap_delay_zero_bias": 6.0,
    "bootstrap_route_weight": 1.0,
    "bootstrap_delay_weight": 0.1,
    "split_mode": "clone",
    "split_parent_policy": "utility",
    "split_mutation_scale": 0.02,
    "gradient_norm_threshold": 1.0e-8,
    "utility_ema_decay": 0.95,
    "utility_success_alpha": 0.0,
    "utility_query_visit_weight": 0.0,
    "utility_tail_fraction": 0.5,
    "best_metric_final_stage_only": True,
    "stage_active_counts": [4, 6, 8, 12, 16, 24, 32],
}

ARMS = {
    "visit_taskgrad_half": {
        "display": "VT-0.5",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "utility_query_grad_weight": 0.0,
        },
    },
    "stageadaptive_late_half": {
        "display": "StageLate-0.5",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "utility_query_grad_weight": 0.0,
            "adaptive_selector_stage_index_min": 5,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.5,
            "adaptive_utility_query_grad_weight": 0.0,
        },
    },
}

SCHEDULE = {
    "name": "xl",
    "train_steps": 4590,
    "stage_steps": [120, 120, 150, 150, 180, 220, 3650],
    "eval_interval": 255,
    "save_interval": 255,
    "anneal_steps": 1100,
}

REGIMES = {
    "core": {
        "writers_per_episode": 2,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [2, 6, 10],
    },
    "t1": {
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t1r": {
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2a": {
        "writers_per_episode": 4,
        "start_node_pool_size": 1,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
}


def dump_yaml(payload: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_script(path: Path, config_name: str, run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"#!/usr/bin/env bash\nCONFIG=configs/{config_name} DEFAULT_SEED=86234 DEFAULT_RUN_NAME={run_name} exec \"$(dirname \"$0\")/_train_wrapper_single_gpu.sh\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def build_config(arm: str, regime: str) -> dict:
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {"name": "memory_growth", **copy.deepcopy(REGIMES[regime])},
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            "train_steps": SCHEDULE["train_steps"],
            "eval_interval": SCHEDULE["eval_interval"],
            "log_interval": 50,
            "save_interval": SCHEDULE["save_interval"],
            "first_hop_teacher_force_anneal_steps": SCHEDULE["anneal_steps"],
        },
        "runtime": {
            "run_name": f"v43-{regime}-{arm}-32-xl",
            "notes": f"selector={ARMS[arm]['display']}",
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **copy.deepcopy(ARMS[arm]["growth"]),
            "stage_steps": copy.deepcopy(SCHEDULE["stage_steps"]),
        },
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for regime in REGIMES:
            config_name = f"v43_{regime}_{arm}_32_xl.yaml"
            run_name = f"v43-{regime}-{arm}-32-xl"
            dump_yaml(build_config(arm, regime), CONFIGS / config_name)
            write_script(SCRIPTS / f"train_v43_{regime}_{arm}_32_xl.sh", config_name, run_name)


if __name__ == "__main__":
    main()
