#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "reports"
PREFIX = "v83"
PAIR = "visit_taskgrad_half_d"


MODEL_BASE = {
    "nodes_total": 33,
    "max_ttl": 32,
    "d_model": 128,
    "address_dim": 128,
    "nhead": 4,
    "delay_bins": 8,
    "cache_capacity": 48,
    "enable_cache": True,
    "cache_retrieval_score_temperature": 1.0,
    "cache_retrieval_topk": 0,
    "cache_visible_recent_limit": 0,
    "cache_read_bypass_mode": "none",
    "cache_home_summary_fusion": False,
    "cache_output_summary_readout": True,
    "cache_output_summary_source": "ambiguity_gate",
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
    "use_reserved_class_slice": True,
}

TRAIN_BASE = {
    "batch_size_per_gpu": 1,
    "val_batches": 24,
    "lr": 2.0e-4,
    "aux_writer_weight": 2.0,
    "aux_query_weight": 2.0,
    "aux_home_out_weight": 1.0,
    "first_hop_teacher_force_start": 1.0,
    "first_hop_teacher_force_end": 0.0,
    "first_hop_teacher_force_anneal_steps": 100,
    "delay_reg_weight": 0.05,
    "contract_kind": "detached_warmup",
    "contract_detach_temporal_state": True,
    "contract_query_supervision": "final_output",
    "contract_penultimate_keep_prob": 0.0,
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
    "split_parent_policy": "utility",
    "split_mode": "clone",
    "split_mutation_scale": 0.02,
    "gradient_norm_threshold": 1.0e-8,
    "utility_ema_decay": 0.95,
    "utility_success_alpha": 0.0,
    "utility_visit_weight": 1.0,
    "utility_grad_weight": 0.5,
    "utility_query_visit_weight": 0.0,
    "utility_query_grad_weight": 0.0,
    "utility_tail_fraction": 0.5,
    "best_metric_final_stage_only": True,
    "stage_active_counts": [4, 6, 8, 12, 16, 24, 32],
}

SCHEDULES = {
    "p": {
        "display": "P",
        "train_steps": 300,
        "stage_steps": [8, 8, 10, 10, 12, 16, 236],
        "eval_interval": 25,
        "save_interval": 25,
        "anneal_steps": 100,
    },
    "m": {
        "display": "M",
        "train_steps": 900,
        "stage_steps": [24, 24, 30, 30, 36, 48, 708],
        "eval_interval": 60,
        "save_interval": 60,
        "anneal_steps": 300,
    },
    "l": {
        "display": "L",
        "train_steps": 1350,
        "stage_steps": [36, 36, 45, 45, 54, 72, 1062],
        "eval_interval": 90,
        "save_interval": 90,
        "anneal_steps": 450,
    },
}

COLLISION_REGIMES = {
    "c1": {
        "display": "C1",
        "writers_per_episode": 6,
        "start_node_pool_size": 2,
        "home_node_pool_size": 8,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [6, 10],
    },
    "c2": {
        "display": "C2",
        "writers_per_episode": 6,
        "start_node_pool_size": 2,
        "home_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [6, 10],
    },
}

CONDITIONS = {
    "ambig": {
        "display": "AmbiguityAwareReadout",
        "model": {
            "enable_cache": True,
            "cache_output_summary_readout": True,
            "cache_output_summary_source": "ambiguity_gate",
            "cache_output_gate_feature_scale": 1.0,
        },
    },
    "collswitch": {
        "display": "CollisionSwitchReadout",
        "model": {
            "enable_cache": True,
            "cache_output_summary_readout": True,
            "cache_output_summary_source": "collision_switch_gate",
            "cache_output_gate_feature_scale": 1.0,
            "cache_output_collision_switch_width": 2.0,
        },
    },
}


def base_config() -> dict:
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {
            "name": "memory_growth",
            "writers_per_episode": 6,
            "writer_inject_step": 0,
            "query_inject_step": 2,
            "start_node_pool_size": 2,
            "home_node_pool_size": 8,
            "query_ttl_min": 2,
            "query_ttl_max": 3,
            "max_rollout_steps": 12,
            "delay_mode": "none",
            "required_delay_min": 0,
            "required_delay_max": 0,
            "required_delay_hash_bits": 4,
            "train_eval_writers": [6, 10],
            "hash_seed": 17,
        },
        "train": copy.deepcopy(TRAIN_BASE),
        "runtime": {
            "output_root": "runs",
            "report_root": "reports",
            "run_name": "",
            "notes": "v83 collision-switch gate",
        },
        "growth": copy.deepcopy(GROWTH_BASE),
    }


def build_collision_config(regime: str, schedule: str, *, condition: str) -> dict:
    cfg = base_config()
    cfg["task"].update(copy.deepcopy(COLLISION_REGIMES[regime]))
    cfg["model"].update(copy.deepcopy(CONDITIONS[condition]["model"]))
    sched = SCHEDULES[schedule]
    cfg["train"]["train_steps"] = sched["train_steps"]
    cfg["train"]["eval_interval"] = sched["eval_interval"]
    cfg["train"]["save_interval"] = sched["save_interval"]
    cfg["train"]["first_hop_teacher_force_anneal_steps"] = sched["anneal_steps"]
    cfg["growth"]["stage_steps"] = list(sched["stage_steps"])
    cfg["runtime"]["run_name"] = f"{PREFIX}-collision-{regime}-{condition}-{PAIR}-32-{schedule}"
    return cfg


def config_filename(regime: str, condition: str, schedule: str) -> str:
    return f"{PREFIX}_collision_{regime}_{condition}_{PAIR}_32_{schedule}.yaml"


def script_filename(regime: str, condition: str, schedule: str) -> str:
    return f"train_{PREFIX}_collision_{regime}_{condition}_{PAIR}_32_{schedule}.sh"


def write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_train_script(path: Path, config_path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                'cd "$(dirname "$0")/.."',
                "source .venv/bin/activate",
                'GPU_ID="${GPU_ID:-0}"',
                'SEED="${SEED:-1234}"',
                'RUN_NAME="${RUN_NAME:-}"',
                'TRAIN_STEPS_OVERRIDE="${TRAIN_STEPS_OVERRIDE:-}"',
                'EXTRA_ARGS="${EXTRA_ARGS:-}"',
                f'ARGS=(--config "{config_path}" --seed "${{SEED}}")',
                'if [[ -n "${RUN_NAME}" ]]; then ARGS+=(--run-name "${RUN_NAME}"); fi',
                'if [[ -n "${TRAIN_STEPS_OVERRIDE}" ]]; then ARGS+=(--train-steps "${TRAIN_STEPS_OVERRIDE}"); fi',
                'if [[ -n "${EXTRA_ARGS}" ]]; then',
                "  # shellcheck disable=SC2206",
                '  EXTRA_ARR=(${EXTRA_ARGS})',
                '  ARGS+=("${EXTRA_ARR[@]}")',
                "fi",
                'export CUDA_VISIBLE_DEVICES="${GPU_ID}"',
                'python3 -m apsgnn.train "${ARGS[@]}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def pack_definitions() -> dict:
    return {
        "experiment": "v83_collision_switch_gate",
        "pair": PAIR,
        "conditions": CONDITIONS,
        "collision_regimes": COLLISION_REGIMES,
        "schedules": SCHEDULES,
        "promotion_rule": "Positive only if the collision-switch gate improves C2 dense_mean in the main seeds and does not lose the fresh rerun.",
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    for regime in COLLISION_REGIMES:
        for condition in CONDITIONS:
            for schedule in SCHEDULES:
                config_path = CONFIGS / config_filename(regime, condition, schedule)
                script_path = SCRIPTS / script_filename(regime, condition, schedule)
                write_yaml(config_path, build_collision_config(regime, schedule, condition=condition))
                write_train_script(script_path, config_path)
    (REPORTS / "v83_pack_definitions.json").write_text(
        json.dumps(pack_definitions(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
