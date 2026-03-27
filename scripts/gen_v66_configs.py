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
PREFIX = "v66"


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
    "delay_override_mode": "learned",
    "delay_override_value": 0,
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
        "train_steps": 1350,
        "stage_steps": [36, 36, 45, 45, 54, 72, 1062],
        "eval_interval": 90,
        "save_interval": 90,
        "anneal_steps": 450,
    },
    "l": {
        "display": "L",
        "train_steps": 2160,
        "stage_steps": [56, 56, 70, 70, 84, 112, 1712],
        "eval_interval": 120,
        "save_interval": 120,
        "anneal_steps": 700,
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

COLLISION_CONDITIONS = {
    "cacheon": {
        "display": "Cache-On",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 0,
            "cache_retrieval_topk": 0,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": True,
        },
    },
    "nocache": {
        "display": "No-Cache",
        "model": {
            "enable_cache": False,
            "cache_visible_recent_limit": 0,
            "cache_retrieval_topk": 0,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": True,
        },
    },
    "recent1": {
        "display": "Recent-1",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 1,
            "cache_retrieval_topk": 0,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": True,
        },
    },
    "topk1": {
        "display": "TopK-1",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 0,
            "cache_retrieval_topk": 1,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": True,
        },
    },
    "classoff": {
        "display": "Class-Off",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 0,
            "cache_retrieval_topk": 0,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": False,
        },
    },
    "recent1classoff": {
        "display": "Recent-1 Class-Off",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 1,
            "cache_retrieval_topk": 0,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": False,
        },
    },
    "topk1classoff": {
        "display": "TopK-1 Class-Off",
        "model": {
            "enable_cache": True,
            "cache_visible_recent_limit": 0,
            "cache_retrieval_topk": 1,
            "cache_retrieval_score_temperature": 1.0,
            "cache_read_bypass_mode": "none",
            "use_reserved_class_slice": False,
        },
    },
}

DELAY_REGIMES = {
    "d1": {
        "display": "D1",
        "mode": "required_wait",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "home_node_pool_size": 8,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4],
        "required_delay_min": 1,
        "required_delay_max": 2,
        "required_delay_hash_bits": 2,
        "fixed_good_delay": 2,
    },
    "d2": {
        "display": "D2",
        "mode": "required_wait",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "home_node_pool_size": 8,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4],
        "required_delay_min": 3,
        "required_delay_max": 4,
        "required_delay_hash_bits": 2,
        "fixed_good_delay": 3,
    },
    "rd1": {
        "display": "RD1",
        "mode": "key_hash_exact_wait",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "home_node_pool_size": 8,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4],
        "required_delay_min": 1,
        "required_delay_max": 2,
        "required_delay_hash_bits": 2,
        "fixed_good_delay": 1,
    },
    "rd2": {
        "display": "RD2",
        "mode": "key_hash_exact_wait",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "home_node_pool_size": 8,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4],
        "required_delay_min": 2,
        "required_delay_max": 4,
        "required_delay_hash_bits": 3,
        "fixed_good_delay": 3,
    },
}

DELAY_CONDITIONS = {
    "learned": {
        "display": "Learned",
        "train": {
            "delay_override_mode": "learned",
            "delay_override_value": 0,
        },
    },
    "zero": {
        "display": "Zero",
        "train": {
            "delay_override_mode": "zero",
            "delay_override_value": 0,
        },
    },
    "random": {
        "display": "Random",
        "train": {
            "delay_override_mode": "random",
            "delay_override_value": 0,
        },
    },
    "fixed": {
        "display": "Fixed",
        "train": {
            "delay_override_mode": "fixed",
            "delay_override_value": 0,
        },
    },
    "required": {
        "display": "Required-Oracle",
        "train": {
            "delay_override_mode": "required",
            "delay_override_value": 0,
        },
    },
}

SELECTORS = {
    "visitonly": {
        "display": "V",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half": {
        "display": "VT-0.5",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
        },
    },
}

PAIRS = {
    "visit_taskgrad_half_d": {"display": "VT-0.5/D"},
    "visitonly_d": {"display": "V/D"},
}


def deep_merge(dst: dict, src: dict) -> dict:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            deep_merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst


def base_config(run_name: str, schedule: str) -> dict:
    schedule_cfg = SCHEDULES[schedule]
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {
            "name": "memory_growth",
        },
        "train": deep_merge(
            copy.deepcopy(TRAIN_BASE),
            {
                "seed": 1234,
                "train_steps": schedule_cfg["train_steps"],
                "eval_interval": schedule_cfg["eval_interval"],
                "save_interval": schedule_cfg["save_interval"],
                "first_hop_teacher_force_anneal_steps": schedule_cfg["anneal_steps"],
            },
        ),
        "runtime": {
            "run_name": run_name,
            "notes": "",
        },
        "growth": deep_merge(
            copy.deepcopy(GROWTH_BASE),
            {
                "stage_steps": schedule_cfg["stage_steps"],
            },
        ),
    }


def apply_pair(cfg: dict, pair: str) -> None:
    selector_key = pair.rsplit("_", 1)[0]
    deep_merge(cfg["growth"], SELECTORS[selector_key]["growth"])
    cfg["runtime"]["notes"] = f"selector={selector_key};contract=d"


def apply_collision_regime(cfg: dict, regime: str) -> None:
    deep_merge(cfg["task"], COLLISION_REGIMES[regime])


def apply_delay_regime(cfg: dict, regime: str) -> None:
    delay_regime = DELAY_REGIMES[regime]
    deep_merge(
        cfg["task"],
        {
            "writers_per_episode": delay_regime["writers_per_episode"],
            "start_node_pool_size": delay_regime["start_node_pool_size"],
            "home_node_pool_size": delay_regime["home_node_pool_size"],
            "query_ttl_min": delay_regime["query_ttl_min"],
            "query_ttl_max": delay_regime["query_ttl_max"],
            "max_rollout_steps": delay_regime["max_rollout_steps"],
            "train_eval_writers": delay_regime["train_eval_writers"],
            "delay_mode": delay_regime["mode"],
            "required_delay_min": delay_regime["required_delay_min"],
            "required_delay_max": delay_regime["required_delay_max"],
            "required_delay_hash_bits": delay_regime["required_delay_hash_bits"],
        },
    )


def build_collision_config(pair: str, regime: str, schedule: str, *, condition: str) -> dict:
    run_name = f"{PREFIX}-collision-{regime}-{condition}-{pair}-32-{schedule}"
    cfg = base_config(run_name, schedule)
    apply_pair(cfg, pair)
    apply_collision_regime(cfg, regime)
    deep_merge(cfg["model"], COLLISION_CONDITIONS[condition]["model"])
    cfg["runtime"]["notes"] += f";pack=collision;regime={regime};condition={condition}"
    return cfg


def build_delay_config(pair: str, regime: str, schedule: str, *, condition: str) -> dict:
    run_name = f"{PREFIX}-delay-{regime}-{condition}-{pair}-32-{schedule}"
    cfg = base_config(run_name, schedule)
    apply_pair(cfg, pair)
    apply_delay_regime(cfg, regime)
    deep_merge(cfg["train"], DELAY_CONDITIONS[condition]["train"])
    if condition == "fixed":
        cfg["train"]["delay_override_value"] = DELAY_REGIMES[regime]["fixed_good_delay"]
    cfg["runtime"]["notes"] += f";pack=delay;regime={regime};condition={condition}"
    return cfg


def build_train_wrapper(config_path: Path) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")/.."',
            'source .venv/bin/activate',
            'GPU_ID="${GPU_ID:-0}"',
            'SEED="${SEED:-1234}"',
            'RUN_NAME="${RUN_NAME:-}"',
            'TRAIN_STEPS_OVERRIDE="${TRAIN_STEPS_OVERRIDE:-}"',
            'EXTRA_ARGS="${EXTRA_ARGS:-}"',
            'ARGS=(--config "{}" --seed "${{SEED}}")'.format(config_path.as_posix()),
            'if [[ -n "${RUN_NAME}" ]]; then ARGS+=(--run-name "${RUN_NAME}"); fi',
            'if [[ -n "${TRAIN_STEPS_OVERRIDE}" ]]; then ARGS+=(--train-steps "${TRAIN_STEPS_OVERRIDE}"); fi',
            'if [[ -n "${EXTRA_ARGS}" ]]; then',
            '  # shellcheck disable=SC2206',
            '  EXTRA_ARR=(${EXTRA_ARGS})',
            '  ARGS+=("${EXTRA_ARR[@]}")',
            'fi',
            'export CUDA_VISIBLE_DEVICES="${GPU_ID}"',
            'python3 -m apsgnn.train "${ARGS[@]}"',
            "",
        ]
    )


def write_config(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    wrapper_path = SCRIPTS / f"train_{path.stem}.sh"
    wrapper_path.write_text(build_train_wrapper(path), encoding="utf-8")
    wrapper_path.chmod(0o755)


def pack_definitions() -> dict:
    return {
        "budgets": {key: value["train_steps"] for key, value in SCHEDULES.items()},
        "pairs": PAIRS,
        "collision_pack": {
            "regimes": COLLISION_REGIMES,
            "conditions": ["cacheon", "nocache", "recent1", "topk1"],
            "runner_up_control": ["cacheon", "nocache"],
            "promotion_gate": {
                "pass_requires": "intervention_recovers_meaningful_fraction_of_cache_on_deficit_or_decodability_redirects_blame",
                "metric": "recovery_fraction_or_decodability_signal",
            },
        },
        "delay_pack": {
            "current_regimes": {key: DELAY_REGIMES[key] for key in ("d1", "d2")},
            "redesigned_regimes": {key: DELAY_REGIMES[key] for key in ("rd1", "rd2")},
            "audit_controls": ["learned", "zero", "random", "fixed", "required"],
            "promotion_gate": {
                "pass_requires": "current_benchmark_valid_or_redesigned_learned_delay_beats_zero",
                "metric": "learned_minus_zero_gap_and_timing_oracle_alignment",
            },
        },
        "optional_followup_pack": {
            "choices": ["class_slice_removal", "adaptive_compute", "branch_packets"],
        },
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    collision_pairs = list(PAIRS)
    for pair in collision_pairs:
        for schedule in SCHEDULES:
            for regime in COLLISION_REGIMES:
                for condition in ("cacheon", "nocache", "recent1", "topk1", "classoff", "recent1classoff", "topk1classoff"):
                    cfg = build_collision_config(pair, regime, schedule, condition=condition)
                    stem = f"{PREFIX}_collision_{regime}_{condition}_{pair}_32_{schedule}"
                    write_config(CONFIGS / f"{stem}.yaml", cfg)
            for regime in DELAY_REGIMES:
                for condition in DELAY_CONDITIONS:
                    cfg = build_delay_config(pair, regime, schedule, condition=condition)
                    stem = f"{PREFIX}_delay_{regime}_{condition}_{pair}_32_{schedule}"
                    write_config(CONFIGS / f"{stem}.yaml", cfg)

    (REPORTS / "v66_pack_definitions.json").write_text(
        json.dumps(pack_definitions(), indent=2, sort_keys=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
