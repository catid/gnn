#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
PREFIX = "v63ee"


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
    "contract_kind": "baseline",
    "contract_detach_temporal_state": False,
    "contract_query_supervision": "first_output",
    "contract_penultimate_keep_prob": 0.0,
    "contract_shallow_train_fraction": 0.0,
    "contract_shallow_rollout_steps": 0,
    "late_stage_stability_weight": 0.0,
    "late_stage_stability_start_fraction": 0.5,
    "late_stage_stability_after_home_only": True,
    "slow_commit_interval": 0,
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

# The 100-step profile justifies taking the one allowed 10% reduction on S/M/L.
SCHEDULES = {
    "p": {
        "display": "P",
        "train_steps": 300,
        "stage_steps": [8, 8, 10, 10, 12, 15, 237],
        "eval_interval": 24,
        "save_interval": 24,
        "anneal_steps": 80,
    },
    "s": {
        "display": "S",
        "train_steps": 810,
        "stage_steps": [22, 22, 26, 26, 32, 40, 642],
        "eval_interval": 60,
        "save_interval": 60,
        "anneal_steps": 220,
    },
    "m": {
        "display": "M",
        "train_steps": 1620,
        "stage_steps": [43, 43, 53, 53, 64, 77, 1287],
        "eval_interval": 100,
        "save_interval": 100,
        "anneal_steps": 420,
    },
    "l": {
        "display": "L",
        "train_steps": 2268,
        "stage_steps": [60, 60, 74, 74, 89, 108, 1803],
        "eval_interval": 140,
        "save_interval": 140,
        "anneal_steps": 600,
    },
}

REGIMES = {
    "core": {
        "display": "Core",
        "writers_per_episode": 2,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [2, 6, 10],
    },
    "t1": {
        "display": "T1",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t1r": {
        "display": "T1r",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2a": {
        "display": "T2a",
        "writers_per_episode": 4,
        "start_node_pool_size": 1,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2b": {
        "display": "T2b",
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 2,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2c": {
        "display": "T2c",
        "writers_per_episode": 6,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [6, 10, 14, 16],
    },
    "hmid": {
        "display": "Hmid",
        "writers_per_episode": 3,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [3, 7, 11],
    },
    "hmix": {
        "display": "Hmix",
        "writers_per_episode": 3,
        "start_node_pool_size": 1,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [3, 7, 11],
    },
}

CONTRACTS = {
    "b": {
        "family": "baseline",
        "display": "B",
        "train": {
            "contract_kind": "baseline",
            "contract_detach_temporal_state": False,
            "contract_query_supervision": "first_output",
            "contract_penultimate_keep_prob": 0.0,
            "contract_shallow_train_fraction": 0.0,
            "contract_shallow_rollout_steps": 0,
        },
    },
    "d": {
        "family": "detached_warmup",
        "display": "D",
        "train": {
            "contract_kind": "detached_warmup",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.0,
            "contract_shallow_train_fraction": 0.0,
            "contract_shallow_rollout_steps": 0,
        },
    },
    "ds": {
        "family": "detached_stochastic_penultimate",
        "display": "DS",
        "train": {
            "contract_kind": "detached_stochastic_penultimate",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.10,
            "contract_shallow_train_fraction": 0.0,
            "contract_shallow_rollout_steps": 0,
        },
    },
    "dsg": {
        "family": "detached_stochastic_growshort",
        "display": "DSG",
        "train": {
            "contract_kind": "detached_stochastic_growshort",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.10,
            "contract_shallow_train_fraction": 0.80,
            "contract_shallow_rollout_steps": 8,
        },
    },
}

STATIC_SELECTORS = {
    "visitonly": {
        "display": "V",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half": {
        "display": "VT-0.5",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.5,
        },
    },
}

EXPLOITATION_ARMS = {
    f"{selector}_{contract}": {
        "category": "exploit",
        "display": f"{STATIC_SELECTORS[selector]['display']}/{CONTRACTS[contract]['display']}",
        "selector_label": STATIC_SELECTORS[selector]["display"],
        "contract_label": CONTRACTS[contract]["display"],
        "selector": selector,
        "contract": contract,
        "growth": STATIC_SELECTORS[selector]["growth"],
        "train": CONTRACTS[contract]["train"],
    }
    for selector in STATIC_SELECTORS
    for contract in CONTRACTS
}

EXPLORATION_ARMS = {
    "stage_late_vt": {
        "category": "explore",
        "display": "StageLate-VT",
        "selector_label": "V→VT-0.5",
        "contract_label": "DSG",
        "selector": "visitonly",
        "contract": "dsg",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "adaptive_selector_stage_index_min": 6,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.5,
        },
        "train": CONTRACTS["dsg"]["train"],
    },
    "stage_early_vt": {
        "category": "explore",
        "display": "StageEarly-VT",
        "selector_label": "VT-0.5→V",
        "contract_label": "DSG",
        "selector": "visit_taskgrad_half",
        "contract": "dsg",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "adaptive_selector_stage_index_min": 6,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.0,
        },
        "train": CONTRACTS["dsg"]["train"],
    },
    "stable_v": {
        "category": "explore",
        "display": "Stable-V",
        "selector_label": "V",
        "contract_label": "DSG+stable",
        "selector": "visitonly",
        "contract": "dsg",
        "growth": STATIC_SELECTORS["visitonly"]["growth"],
        "train": {**CONTRACTS["dsg"]["train"], "late_stage_stability_weight": 0.005},
    },
    "stable_vt": {
        "category": "explore",
        "display": "Stable-VT",
        "selector_label": "VT-0.5",
        "contract_label": "DSG+stable",
        "selector": "visit_taskgrad_half",
        "contract": "dsg",
        "growth": STATIC_SELECTORS["visit_taskgrad_half"]["growth"],
        "train": {**CONTRACTS["dsg"]["train"], "late_stage_stability_weight": 0.005},
    },
    "gonline": {
        "category": "explore",
        "display": "GOnline",
        "selector_label": "V|VT-0.5",
        "contract_label": "DSG+gate",
        "selector": "visitonly",
        "contract": "dsg",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "online",
            "selector_gate_label": "GOnline-A",
            "selector_gate_online_stage_index_min": 5,
            "selector_gate_online_entropy_high_threshold": 0.80,
            "selector_gate_online_gini_high_threshold": 0.55,
        },
        "train": CONTRACTS["dsg"]["train"],
    },
    "slowcommit": {
        "category": "explore",
        "display": "SlowCommit",
        "selector_label": "VT-0.5",
        "contract_label": "DSG+slow",
        "selector": "visit_taskgrad_half",
        "contract": "dsg",
        "growth": STATIC_SELECTORS["visit_taskgrad_half"]["growth"],
        "train": {**CONTRACTS["dsg"]["train"], "slow_commit_interval": 2},
    },
}

ARMS = {**EXPLOITATION_ARMS, **EXPLORATION_ARMS}


def build_config(arm: str, regime: str, schedule: str) -> dict[str, object]:
    arm_spec = ARMS[arm]
    regime_spec = REGIMES[regime]
    schedule_spec = SCHEDULES[schedule]
    task_spec = {key: value for key, value in regime_spec.items() if key != "display"}
    train_spec = {key: value for key, value in schedule_spec.items() if key in {"train_steps", "eval_interval", "save_interval", "anneal_steps"}}
    growth_schedule = {"stage_steps": copy.deepcopy(schedule_spec["stage_steps"])}

    config = {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {
            "name": "memory_growth",
            **copy.deepcopy(task_spec),
        },
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            **copy.deepcopy(train_spec),
        },
        "runtime": {
            "run_name": f"{PREFIX}-{regime}-{arm}-32-{schedule}",
            "notes": f"{PREFIX.upper()} {arm_spec['display']} on {regime_spec['display']} ({schedule_spec['display']})",
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **growth_schedule,
        },
    }
    config["growth"].update(copy.deepcopy(arm_spec["growth"]))
    config["train"].update(copy.deepcopy(arm_spec["train"]))
    return config


def write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_wrapper(path: Path, config_name: str, default_run_name: str) -> None:
    path.write_text(
        "\n".join(
            [
                "#!/usr/bin/env bash",
                f"CONFIG={config_name} DEFAULT_SEED=1234 DEFAULT_RUN_NAME={default_run_name} exec \"$(dirname \"$0\")/_train_wrapper_single_gpu.sh\"",
                "",
            ]
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def emit_all() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for regime in REGIMES:
        for arm in ARMS:
            for schedule in SCHEDULES:
                config_name = f"{PREFIX}_{regime}_{arm}_32_{schedule}.yaml"
                script_name = f"train_{PREFIX}_{regime}_{arm}_32_{schedule}.sh"
                run_name = f"{PREFIX}-{regime}-{arm}-32-{schedule}"
                write_yaml(CONFIGS / config_name, build_config(arm, regime, schedule))
                write_wrapper(SCRIPTS / script_name, f"configs/{config_name}", run_name)


if __name__ == "__main__":
    emit_all()
