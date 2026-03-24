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
    "contract_kind": "baseline",
    "contract_detach_temporal_state": False,
    "contract_query_supervision": "first_output",
    "contract_penultimate_keep_prob": 0.0,
    "contract_shallow_train_fraction": 0.0,
    "contract_shallow_rollout_steps": 0,
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
        "train_steps": 420,
        "stage_steps": [11, 11, 14, 14, 16, 20, 334],
        "eval_interval": 30,
        "save_interval": 30,
        "anneal_steps": 100,
    },
    "s": {
        "display": "S",
        "train_steps": 1134,
        "stage_steps": [30, 30, 37, 37, 45, 54, 901],
        "eval_interval": 80,
        "save_interval": 80,
        "anneal_steps": 300,
    },
    "m": {
        "display": "M",
        "train_steps": 2268,
        "stage_steps": [60, 60, 74, 74, 89, 108, 1803],
        "eval_interval": 140,
        "save_interval": 140,
        "anneal_steps": 600,
    },
    "l": {
        "display": "L",
        "train_steps": 3024,
        "stage_steps": [79, 79, 99, 99, 119, 145, 2404],
        "eval_interval": 180,
        "save_interval": 180,
        "anneal_steps": 800,
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

SELECTORS = {
    "visitonly": {
        "family": "visitonly",
        "display": "V",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half": {
        "family": "visit_taskgrad_half",
        "display": "VT-0.5",
        "growth": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
        },
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

PAIRS = {
    f"{selector}_{contract}": {
        "selector": selector,
        "selector_display": selector_spec["display"],
        "contract": contract,
        "contract_display": contract_spec["display"],
        "label": f"{selector_spec['display']}/{contract_spec['display']}",
        "growth": selector_spec["growth"],
        "train": contract_spec["train"],
        "family": f"{selector_spec['family']}_{contract_spec['family']}",
    }
    for selector, selector_spec in SELECTORS.items()
    for contract, contract_spec in CONTRACTS.items()
}

PILOT_REGIMES = ["core", "t1"]
ANCHOR_REGIMES = ["core", "t1", "t2a"]
VERIFICATION_REGIMES = ["t1r", "t2b", "t2c", "hmid", "hmix"]
EXTRA_DEPTH_REGIMES = ["core", "t1", "t2a", "hmix"]


def dump_yaml(payload: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_script(path: Path, config_name: str, run_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "#!/usr/bin/env bash\n"
            f"CONFIG=configs/{config_name} DEFAULT_SEED=1234 DEFAULT_RUN_NAME={run_name} "
            "exec \"$(dirname \"$0\")/_train_wrapper_single_gpu.sh\"\n"
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)


def build_config(pair: str, regime: str, schedule: str) -> dict:
    pair_spec = PAIRS[pair]
    schedule_spec = SCHEDULES[schedule]
    selector_display = pair_spec["selector_display"]
    contract_display = pair_spec["contract_display"]
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {"name": "memory_growth", **copy.deepcopy(REGIMES[regime])},
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            **copy.deepcopy(pair_spec["train"]),
            "train_steps": schedule_spec["train_steps"],
            "eval_interval": schedule_spec["eval_interval"],
            "log_interval": 50,
            "save_interval": schedule_spec["save_interval"],
            "first_hop_teacher_force_anneal_steps": schedule_spec["anneal_steps"],
        },
        "runtime": {
            "run_name": f"v63-{regime}-{pair}-32-{schedule}",
            "notes": (
                f"selector={selector_display}; contract={contract_display}; "
                f"schedule={schedule_spec['display']}; regime={REGIMES[regime]['display']}"
            ),
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **copy.deepcopy(pair_spec["growth"]),
            "stage_steps": copy.deepcopy(schedule_spec["stage_steps"]),
        },
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for pair in PAIRS:
        for regime in REGIMES:
            for schedule in SCHEDULES:
                config_name = f"v63_{regime}_{pair}_32_{schedule}.yaml"
                run_name = f"v63-{regime}-{pair}-32-{schedule}"
                dump_yaml(build_config(pair, regime, schedule), CONFIGS / config_name)
                write_script(SCRIPTS / f"train_v63_{regime}_{pair}_32_{schedule}.sh", config_name, run_name)


if __name__ == "__main__":
    main()
