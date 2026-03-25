#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SCRIPTS = ROOT / "scripts"
PREFIX = "v64"


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
    "aux_home_out_weight": 1.0,
    "first_hop_teacher_force_start": 1.0,
    "first_hop_teacher_force_end": 0.0,
    "first_hop_teacher_force_anneal_steps": 100,
    "contract_kind": "baseline",
    "contract_detach_temporal_state": False,
    "contract_query_supervision": "first_output",
    "contract_penultimate_keep_prob": 0.0,
    "contract_shallow_train_fraction": 0.0,
    "contract_shallow_rollout_steps": 0,
    "contract_aux_anneal_start_fraction": 1.0,
    "contract_aux_anneal_final_multiplier": 1.0,
    "contract_rand_depth_train_fraction": 0.0,
    "contract_rand_depth_multipliers": [],
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

# The 100-step profile from the same 2-GPU machine justifies the single allowed 10% reduction.
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

RAW_CORE_CONTRACTS = {
    "d": {
        "display": "D",
        "family": "detached_warmup",
        "train": {
            "contract_kind": "detached_warmup",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.0,
        },
    },
    "ds_p005": {
        "display": "DS-p0.05",
        "family": "detached_stochastic_penultimate",
        "train": {
            "contract_kind": "detached_stochastic_penultimate_p005",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.05,
        },
    },
    "ds_p010": {
        "display": "DS-p0.10",
        "family": "detached_stochastic_penultimate",
        "train": {
            "contract_kind": "detached_stochastic_penultimate_p010",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.10,
        },
    },
    "ds_p025": {
        "display": "DS-p0.25",
        "family": "detached_stochastic_penultimate",
        "train": {
            "contract_kind": "detached_stochastic_penultimate_p025",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.25,
        },
    },
    "ds_p040": {
        "display": "DS-p0.40",
        "family": "detached_stochastic_penultimate",
        "train": {
            "contract_kind": "detached_stochastic_penultimate_p040",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.40,
        },
    },
    "ds_fixed1step": {
        "display": "DS-fixed1step",
        "family": "detached_fixed1step",
        "train": {
            "contract_kind": "detached_stochastic_fixed1step",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 0.0,
        },
    },
    "ds_fixed2step": {
        "display": "DS-fixed2step",
        "family": "detached_fixed2step",
        "train": {
            "contract_kind": "detached_stochastic_fixed2step",
            "contract_detach_temporal_state": True,
            "contract_query_supervision": "final_output",
            "contract_penultimate_keep_prob": 1.0,
        },
    },
}


def build_main_contracts(
    *,
    core_best: str,
    core_runner_up: str,
    aux_final_multiplier: float,
) -> dict[str, dict[str, object]]:
    if core_best not in RAW_CORE_CONTRACTS:
        raise KeyError(f"Unknown core-best contract: {core_best}")
    if core_runner_up not in RAW_CORE_CONTRACTS:
        raise KeyError(f"Unknown core runner-up contract: {core_runner_up}")

    def _with_train(base_key: str, display: str, train_updates: dict[str, object]) -> dict[str, object]:
        spec = copy.deepcopy(RAW_CORE_CONTRACTS[base_key])
        spec["display"] = display
        spec["source_key"] = base_key
        spec["train"] = {**spec["train"], **train_updates}
        return spec

    return {
        "d": _with_train("d", "D", {}),
        "ds_core_best": _with_train(
            core_best,
            "DS-core-best",
            {"contract_kind": f"ds_core_best_from_{core_best}"},
        ),
        "ds_core_runner_up": _with_train(
            core_runner_up,
            "DS-core-runner-up",
            {"contract_kind": f"ds_core_runner_up_from_{core_runner_up}"},
        ),
        "ds_auxanneal_050": _with_train(
            core_best,
            "DS+AuxAnneal(0.50)",
            {
                "contract_kind": f"ds_auxanneal050_from_{core_best}",
                "contract_aux_anneal_start_fraction": 0.60,
                "contract_aux_anneal_final_multiplier": 0.50,
            },
        ),
        "ds_auxanneal_025": _with_train(
            core_best,
            "DS+AuxAnneal(0.25)",
            {
                "contract_kind": f"ds_auxanneal025_from_{core_best}",
                "contract_aux_anneal_start_fraction": 0.60,
                "contract_aux_anneal_final_multiplier": 0.25,
            },
        ),
        "ds_auxanneal": _with_train(
            core_best,
            "DS+AuxAnneal",
            {
                "contract_kind": f"ds_auxanneal_from_{core_best}",
                "contract_aux_anneal_start_fraction": 0.60,
                "contract_aux_anneal_final_multiplier": float(aux_final_multiplier),
            },
        ),
        "ds_randdepth": _with_train(
            core_best,
            "DS+RandDepth",
            {
                "contract_kind": f"ds_randdepth_from_{core_best}",
                "contract_rand_depth_train_fraction": 0.80,
                "contract_rand_depth_multipliers": [0.75, 1.0, 1.25],
            },
        ),
    }


def build_contracts(
    *,
    core_best: str,
    core_runner_up: str,
    aux_final_multiplier: float,
) -> dict[str, dict[str, object]]:
    contracts = copy.deepcopy(RAW_CORE_CONTRACTS)
    contracts.update(
        build_main_contracts(
            core_best=core_best,
            core_runner_up=core_runner_up,
            aux_final_multiplier=aux_final_multiplier,
        )
    )
    return contracts


def build_pair_specs(
    *,
    core_best: str,
    core_runner_up: str,
    aux_final_multiplier: float,
) -> dict[str, dict[str, object]]:
    contracts = build_contracts(
        core_best=core_best,
        core_runner_up=core_runner_up,
        aux_final_multiplier=aux_final_multiplier,
    )
    out: dict[str, dict[str, object]] = {}
    for selector_key, selector_spec in SELECTORS.items():
        for contract_key, contract_spec in contracts.items():
            pair_key = f"{selector_key}_{contract_key}"
            out[pair_key] = {
                "selector": selector_key,
                "selector_label": selector_spec["display"],
                "contract": contract_key,
                "contract_label": contract_spec["display"],
                "contract_source_key": contract_spec.get("source_key", contract_key),
                "growth": selector_spec["growth"],
                "train": contract_spec["train"],
            }
    return out


def build_config(
    pair_key: str,
    regime: str,
    schedule: str,
    *,
    core_best: str = "ds_p010",
    core_runner_up: str = "ds_p025",
    aux_final_multiplier: float = 0.50,
) -> dict[str, object]:
    pair_specs = build_pair_specs(
        core_best=core_best,
        core_runner_up=core_runner_up,
        aux_final_multiplier=aux_final_multiplier,
    )
    if pair_key not in pair_specs:
        raise KeyError(f"Unknown v64 pair: {pair_key}")
    if regime not in REGIMES:
        raise KeyError(f"Unknown regime: {regime}")
    if schedule not in SCHEDULES:
        raise KeyError(f"Unknown schedule: {schedule}")

    pair = pair_specs[pair_key]
    schedule_spec = copy.deepcopy(SCHEDULES[schedule])
    cfg = {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {
            "name": "memory_growth",
            **copy.deepcopy(REGIMES[regime]),
        },
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            **copy.deepcopy(pair["train"]),
            "train_steps": schedule_spec["train_steps"],
            "eval_interval": schedule_spec["eval_interval"],
            "save_interval": schedule_spec["save_interval"],
            "first_hop_teacher_force_anneal_steps": schedule_spec["anneal_steps"],
        },
        "runtime": {
            "run_name": f"{PREFIX}-{regime}-{pair_key}-32-{schedule}",
            "notes": (
                f"selector={pair['selector_label']};contract={pair['contract_label']};"
                f"source={pair['contract_source_key']};core_best={core_best};"
                f"core_runner_up={core_runner_up};aux_final={aux_final_multiplier:.2f}"
            ),
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **copy.deepcopy(pair["growth"]),
            "stage_steps": schedule_spec["stage_steps"],
        },
    }
    return cfg


def emit_pair_config(
    pair_key: str,
    regime: str,
    schedule: str,
    *,
    core_best: str,
    core_runner_up: str,
    aux_final_multiplier: float,
) -> None:
    cfg = build_config(
        pair_key,
        regime,
        schedule,
        core_best=core_best,
        core_runner_up=core_runner_up,
        aux_final_multiplier=aux_final_multiplier,
    )
    config_path = CONFIGS / f"{PREFIX}_{regime}_{pair_key}_32_{schedule}.yaml"
    script_path = SCRIPTS / f"train_{PREFIX}_{regime}_{pair_key}_32_{schedule}.sh"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    script_path.write_text(
        "#!/usr/bin/env bash\n"
        f'CONFIG=configs/{PREFIX}_{regime}_{pair_key}_32_{schedule}.yaml '
        f"DEFAULT_SEED=1234 DEFAULT_RUN_NAME={PREFIX}-{regime}-{pair_key}-32-{schedule} "
        'exec "$(dirname "$0")/_train_wrapper_single_gpu.sh"\n',
        encoding="utf-8",
    )
    script_path.chmod(0o755)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate v64 DS-factorization configs and wrappers.")
    parser.add_argument("--core-best", default="ds_p010")
    parser.add_argument("--core-runner-up", default="ds_p025")
    parser.add_argument("--aux-final-multiplier", type=float, default=0.50)
    args = parser.parse_args()

    pair_specs = build_pair_specs(
        core_best=args.core_best,
        core_runner_up=args.core_runner_up,
        aux_final_multiplier=args.aux_final_multiplier,
    )
    for regime in REGIMES:
        for schedule in SCHEDULES:
            for pair_key in sorted(pair_specs):
                emit_pair_config(
                    pair_key,
                    regime,
                    schedule,
                    core_best=args.core_best,
                    core_runner_up=args.core_runner_up,
                    aux_final_multiplier=args.aux_final_multiplier,
                )


if __name__ == "__main__":
    main()
