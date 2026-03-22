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
    "split_parent_policy": "utility",
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
        "train_steps": 1260,
        "stage_steps": [33, 33, 41, 41, 49, 60, 1003],
        "eval_interval": 70,
        "save_interval": 70,
        "anneal_steps": 300,
    },
    "m": {
        "display": "M",
        "train_steps": 2520,
        "stage_steps": [66, 66, 82, 82, 99, 120, 2005],
        "eval_interval": 140,
        "save_interval": 140,
        "anneal_steps": 600,
    },
    "l": {
        "display": "L",
        "train_steps": 3360,
        "stage_steps": [88, 88, 110, 110, 132, 161, 2671],
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

ARMS = {
    "visitonly": {
        "family": "visitonly",
        "category": "static",
        "display": "V",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half": {
        "family": "visit_taskgrad_half",
        "category": "static",
        "display": "VT-0.5",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
        },
    },
    "gate_writers_le2": {
        "family": "g_writers",
        "category": "gate",
        "display": "G_writers<=2",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "writers",
            "selector_gate_label": "G_writers<=2",
            "selector_gate_writers_threshold": 2,
        },
    },
    "gate_writers_le3": {
        "family": "g_writers",
        "category": "gate",
        "display": "G_writers<=3",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "writers",
            "selector_gate_label": "G_writers<=3",
            "selector_gate_writers_threshold": 3,
        },
    },
    "gate_ingress_pool1": {
        "family": "g_ingress",
        "category": "gate",
        "display": "G_ingress(pool1)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "ingress",
            "selector_gate_label": "G_ingress(pool1)",
            "selector_gate_ingress_start_node_pool_threshold": 1,
            "selector_gate_ingress_allow_tight_ttl": False,
        },
    },
    "gate_ingress_pool1_or_tightttl": {
        "family": "g_ingress",
        "category": "gate",
        "display": "G_ingress(pool1|tightttl)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "ingress",
            "selector_gate_label": "G_ingress(pool1|tightttl)",
            "selector_gate_ingress_start_node_pool_threshold": 1,
            "selector_gate_ingress_allow_tight_ttl": True,
        },
    },
    "gate_meta_a": {
        "family": "g_meta",
        "category": "gate",
        "display": "G_meta(A)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "meta",
            "selector_gate_label": "G_meta(A)",
            "selector_gate_meta_writer_weight": 1.0,
            "selector_gate_meta_ingress_bonus": -2.5,
            "selector_gate_meta_tight_ttl_bonus": 1.0,
            "selector_gate_meta_bias": 0.0,
            "selector_gate_meta_threshold": 2.5,
        },
    },
    "gate_meta_b": {
        "family": "g_meta",
        "category": "gate",
        "display": "G_meta(B)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "meta",
            "selector_gate_label": "G_meta(B)",
            "selector_gate_meta_writer_weight": 1.0,
            "selector_gate_meta_ingress_bonus": -2.0,
            "selector_gate_meta_tight_ttl_bonus": 1.0,
            "selector_gate_meta_bias": 0.0,
            "selector_gate_meta_threshold": 2.8,
        },
    },
    "gate_online_a": {
        "family": "g_online",
        "category": "gate",
        "display": "G_online(A)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "online",
            "selector_gate_label": "G_online(A)",
            "selector_gate_online_stage_index_min": 4,
            "selector_gate_online_entropy_high_threshold": 0.68,
            "selector_gate_online_gini_high_threshold": 0.75,
        },
    },
    "gate_online_b": {
        "family": "g_online",
        "category": "gate",
        "display": "G_online(B)",
        "growth": {
            "split_mode": "clone",
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "selector_gate_kind": "online",
            "selector_gate_label": "G_online(B)",
            "selector_gate_online_stage_index_min": 4,
            "selector_gate_online_entropy_high_threshold": 0.66,
            "selector_gate_online_gini_high_threshold": 0.73,
        },
    },
}


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


def build_config(arm: str, regime: str, schedule: str) -> dict:
    schedule_spec = SCHEDULES[schedule]
    arm_spec = ARMS[arm]
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {"name": "memory_growth", **copy.deepcopy(REGIMES[regime])},
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            "train_steps": schedule_spec["train_steps"],
            "eval_interval": schedule_spec["eval_interval"],
            "log_interval": 50,
            "save_interval": schedule_spec["save_interval"],
            "first_hop_teacher_force_anneal_steps": schedule_spec["anneal_steps"],
        },
        "runtime": {
            "run_name": f"v61-{regime}-{arm}-32-{schedule}",
            "notes": (
                f"selector={arm_spec['display']}; family={arm_spec['family']}; "
                f"category={arm_spec['category']}; schedule={schedule_spec['display']}; regime={REGIMES[regime]['display']}"
            ),
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **copy.deepcopy(arm_spec["growth"]),
            "stage_steps": copy.deepcopy(schedule_spec["stage_steps"]),
        },
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    for arm in ARMS:
        for regime in REGIMES:
            for schedule in SCHEDULES:
                config_name = f"v61_{regime}_{arm}_32_{schedule}.yaml"
                run_name = f"v61-{regime}-{arm}-32-{schedule}"
                dump_yaml(build_config(arm, regime, schedule), CONFIGS / config_name)
                write_script(SCRIPTS / f"train_v61_{regime}_{arm}_32_{schedule}.sh", config_name, run_name)


if __name__ == "__main__":
    main()
