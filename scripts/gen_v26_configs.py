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
LR_CHOICES_PATH = REPORTS / "v26_lr_choices.json"

BASE_LR = 2.0e-4
PILOT_LR_MULTIPLIERS = (0.8, 1.0, 0.6)
PILOT_PRIMARY_MULTIPLIERS = (0.8, 1.0)

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
    "lr": BASE_LR,
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

SELECTORS = {
    "visitonly": {
        "display": "V",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "utility_query_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_025": {
        "display": "VT-0.25",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.25,
            "utility_query_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_0375": {
        "display": "VT-0.375",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.375,
            "utility_query_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half": {
        "display": "VT-0.5",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "utility_query_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_0625": {
        "display": "VT-0.625",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.625,
            "utility_query_grad_weight": 0.0,
        },
    },
    "visit_taskgrad_half_querygrad_0125": {
        "display": "VT-0.5+Q0.125",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.5,
            "utility_query_grad_weight": 0.125,
        },
    },
    "stageadaptive_vt": {
        "display": "StageAdaptive-VT",
        "weights": {
            "utility_visit_weight": 1.0,
            "utility_grad_weight": 0.0,
            "utility_query_grad_weight": 0.0,
            "adaptive_selector_stage_index_min": 5,
            "adaptive_utility_visit_weight": 1.0,
            "adaptive_utility_grad_weight": 0.5,
            "adaptive_utility_query_grad_weight": 0.0,
        },
    },
    "querygradonly": {
        "display": "Q",
        "weights": {
            "utility_visit_weight": 0.0,
            "utility_grad_weight": 0.0,
            "utility_query_grad_weight": 1.0,
        },
    },
}

SCHEDULES = {
    "p": {
        "train_steps": 450,
        "stage_steps": [20, 20, 30, 30, 40, 50, 260],
        "eval_interval": 75,
        "save_interval": 75,
        "anneal_steps": 220,
    },
    "s": {
        "train_steps": 1350,
        "stage_steps": [60, 60, 75, 75, 90, 120, 870],
        "eval_interval": 90,
        "save_interval": 90,
        "anneal_steps": 450,
    },
    "m": {
        "train_steps": 2550,
        "stage_steps": [90, 90, 120, 120, 150, 180, 1800],
        "eval_interval": 150,
        "save_interval": 150,
        "anneal_steps": 750,
    },
    "l": {
        "train_steps": 3570,
        "stage_steps": [120, 120, 150, 150, 180, 220, 2630],
        "eval_interval": 210,
        "save_interval": 210,
        "anneal_steps": 900,
    },
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
    "t2a": {
        "writers_per_episode": 4,
        "start_node_pool_size": 1,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2b": {
        "writers_per_episode": 4,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 2,
        "max_rollout_steps": 12,
        "train_eval_writers": [4, 8, 12, 14],
    },
    "t2c": {
        "writers_per_episode": 6,
        "start_node_pool_size": 2,
        "query_ttl_min": 2,
        "query_ttl_max": 3,
        "max_rollout_steps": 12,
        "train_eval_writers": [6, 10, 14, 16],
    },
}


def dump_yaml(payload: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def write_script(path: Path, config_name: str, run_name: str, smoke: bool) -> None:
    wrapper = "_smoke_wrapper_single_gpu.sh" if smoke else "_train_wrapper_single_gpu.sh"
    path.write_text(
        f"#!/usr/bin/env bash\nCONFIG=configs/{config_name} DEFAULT_SEED=1234 DEFAULT_RUN_NAME={run_name} exec \"$(dirname \"$0\")/{wrapper}\"\n",
        encoding="utf-8",
    )
    path.chmod(0o755)


def lr_tag(multiplier: float) -> str:
    return f"lr{int(round(multiplier * 100)):03d}"


def load_lr_choices() -> dict[str, float]:
    selected = {selector: 1.0 for selector in SELECTORS}
    if LR_CHOICES_PATH.exists():
        payload = json.loads(LR_CHOICES_PATH.read_text())
        for selector, value in payload.items():
            if selector in selected:
                selected[selector] = float(value)
    return selected


def build_config(selector: str, regime: str, schedule: str, *, lr_multiplier: float) -> dict:
    regime_spec = REGIMES[regime]
    schedule_spec = SCHEDULES[schedule]
    selector_spec = SELECTORS[selector]
    return {
        "model": copy.deepcopy(MODEL_BASE),
        "task": {
            "name": "memory_growth",
            **copy.deepcopy(regime_spec),
        },
        "train": {
            **copy.deepcopy(TRAIN_BASE),
            "lr": BASE_LR * float(lr_multiplier),
            "train_steps": schedule_spec["train_steps"],
            "eval_interval": schedule_spec["eval_interval"],
            "log_interval": 50,
            "save_interval": schedule_spec["save_interval"],
            "first_hop_teacher_force_anneal_steps": schedule_spec["anneal_steps"],
        },
        "runtime": {
            "run_name": f"v26-{regime}-{selector}-32-{schedule}",
            "notes": f"selector={selector}; lr_multiplier={lr_multiplier:.3f}",
        },
        "growth": {
            **copy.deepcopy(GROWTH_BASE),
            **copy.deepcopy(selector_spec["weights"]),
            "stage_steps": copy.deepcopy(schedule_spec["stage_steps"]),
        },
    }


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)
    SCRIPTS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    selected_lrs = load_lr_choices()

    for selector in SELECTORS:
        for multiplier in PILOT_PRIMARY_MULTIPLIERS:
            config_name = f"v26_core_{selector}_32_p_{lr_tag(multiplier)}.yaml"
            payload = build_config(selector, "core", "p", lr_multiplier=multiplier)
            payload["runtime"]["run_name"] = f"v26-core-{selector}-32-p-{lr_tag(multiplier)}"
            dump_yaml(payload, CONFIGS / config_name)
            write_script(
                SCRIPTS / f"train_v26_core_{selector}_32_p_{lr_tag(multiplier)}.sh",
                config_name,
                f"v26-core-{selector}-32-p-{lr_tag(multiplier)}",
                smoke=False,
            )

        fallback_name = f"v26_core_{selector}_32_p_{lr_tag(0.6)}.yaml"
        payload = build_config(selector, "core", "p", lr_multiplier=0.6)
        payload["runtime"]["run_name"] = f"v26-core-{selector}-32-p-{lr_tag(0.6)}"
        dump_yaml(payload, CONFIGS / fallback_name)
        write_script(
            SCRIPTS / f"train_v26_core_{selector}_32_p_{lr_tag(0.6)}.sh",
            fallback_name,
            f"v26-core-{selector}-32-p-{lr_tag(0.6)}",
            smoke=False,
        )

        for regime in REGIMES:
            for schedule in ("s", "m", "l"):
                config_name = f"v26_{regime}_{selector}_32_{schedule}.yaml"
                dump_yaml(
                    build_config(selector, regime, schedule, lr_multiplier=selected_lrs[selector]),
                    CONFIGS / config_name,
                )
                write_script(
                    SCRIPTS / f"train_v26_{regime}_{selector}_32_{schedule}.sh",
                    config_name,
                    f"v26-{regime}-{selector}-32-{schedule}",
                    smoke=False,
                )
                if regime in {"core", "t1", "t2a"} and schedule == "s":
                    write_script(
                        SCRIPTS / f"smoke_v26_{regime}_{selector}_32_{schedule}.sh",
                        config_name,
                        f"v26-{regime}-{selector}-32-{schedule}",
                        smoke=True,
                    )


if __name__ == "__main__":
    main()
