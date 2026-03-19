#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


MODEL_BASE = {
    "d_model": 128,
    "address_dim": 128,
    "nhead": 4,
    "delay_bins": 8,
    "cache_capacity": 64,
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
}

SELECTORS = {
    "visitonly": {
        "utility_visit_weight": 1.0,
        "utility_grad_weight": 0.0,
        "utility_query_grad_weight": 0.0,
    },
    "visit_taskgrad": {
        "utility_visit_weight": 1.0,
        "utility_grad_weight": 1.0,
        "utility_query_grad_weight": 0.0,
    },
    "visit_querygrad": {
        "utility_visit_weight": 1.0,
        "utility_grad_weight": 0.0,
        "utility_query_grad_weight": 1.0,
    },
    "full_querygrad": {
        "utility_visit_weight": 1.0,
        "utility_grad_weight": 1.0,
        "utility_query_grad_weight": 1.0,
    },
    "querygradonly": {
        "utility_visit_weight": 0.0,
        "utility_grad_weight": 0.0,
        "utility_query_grad_weight": 1.0,
    },
}

SCALE_SPECS = {
    32: {
        "nodes_total": 33,
        "max_ttl": 32,
        "stage_active_counts": [4, 6, 8, 12, 16, 24, 32],
        "s_steps": [150, 150, 200, 200, 250, 350, 1700],
        "m_steps": [150, 150, 200, 200, 250, 350, 3200],
        "l_steps": [150, 150, 200, 200, 250, 350, 4700],
        "core_eval_writers": [4, 8, 12, 16],
        "t1_eval_writers": [6, 10, 14, 18],
    },
    40: {
        "nodes_total": 41,
        "max_ttl": 40,
        "stage_active_counts": [4, 6, 8, 12, 16, 24, 32, 40],
        "s_steps": [100, 100, 150, 150, 200, 250, 350, 1700],
        "m_steps": [100, 100, 150, 150, 200, 250, 350, 3200],
        "l_steps": [100, 100, 150, 150, 200, 250, 350, 4700],
        "core_eval_writers": [4, 8, 12, 16],
        "t1_eval_writers": [6, 10, 14, 18],
    },
}

GATE_SPECS = {
    64: {
        "nodes_total": 65,
        "max_ttl": 64,
        "stage_active_counts": [4, 6, 8, 12, 16, 24, 32, 48, 64],
        "s_steps": [80, 80, 120, 120, 160, 200, 260, 300, 1680],
        "eval_writers": [4, 8, 12, 16],
    },
    48: {
        "nodes_total": 49,
        "max_ttl": 48,
        "stage_active_counts": [4, 6, 8, 12, 16, 24, 32, 48],
        "s_steps": [100, 100, 150, 150, 200, 250, 350, 1700],
        "eval_writers": [4, 8, 12, 16],
    },
}


def dump_yaml(payload: dict, path: Path) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def build_config(
    *,
    nodes_total: int,
    max_ttl: int,
    writers: int,
    eval_writers: list[int],
    stage_active_counts: list[int],
    stage_steps: list[int],
    train_steps: int,
    eval_interval: int,
    save_interval: int,
    log_interval: int,
    anneal_steps: int,
    run_name: str,
    start_node_pool_size: int = 2,
    selector: str = "visitonly",
) -> dict:
    payload = {
        "model": {
            "nodes_total": nodes_total,
            "max_ttl": max_ttl,
            **MODEL_BASE,
        },
        "task": {
            "name": "memory_growth",
            "writers_per_episode": writers,
            "start_node_pool_size": start_node_pool_size,
            "query_ttl_min": 2,
            "query_ttl_max": 4,
            "max_rollout_steps": 14,
            "train_eval_writers": eval_writers,
        },
        "train": {
            **TRAIN_BASE,
            "train_steps": train_steps,
            "eval_interval": eval_interval,
            "log_interval": log_interval,
            "save_interval": save_interval,
            "first_hop_teacher_force_anneal_steps": anneal_steps,
        },
        "runtime": {
            "run_name": run_name,
        },
        "growth": {
            **GROWTH_BASE,
            **SELECTORS[selector],
            "stage_active_counts": stage_active_counts,
            "stage_steps": stage_steps,
        },
    }
    return payload


def main() -> None:
    CONFIGS.mkdir(parents=True, exist_ok=True)

    for scale, spec in GATE_SPECS.items():
        payload = build_config(
            nodes_total=spec["nodes_total"],
            max_ttl=spec["max_ttl"],
            writers=4,
            eval_writers=spec["eval_writers"],
            stage_active_counts=spec["stage_active_counts"],
            stage_steps=spec["s_steps"],
            train_steps=3000,
            eval_interval=150,
            save_interval=150,
            log_interval=50,
            anneal_steps=600,
            run_name=f"v20-gate-core-visitonly-{scale}",
            selector="visitonly",
        )
        dump_yaml(payload, CONFIGS / f"v20_gate_core_visitonly_{scale}.yaml")

    for scale, spec in SCALE_SPECS.items():
        for schedule_name, train_steps, stage_steps, eval_interval, save_interval, anneal_steps in [
            ("s", 3000, spec["s_steps"], 150, 150, 600),
            ("m", 4500, spec["m_steps"], 225, 225, 900),
            ("l", 6000, spec["l_steps"], 300, 300, 1200),
        ]:
            for selector in SELECTORS:
                core = build_config(
                    nodes_total=spec["nodes_total"],
                    max_ttl=spec["max_ttl"],
                    writers=4,
                    eval_writers=copy.deepcopy(spec["core_eval_writers"]),
                    stage_active_counts=copy.deepcopy(spec["stage_active_counts"]),
                    stage_steps=copy.deepcopy(stage_steps),
                    train_steps=train_steps,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    log_interval=50,
                    anneal_steps=anneal_steps,
                    run_name=f"v20-core-{selector}-{scale}-{schedule_name}",
                    selector=selector,
                )
                dump_yaml(core, CONFIGS / f"v20_core_{selector}_{scale}_{schedule_name}.yaml")

                t1 = build_config(
                    nodes_total=spec["nodes_total"],
                    max_ttl=spec["max_ttl"],
                    writers=6,
                    eval_writers=copy.deepcopy(spec["t1_eval_writers"]),
                    stage_active_counts=copy.deepcopy(spec["stage_active_counts"]),
                    stage_steps=copy.deepcopy(stage_steps),
                    train_steps=train_steps,
                    eval_interval=eval_interval,
                    save_interval=save_interval,
                    log_interval=50,
                    anneal_steps=anneal_steps,
                    run_name=f"v20-transfer-t1-{selector}-{scale}-{schedule_name}",
                    selector=selector,
                )
                dump_yaml(t1, CONFIGS / f"v20_transfer_t1_{selector}_{scale}_{schedule_name}.yaml")

                if schedule_name == "l":
                    t2a = build_config(
                        nodes_total=spec["nodes_total"],
                        max_ttl=spec["max_ttl"],
                        writers=6,
                        eval_writers=copy.deepcopy(spec["t1_eval_writers"]),
                        stage_active_counts=copy.deepcopy(spec["stage_active_counts"]),
                        stage_steps=copy.deepcopy(stage_steps),
                        train_steps=train_steps,
                        eval_interval=eval_interval,
                        save_interval=save_interval,
                        log_interval=50,
                        anneal_steps=anneal_steps,
                        run_name=f"v20-transfer-t2a-{selector}-{scale}-{schedule_name}",
                        start_node_pool_size=1,
                        selector=selector,
                    )
                    dump_yaml(t2a, CONFIGS / f"v20_transfer_t2a_{selector}_{scale}_{schedule_name}.yaml")


if __name__ == "__main__":
    main()
