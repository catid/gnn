from __future__ import annotations

from apsgnn.config import load_config


def test_v23_selector_weight_configs_match_intended_design() -> None:
    vt_quarter = load_config("configs/v23_core_visit_taskgrad_quarter_32_l.yaml")
    vt_half = load_config("configs/v23_core_visit_taskgrad_half_32_l.yaml")
    vt_threequarter = load_config("configs/v23_core_visit_taskgrad_threequarter_32_l.yaml")
    vt_half_q = load_config("configs/v23_core_visit_taskgrad_half_query_quarter_32_l.yaml")

    assert vt_quarter.growth.utility_visit_weight == 1.0
    assert vt_quarter.growth.utility_grad_weight == 0.25
    assert vt_quarter.growth.utility_query_grad_weight == 0.0

    assert vt_half.growth.utility_visit_weight == 1.0
    assert vt_half.growth.utility_grad_weight == 0.5
    assert vt_half.growth.utility_query_grad_weight == 0.0

    assert vt_threequarter.growth.utility_visit_weight == 1.0
    assert vt_threequarter.growth.utility_grad_weight == 0.75
    assert vt_threequarter.growth.utility_query_grad_weight == 0.0

    assert vt_half_q.growth.utility_visit_weight == 1.0
    assert vt_half_q.growth.utility_grad_weight == 0.5
    assert vt_half_q.growth.utility_query_grad_weight == 0.25


def test_v23_schedule_budget_matching_is_fair_within_phases() -> None:
    selectors = [
        "visit_taskgrad_quarter",
        "visit_taskgrad_half",
        "visit_taskgrad_threequarter",
        "visit_taskgrad_half_query_quarter",
    ]
    core_l = [load_config(f"configs/v23_core_{name}_32_l.yaml") for name in selectors]
    t1_l = [load_config(f"configs/v23_t1_{name}_32_l.yaml") for name in selectors]
    core_xl = [load_config(f"configs/v23_core_{name}_32_xl.yaml") for name in selectors]

    assert all(cfg.model == core_l[0].model for cfg in core_l[1:])
    assert all(cfg.task == core_l[0].task for cfg in core_l[1:])
    assert all(cfg.train == core_l[0].train for cfg in core_l[1:])
    assert core_l[0].growth.stage_steps == [120, 120, 150, 150, 180, 220, 2630]
    assert core_l[0].train.train_steps == 3570

    assert all(cfg.model == t1_l[0].model for cfg in t1_l[1:])
    assert all(cfg.task == t1_l[0].task for cfg in t1_l[1:])
    assert all(cfg.train == t1_l[0].train for cfg in t1_l[1:])
    assert t1_l[0].task.writers_per_episode == 4

    assert all(cfg.model == core_xl[0].model for cfg in core_xl[1:])
    assert all(cfg.task == core_xl[0].task for cfg in core_xl[1:])
    assert all(cfg.train == core_xl[0].train for cfg in core_xl[1:])
    assert core_xl[0].growth.stage_steps == [120, 120, 150, 150, 180, 220, 3650]
    assert core_xl[0].train.train_steps == 4590


def test_v23_transfer_and_rerun_configs_are_consistent() -> None:
    vt_half_xl = load_config("configs/v23_t1_visit_taskgrad_half_32_xl.yaml")
    vt_half_r = load_config("configs/v23_t1_visit_taskgrad_half_32_r.yaml")
    vt_threequarter_t2a = load_config("configs/v23_t2a_visit_taskgrad_threequarter_32_xl.yaml")

    assert vt_half_xl.task == vt_half_r.task
    assert vt_half_xl.growth.stage_steps == vt_half_r.growth.stage_steps
    assert vt_half_xl.train.train_steps == vt_half_r.train.train_steps
    assert vt_half_xl.task.train_eval_writers == [4, 8, 12, 14]

    assert vt_threequarter_t2a.task.start_node_pool_size == 1
    assert vt_threequarter_t2a.task.writers_per_episode == 4
