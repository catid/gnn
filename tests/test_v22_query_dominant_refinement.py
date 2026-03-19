from __future__ import annotations

from apsgnn.config import load_config


def test_v22_selector_weight_configs_match_intended_design() -> None:
    q = load_config("configs/v22_core_querygradonly_32_l.yaml")
    vt_half = load_config("configs/v22_core_visit_taskgrad_half_32_l.yaml")
    qv_quarter = load_config("configs/v22_core_querygrad_visit_quarter_32_l.yaml")
    qv_half = load_config("configs/v22_core_querygrad_visit_half_32_l.yaml")

    assert q.growth.utility_visit_weight == 0.0
    assert q.growth.utility_grad_weight == 0.0
    assert q.growth.utility_query_grad_weight == 1.0

    assert vt_half.growth.utility_visit_weight == 1.0
    assert vt_half.growth.utility_grad_weight == 0.5
    assert vt_half.growth.utility_query_grad_weight == 0.0

    assert qv_quarter.growth.utility_visit_weight == 0.25
    assert qv_quarter.growth.utility_grad_weight == 0.0
    assert qv_quarter.growth.utility_query_grad_weight == 1.0

    assert qv_half.growth.utility_visit_weight == 0.5
    assert qv_half.growth.utility_grad_weight == 0.0
    assert qv_half.growth.utility_query_grad_weight == 1.0


def test_v22_schedule_budget_matching_is_fair_within_phases() -> None:
    selectors = [
        "querygradonly",
        "visit_taskgrad_half",
        "querygrad_visit_quarter",
        "querygrad_visit_half",
    ]
    core_l = [load_config(f"configs/v22_core_{name}_32_l.yaml") for name in selectors]
    t1_l = [load_config(f"configs/v22_t1_{name}_32_l.yaml") for name in selectors]
    core_xl = [load_config(f"configs/v22_core_{name}_32_xl.yaml") for name in selectors]

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


def test_v22_transfer_and_rerun_configs_are_consistent() -> None:
    q_xl = load_config("configs/v22_t1_querygradonly_32_xl.yaml")
    q_r = load_config("configs/v22_t1_querygradonly_32_r.yaml")
    vt_half_t2a = load_config("configs/v22_t2a_visit_taskgrad_half_32_xl.yaml")

    assert q_xl.task == q_r.task
    assert q_xl.growth.stage_steps == q_r.growth.stage_steps
    assert q_xl.train.train_steps == q_r.train.train_steps
    assert q_xl.task.train_eval_writers == [4, 8, 12, 14]

    assert vt_half_t2a.task.start_node_pool_size == 1
    assert vt_half_t2a.task.writers_per_episode == 4
