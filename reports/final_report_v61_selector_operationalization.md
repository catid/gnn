# APSGNN v61: Selector Operationalization

## What Changed From v60

v61 keeps the v60 static conclusion under direct matched re-check, then uses the other half of the budget to test four non-oracle gates that choose between `V` and `VT-0.5` without using the regime label directly.

This round is split 50/50 between exploitation and exploration so the static v60 answer gets a clean re-check while adaptive gates still receive a fair enough screening matrix and fresh holdout verification before being discarded.

## Budgets

- `P = 420`
- `S = 1260`
- `M = 2520`
- `L = 3360`
- visible GPUs used: `2`
- rolling late-stage window: `5` evals

## Exact Regimes

- Known: `Core`, `T1`, `T1r`, `T2a`, `T2b`, `T2c`
- Holdouts: `Hmid`, `Hmix`
- Holdout configs are distinct from the known six because they use writer density `3` and eval densities `3/7/11`, with `Hmix` also setting `start_node_pool_size = 1`.

## Calibration Summary

| Family | Chosen Variant | LR x | Pilot Composite | Dense | Last |
| --- | --- | --- | --- | --- | --- |
| G_ingress | G_ingress(pool1) | 0.8 | 0.0581 | 0.0000 | 0.1042 |
| G_meta | G_meta(A) | 0.8 | 0.0581 | 0.0000 | 0.1042 |
| G_online | G_online(A) | 0.8 | 0.0581 | 0.0000 | 0.1042 |
| G_writers | G_writers<=2 | 0.8 | 0.0581 | 0.0000 | 0.1042 |
| VT-0.5 | VT-0.5 | 0.8 | 0.0581 | 0.0000 | 0.1042 |
| V | V | 0.8 | 0.0581 | 0.0000 | 0.1042 |

Chosen settings after pilot:

| Family | Chosen Arm | Chosen Setting | LR x |
| --- | --- | --- | --- |
| G_ingress | G_ingress(pool1) | Use `VT-0.5` when `start_node_pool_size == 1`, else `V` | 0.8 |
| G_meta | G_meta(A) | Metadata linear gate A on writers / ingress / tight TTL | 0.8 |
| G_online | G_online(A) | Early-stage online gate A using visit entropy / Gini after stage 4 | 0.8 |
| G_writers | G_writers<=2 | Use `VT-0.5` when writers <= 2, else `V` | 0.8 |
| VT-0.5 | VT-0.5 | Static `z(task_visits) + 0.5*z(task_grad)` | 0.8 |
| V | V | Static `z(task_visits)` | 0.8 |

## Exact Configs Used

- `configs/v61_core_gate_ingress_pool1_32_p.yaml`
- `configs/v61_core_gate_ingress_pool1_32_s.yaml`
- `configs/v61_core_gate_ingress_pool1_or_tightttl_32_p.yaml`
- `configs/v61_core_gate_meta_a_32_p.yaml`
- `configs/v61_core_gate_meta_a_32_s.yaml`
- `configs/v61_core_gate_meta_b_32_p.yaml`
- `configs/v61_core_gate_online_a_32_p.yaml`
- `configs/v61_core_gate_online_a_32_s.yaml`
- `configs/v61_core_gate_online_b_32_p.yaml`
- `configs/v61_core_gate_writers_le2_32_p.yaml`
- `configs/v61_core_gate_writers_le2_32_s.yaml`
- `configs/v61_core_gate_writers_le3_32_p.yaml`
- `configs/v61_core_visit_taskgrad_half_32_m.yaml`
- `configs/v61_core_visit_taskgrad_half_32_p.yaml`
- `configs/v61_core_visitonly_32_m.yaml`
- `configs/v61_core_visitonly_32_p.yaml`
- `configs/v61_hmid_gate_ingress_pool1_32_l.yaml`
- `configs/v61_hmid_visit_taskgrad_half_32_l.yaml`
- `configs/v61_hmid_visitonly_32_l.yaml`
- `configs/v61_hmix_gate_ingress_pool1_32_l.yaml`
- `configs/v61_hmix_visit_taskgrad_half_32_l.yaml`
- `configs/v61_hmix_visitonly_32_l.yaml`
- `configs/v61_t1_gate_ingress_pool1_32_p.yaml`
- `configs/v61_t1_gate_ingress_pool1_32_s.yaml`
- `configs/v61_t1_gate_ingress_pool1_or_tightttl_32_p.yaml`
- `configs/v61_t1_gate_meta_a_32_p.yaml`
- `configs/v61_t1_gate_meta_a_32_s.yaml`
- `configs/v61_t1_gate_meta_b_32_p.yaml`
- `configs/v61_t1_gate_online_a_32_p.yaml`
- `configs/v61_t1_gate_online_a_32_s.yaml`
- `configs/v61_t1_gate_online_b_32_p.yaml`
- `configs/v61_t1_gate_writers_le2_32_p.yaml`
- `configs/v61_t1_gate_writers_le2_32_s.yaml`
- `configs/v61_t1_gate_writers_le3_32_p.yaml`
- `configs/v61_t1_visit_taskgrad_half_32_m.yaml`
- `configs/v61_t1_visit_taskgrad_half_32_p.yaml`
- `configs/v61_t1_visitonly_32_m.yaml`
- `configs/v61_t1_visitonly_32_p.yaml`
- `configs/v61_t1r_visit_taskgrad_half_32_m.yaml`
- `configs/v61_t1r_visitonly_32_m.yaml`
- `configs/v61_t2a_gate_ingress_pool1_32_s.yaml`
- `configs/v61_t2a_gate_meta_a_32_s.yaml`
- `configs/v61_t2a_gate_online_a_32_s.yaml`
- `configs/v61_t2a_gate_writers_le2_32_s.yaml`
- `configs/v61_t2a_visit_taskgrad_half_32_m.yaml`
- `configs/v61_t2a_visitonly_32_m.yaml`
- `configs/v61_t2b_visit_taskgrad_half_32_m.yaml`
- `configs/v61_t2b_visitonly_32_m.yaml`
- `configs/v61_t2c_visit_taskgrad_half_32_m.yaml`
- `configs/v61_t2c_visitonly_32_m.yaml`

## Completed Runs

| Sched | Regime | Arm | Seed | LR x | Best | Last | Dense | Composite |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| L | hmid | G_ingress(pool1) | 3234 | 0.8 | 0.0417 | 0.0417 | 0.0430 | 0.0356 |
| L | hmid | G_ingress(pool1) | 4234 | 0.8 | 0.0833 | 0.0000 | 0.0508 | 0.0279 |
| L | hmid | VT-0.5 | 3234 | 0.8 | 0.0417 | 0.0000 | 0.0352 | 0.0158 |
| L | hmid | VT-0.5 | 4234 | 0.8 | 0.0417 | 0.0000 | 0.0430 | 0.0210 |
| L | hmid | V | 3234 | 0.8 | 0.0417 | 0.0417 | 0.0430 | 0.0356 |
| L | hmid | V | 4234 | 0.8 | 0.0833 | 0.0000 | 0.0508 | 0.0279 |
| L | hmix | G_ingress(pool1) | 3234 | 0.8 | 0.0435 | 0.0000 | 0.0547 | 0.0246 |
| L | hmix | G_ingress(pool1) | 4234 | 0.8 | 0.0417 | 0.0000 | 0.0469 | 0.0211 |
| L | hmix | VT-0.5 | 3234 | 0.8 | 0.0435 | 0.0000 | 0.0547 | 0.0246 |
| L | hmix | VT-0.5 | 4234 | 0.8 | 0.0417 | 0.0000 | 0.0469 | 0.0211 |
| L | hmix | V | 3234 | 0.8 | 0.0417 | 0.0000 | 0.0469 | 0.0211 |
| L | hmix | V | 4234 | 0.8 | 0.0417 | 0.0417 | 0.0430 | 0.0373 |
| M | core | VT-0.5 | 1234 | 0.8 | 0.1667 | 0.0833 | 0.0156 | 0.0495 |
| M | core | VT-0.5 | 2234 | 0.8 | 0.0833 | 0.0417 | 0.0521 | 0.0497 |
| M | core | V | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0260 | 0.0884 |
| M | core | V | 2234 | 0.8 | 0.1250 | 0.0417 | 0.0469 | 0.0390 |
| M | t1 | VT-0.5 | 1234 | 0.8 | 0.1667 | 0.0000 | 0.0243 | 0.0276 |
| M | t1 | VT-0.5 | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0347 | 0.0173 |
| M | t1 | V | 1234 | 0.8 | 0.1667 | 0.0000 | 0.0347 | 0.0306 |
| M | t1 | V | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0347 | 0.0173 |
| M | t1r | VT-0.5 | 1234 | 0.8 | 0.1667 | 0.0000 | 0.0243 | 0.0276 |
| M | t1r | VT-0.5 | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0347 | 0.0173 |
| M | t1r | V | 1234 | 0.8 | 0.1667 | 0.0000 | 0.0347 | 0.0306 |
| M | t1r | V | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0347 | 0.0173 |
| M | t2a | VT-0.5 | 1234 | 0.8 | 0.0833 | 0.0417 | 0.0347 | 0.0369 |
| M | t2a | VT-0.5 | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0382 | 0.0222 |
| M | t2a | V | 1234 | 0.8 | 0.0417 | 0.0000 | 0.0278 | 0.0125 |
| M | t2a | V | 2234 | 0.8 | 0.0833 | 0.0000 | 0.0451 | 0.0220 |
| M | t2b | VT-0.5 | 1234 | 0.8 | 0.1250 | 0.0000 | 0.0486 | 0.0269 |
| M | t2b | VT-0.5 | 2234 | 0.8 | 0.1667 | 0.0417 | 0.0139 | 0.0408 |
| M | t2b | V | 1234 | 0.8 | 0.1250 | 0.0000 | 0.0417 | 0.0221 |
| M | t2b | V | 2234 | 0.8 | 0.1667 | 0.0417 | 0.0139 | 0.0408 |
| M | t2c | VT-0.5 | 1234 | 0.8 | 0.1250 | 0.0833 | 0.0174 | 0.0536 |
| M | t2c | VT-0.5 | 2234 | 0.8 | 0.0417 | 0.0417 | 0.0312 | 0.0303 |
| M | t2c | V | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0208 | 0.0406 |
| M | t2c | V | 2234 | 0.8 | 0.0417 | 0.0417 | 0.0243 | 0.0272 |
| P | core | G_ingress(pool1) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_ingress(pool1) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_ingress(pool1|tightttl) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_ingress(pool1|tightttl) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_meta(A) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_meta(A) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_meta(B) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_meta(B) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_online(A) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_online(A) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_online(B) | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_online(B) | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_writers<=2 | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_writers<=2 | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | G_writers<=3 | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | G_writers<=3 | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | VT-0.5 | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | VT-0.5 | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | core | V | 1234 | 0.8 | 0.2083 | 0.1667 | 0.0000 | 0.0900 |
| P | core | V | 1234 | 1.0 | 0.1667 | 0.0417 | 0.0391 | 0.0555 |
| P | t1 | G_ingress(pool1) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_ingress(pool1) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_ingress(pool1|tightttl) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_ingress(pool1|tightttl) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_meta(A) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_meta(A) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_meta(B) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_meta(B) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_online(A) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_online(A) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_online(B) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_online(B) | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_writers<=2 | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_writers<=2 | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | G_writers<=3 | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | G_writers<=3 | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | VT-0.5 | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | VT-0.5 | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| P | t1 | V | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0000 | 0.0262 |
| P | t1 | V | 1234 | 1.0 | 0.1667 | 0.0833 | 0.0208 | 0.0502 |
| S | core | G_ingress(pool1) | 1234 | 0.8 | 0.2500 | 0.0833 | 0.0521 | 0.0626 |
| S | core | G_ingress(pool1) | 2234 | 0.8 | 0.0833 | 0.0833 | 0.0573 | 0.0684 |
| S | core | G_meta(A) | 1234 | 0.8 | 0.2500 | 0.0833 | 0.0521 | 0.0626 |
| S | core | G_meta(A) | 2234 | 0.8 | 0.0833 | 0.0833 | 0.0573 | 0.0684 |
| S | core | G_online(A) | 1234 | 0.8 | 0.2500 | 0.0833 | 0.0521 | 0.0626 |
| S | core | G_online(A) | 2234 | 0.8 | 0.0833 | 0.0833 | 0.0573 | 0.0684 |
| S | core | G_writers<=2 | 1234 | 0.8 | 0.2500 | 0.0833 | 0.0521 | 0.0626 |
| S | core | G_writers<=2 | 2234 | 0.8 | 0.0833 | 0.0833 | 0.0573 | 0.0684 |
| S | t1 | G_ingress(pool1) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0278 | 0.0304 |
| S | t1 | G_ingress(pool1) | 2234 | 0.8 | 0.1250 | 0.0833 | 0.0139 | 0.0506 |
| S | t1 | G_meta(A) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0278 | 0.0304 |
| S | t1 | G_meta(A) | 2234 | 0.8 | 0.1250 | 0.0833 | 0.0139 | 0.0506 |
| S | t1 | G_online(A) | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0278 | 0.0304 |
| S | t1 | G_online(A) | 2234 | 0.8 | 0.1250 | 0.0833 | 0.0139 | 0.0506 |
| S | t1 | G_writers<=2 | 1234 | 0.8 | 0.1250 | 0.0417 | 0.0278 | 0.0304 |
| S | t1 | G_writers<=2 | 2234 | 0.8 | 0.1250 | 0.0833 | 0.0139 | 0.0506 |
| S | t2a | G_ingress(pool1) | 1234 | 0.8 | 0.0417 | 0.0000 | 0.0451 | 0.0203 |
| S | t2a | G_ingress(pool1) | 2234 | 0.8 | 0.1667 | 0.0435 | 0.0209 | 0.0381 |
| S | t2a | G_meta(A) | 1234 | 0.8 | 0.0417 | 0.0000 | 0.0451 | 0.0203 |
| S | t2a | G_meta(A) | 2234 | 0.8 | 0.1667 | 0.0435 | 0.0209 | 0.0381 |
| S | t2a | G_online(A) | 1234 | 0.8 | 0.0417 | 0.0000 | 0.0451 | 0.0203 |
| S | t2a | G_online(A) | 2234 | 0.8 | 0.1667 | 0.0435 | 0.0209 | 0.0381 |
| S | t2a | G_writers<=2 | 1234 | 0.8 | 0.0417 | 0.0000 | 0.0451 | 0.0203 |
| S | t2a | G_writers<=2 | 2234 | 0.8 | 0.1667 | 0.0435 | 0.0209 | 0.0381 |

## Exploitation Summary

| Regime | Arm | Best | Last | Dense | Composite |
| --- | --- | --- | --- | --- | --- |
| core | V | 0.1667 | 0.1042 | 0.0365 | 0.0637 |
| core | VT-0.5 | 0.1250 | 0.0625 | 0.0339 | 0.0496 |
| t1 | V | 0.1250 | 0.0000 | 0.0347 | 0.0240 |
| t1 | VT-0.5 | 0.1250 | 0.0000 | 0.0295 | 0.0224 |
| t1r | V | 0.1250 | 0.0000 | 0.0347 | 0.0240 |
| t1r | VT-0.5 | 0.1250 | 0.0000 | 0.0295 | 0.0224 |
| t2a | V | 0.0625 | 0.0000 | 0.0365 | 0.0172 |
| t2a | VT-0.5 | 0.0833 | 0.0208 | 0.0365 | 0.0295 |
| t2b | V | 0.1458 | 0.0208 | 0.0278 | 0.0315 |
| t2b | VT-0.5 | 0.1458 | 0.0208 | 0.0312 | 0.0339 |
| t2c | V | 0.0833 | 0.0417 | 0.0226 | 0.0339 |
| t2c | VT-0.5 | 0.0833 | 0.0625 | 0.0243 | 0.0420 |

Known-regime composite totals:
- `V everywhere`: `0.194219`
- `VT-0.5 everywhere`: `0.199870`
- `Known keyed rule`: `0.192422`

Known-regime winners:

| Regime | Winner | V Composite | VT-0.5 Composite |
| --- | --- | --- | --- |
| core | V | 0.0637 | 0.0496 |
| t1 | V | 0.0240 | 0.0224 |
| t1r | V | 0.0240 | 0.0224 |
| t2a | VT-0.5 | 0.0172 | 0.0295 |
| t2b | VT-0.5 | 0.0315 | 0.0339 |
| t2c | VT-0.5 | 0.0339 | 0.0420 |

## Exploration Summary

| Gate Family | Chosen Variant | Best | Last | Dense | Composite |
| --- | --- | --- | --- | --- | --- |
| G_ingress | G_ingress(pool1) | 0.1319 | 0.0559 | 0.0362 | 0.0451 |
| G_meta | G_meta(A) | 0.1319 | 0.0559 | 0.0362 | 0.0451 |
| G_online | G_online(A) | 0.1319 | 0.0559 | 0.0362 | 0.0451 |
| G_writers | G_writers<=2 | 0.1319 | 0.0559 | 0.0362 | 0.0451 |

## Holdout Verification

| Holdout | Arm | Best | Last | Dense | Composite |
| --- | --- | --- | --- | --- | --- |
| hmid | G_ingress(pool1) | 0.0625 | 0.0208 | 0.0469 | 0.0317 |
| hmid | VT-0.5 | 0.0417 | 0.0000 | 0.0391 | 0.0184 |
| hmid | V | 0.0625 | 0.0208 | 0.0469 | 0.0317 |
| hmix | G_ingress(pool1) | 0.0426 | 0.0000 | 0.0508 | 0.0229 |
| hmix | VT-0.5 | 0.0426 | 0.0000 | 0.0508 | 0.0229 |
| hmix | V | 0.0417 | 0.0208 | 0.0449 | 0.0292 |

| Holdout | Winner | Composite | Margin |
| --- | --- | --- | --- |
| hmid | G_ingress(pool1) | 0.0317 | 0.0000 |
| hmix | V | 0.0292 | 0.0063 |

## Best Gate

Top gate from screening: `G_ingress` with composite `0.045059`.

Holdout composite totals: `V 0.0609`, `G_ingress(pool1) 0.0546`, `VT-0.5 0.0413`.

## Final Diagnosis

- Best single static selector on the known-regime `M` matrix so far: `VT-0.5`.
- Home / ingress-stress specialist check: `core` -> V, `t2a` -> VT-0.5.
- Best non-oracle gate kept alive through holdouts: `G_ingress` using `Use `VT-0.5` when `start_node_pool_size == 1`, else `V``.
- Interpret the final recommendation from the combination of known-regime totals and the holdout verification totals above.

- Summary JSON: [summary_metrics_v61.json](/home/catid/gnn/reports/summary_metrics_v61.json)
- Report: [final_report_v61_selector_operationalization.md](/home/catid/gnn/reports/final_report_v61_selector_operationalization.md)
