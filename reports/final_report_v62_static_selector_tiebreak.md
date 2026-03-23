# APSGNN v62: Static Selector Tie-Break

## What Changed From v61

v62 removes all gate logic and all adaptive-selector branches. This round exists solely to settle the static `V` vs `VT-0.5` tie-break on the same 32-leaf APSGNN family, with the full 8-regime matrix plus long-rerun anchors and an XL breaker only if the result stays ambiguous.

## Budgets

- `P = 420`
- `M = 2268`
- `L = 3024`
- `XL = 3780`
- visible GPUs used: `2`
- rolling late-stage window: `5` evals

## Exact Regimes

- `Core`: writers=2, start_pool=2, query_ttl=2..3, rollout=12, eval densities=2/6/10
- `T1`: writers=4, start_pool=2, query_ttl=2..3, rollout=12, eval densities=4/8/12/14
- `T1r`: writers=4, start_pool=2, query_ttl=2..3, rollout=12, eval densities=4/8/12/14
- `T2a`: writers=4, start_pool=1, query_ttl=2..3, rollout=12, eval densities=4/8/12/14
- `T2b`: writers=4, start_pool=2, query_ttl=2..2, rollout=12, eval densities=4/8/12/14
- `T2c`: writers=6, start_pool=2, query_ttl=2..3, rollout=12, eval densities=6/10/14/16
- `Hmid`: writers=3, start_pool=2, query_ttl=2..3, rollout=12, eval densities=3/7/11
- `Hmix`: writers=3, start_pool=1, query_ttl=2..3, rollout=12, eval densities=3/7/11

## Calibration Summary

| Selector | LR x | Composite | Dense | Last |
| --- | --- | --- | --- | --- |
| V | 1.0 | 0.0498 | 0.0288 | 0.0625 |
| VT-0.5 | 1.0 | 0.0498 | 0.0288 | 0.0625 |

## Completed Runs

| Schedule | Runs |
| --- | --- |
| P | 24 |
| M | 48 |
| L | 16 |
| XL | 8 |

## Pooled All-8-Regime Summary

| Selector | Main Total |
| --- | --- |
| V | 0.249757 |
| VT-0.5 | 0.243455 |

## Per-Regime Winners On M

| Regime | Winner | V Composite | VT-0.5 Composite |
| --- | --- | --- | --- |
| core | VT-0.5 | 0.0389 | 0.0404 |
| t1 | V | 0.0285 | 0.0285 |
| t1r | V | 0.0285 | 0.0285 |
| t2a | V | 0.0289 | 0.0289 |
| t2b | V | 0.0315 | 0.0315 |
| t2c | V | 0.0347 | 0.0347 |
| hmid | V | 0.0310 | 0.0241 |
| hmix | V | 0.0279 | 0.0269 |

## Long-Rerun Anchor Winners On L

| Regime | Winner | V Composite | VT-0.5 Composite |
| --- | --- | --- | --- |
| core | VT-0.5 | 0.0308 | 0.0477 |
| t1 | V | 0.0336 | 0.0278 |
| t2a | VT-0.5 | 0.0231 | 0.0370 |
| hmix | V | 0.0444 | 0.0424 |

## Ambiguity Breaker

- Triggered: `True`
- Candidate regimes: `t2a, t1`

| Regime | Winner | V Composite | VT-0.5 Composite |
| --- | --- | --- | --- |
| t1 | VT-0.5 | 0.0355 | 0.0448 |
| t2a | V | 0.0446 | 0.0343 |

## Coverage And Split Diagnostics

- Coverage deltas and split predictiveness were aggregated directly from `coverage_summary.json` for each run.
- The main diagnostic question is whether the selector gap is already visible by steps `10/50/100/200` or only emerges through later split usefulness and stability.

## Final Diagnosis

- Outcome: `D`
- Winner: `unresolved`
- Main totals: `V=0.249757`, `VT-0.5=0.243455`
- Anchor totals: `V=0.131921`, `VT-0.5=0.154805`

## Exact Configs Used

- `configs/v62_core_visit_taskgrad_half_32_l.yaml`
- `configs/v62_core_visit_taskgrad_half_32_m.yaml`
- `configs/v62_core_visit_taskgrad_half_32_p.yaml`
- `configs/v62_core_visitonly_32_l.yaml`
- `configs/v62_core_visitonly_32_m.yaml`
- `configs/v62_core_visitonly_32_p.yaml`
- `configs/v62_hmid_visit_taskgrad_half_32_m.yaml`
- `configs/v62_hmid_visitonly_32_m.yaml`
- `configs/v62_hmix_visit_taskgrad_half_32_l.yaml`
- `configs/v62_hmix_visit_taskgrad_half_32_m.yaml`
- `configs/v62_hmix_visit_taskgrad_half_32_p.yaml`
- `configs/v62_hmix_visitonly_32_l.yaml`
- `configs/v62_hmix_visitonly_32_m.yaml`
- `configs/v62_hmix_visitonly_32_p.yaml`
- `configs/v62_t1_visit_taskgrad_half_32_l.yaml`
- `configs/v62_t1_visit_taskgrad_half_32_m.yaml`
- `configs/v62_t1_visit_taskgrad_half_32_p.yaml`
- `configs/v62_t1_visit_taskgrad_half_32_xl.yaml`
- `configs/v62_t1_visitonly_32_l.yaml`
- `configs/v62_t1_visitonly_32_m.yaml`
- `configs/v62_t1_visitonly_32_p.yaml`
- `configs/v62_t1_visitonly_32_xl.yaml`
- `configs/v62_t1r_visit_taskgrad_half_32_m.yaml`
- `configs/v62_t1r_visitonly_32_m.yaml`
- `configs/v62_t2a_visit_taskgrad_half_32_l.yaml`
- `configs/v62_t2a_visit_taskgrad_half_32_m.yaml`
- `configs/v62_t2a_visit_taskgrad_half_32_p.yaml`
- `configs/v62_t2a_visit_taskgrad_half_32_xl.yaml`
- `configs/v62_t2a_visitonly_32_l.yaml`
- `configs/v62_t2a_visitonly_32_m.yaml`
- `configs/v62_t2a_visitonly_32_p.yaml`
- `configs/v62_t2a_visitonly_32_xl.yaml`
- `configs/v62_t2b_visit_taskgrad_half_32_m.yaml`
- `configs/v62_t2b_visitonly_32_m.yaml`
- `configs/v62_t2c_visit_taskgrad_half_32_m.yaml`
- `configs/v62_t2c_visitonly_32_m.yaml`

- Summary JSON: [summary_metrics_v62.json](/home/catid/gnn/reports/summary_metrics_v62.json)
- Report: [final_report_v62_static_selector_tiebreak.md](/home/catid/gnn/reports/final_report_v62_static_selector_tiebreak.md)
