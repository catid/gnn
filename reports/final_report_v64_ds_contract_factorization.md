# APSGNN v64: DS Contract Factorization

## What Changed From v63

v64 narrows the post-v63 question to DS factorization and stronger DS-like contracts. This round keeps the static selector bases fixed at `V` and `VT-0.5`, reuses the same 32-leaf APSGNN family, and tests whether the remaining instability is really in temporal credit assignment and training dynamics rather than selector weights.

## Budgets

- `P = 420`
- `S = 1134`
- `M = 2268`
- `L = 3024`
- visible GPUs used: `2`
- rolling late-stage window: `5`

## Exact Regimes

- `Core`, `T1`, `T2a`, `Hmix` for the main matrices
- `T1r`, `T2b`, `T2c`, `Hmid` for holdout verification

## DS-Core Pilot Screen

| Contract | LR x | p_keep_prev | Composite |
| --- | --- | --- | --- |
| DS-p0.10 | 1.00 | 0.10 | 0.0536 |
| DS-fixed2step | 1.00 | 1.00 | 0.0536 |
| D | 0.80 | 0.00 | 0.0524 |
| DS-p0.40 | 1.00 | 0.40 | 0.0524 |
| D | 1.00 | 0.00 | 0.0524 |
| DS-p0.05 | 0.80 | 0.05 | 0.0524 |
| DS-fixed1step | 0.80 | 0.00 | 0.0524 |
| DS-fixed1step | 1.00 | 0.00 | 0.0524 |
| DS-p0.10 | 0.80 | 0.10 | 0.0524 |
| DS-p0.25 | 0.80 | 0.25 | 0.0524 |
| DS-p0.25 | 1.00 | 0.25 | 0.0524 |
| D | 0.60 | 0.00 | 0.0518 |
| DS-p0.40 | 0.60 | 0.40 | 0.0518 |
| DS-fixed1step | 0.60 | 0.00 | 0.0518 |
| DS-p0.05 | 0.60 | 0.05 | 0.0518 |
| DS-p0.10 | 0.60 | 0.10 | 0.0518 |
| DS-p0.25 | 0.60 | 0.25 | 0.0518 |
| DS-p0.40 | 0.80 | 0.40 | 0.0424 |
| DS-p0.05 | 1.00 | 0.05 | 0.0424 |
| DS-fixed2step | 0.60 | 1.00 | 0.0411 |
| DS-fixed2step | 0.80 | 1.00 | 0.0411 |

Chosen DS-core contracts:

- best: `ds_p010`
- runner-up: `ds_fixed2step`

## Chosen Pair Settings

| Pair | LR x | p_keep_prev | Aux Final | Composite |
| --- | --- | --- | --- | --- |
| VT-0.5/D | 0.80 | 0.00 | 1.00 | 0.0524 |
| VT-0.5/DS+AuxAnneal | 1.00 | 0.10 | 0.50 | 0.0536 |
| VT-0.5/DS-core-best | 1.00 | 0.10 | 1.00 | 0.0536 |
| VT-0.5/DS-core-runner-up | 1.00 | 1.00 | 1.00 | 0.0536 |
| VT-0.5/DS+RandDepth | 0.80 | 0.10 | 1.00 | 0.0524 |
| V/D | 0.80 | 0.00 | 1.00 | 0.0524 |
| V/DS+AuxAnneal | 1.00 | 0.10 | 0.50 | 0.0536 |
| V/DS-core-best | 1.00 | 0.10 | 1.00 | 0.0536 |
| V/DS-core-runner-up | 1.00 | 1.00 | 1.00 | 0.0536 |
| V/DS+RandDepth | 0.80 | 0.10 | 1.00 | 0.0524 |

## Exact Configs Used

- `configs/v64_core_visit_taskgrad_half_d_32_m.yaml`
- `configs/v64_core_visit_taskgrad_half_d_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_d_32_s.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_auxanneal_025_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_auxanneal_050_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_auxanneal_32_s.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_core_best_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_core_best_32_s.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_core_runner_up_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_core_runner_up_32_s.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_randdepth_32_m.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_randdepth_32_p.yaml`
- `configs/v64_core_visit_taskgrad_half_ds_randdepth_32_s.yaml`
- `configs/v64_core_visitonly_d_32_m.yaml`
- `configs/v64_core_visitonly_d_32_p.yaml`
- `configs/v64_core_visitonly_d_32_s.yaml`
- `configs/v64_core_visitonly_ds_auxanneal_025_32_p.yaml`
- `configs/v64_core_visitonly_ds_auxanneal_050_32_p.yaml`
- `configs/v64_core_visitonly_ds_auxanneal_32_s.yaml`
- `configs/v64_core_visitonly_ds_core_best_32_p.yaml`
- `configs/v64_core_visitonly_ds_core_best_32_s.yaml`
- `configs/v64_core_visitonly_ds_core_runner_up_32_p.yaml`
- `configs/v64_core_visitonly_ds_core_runner_up_32_s.yaml`
- `configs/v64_core_visitonly_ds_fixed1step_32_p.yaml`
- `configs/v64_core_visitonly_ds_fixed2step_32_p.yaml`
- `configs/v64_core_visitonly_ds_p005_32_p.yaml`
- `configs/v64_core_visitonly_ds_p010_32_p.yaml`
- `configs/v64_core_visitonly_ds_p025_32_p.yaml`
- `configs/v64_core_visitonly_ds_p040_32_p.yaml`
- `configs/v64_core_visitonly_ds_randdepth_32_m.yaml`
- `configs/v64_core_visitonly_ds_randdepth_32_p.yaml`
- `configs/v64_core_visitonly_ds_randdepth_32_s.yaml`
- `configs/v64_hmid_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_hmid_visitonly_d_32_l.yaml`
- `configs/v64_hmix_visit_taskgrad_half_d_32_m.yaml`
- `configs/v64_hmix_visit_taskgrad_half_d_32_s.yaml`
- `configs/v64_hmix_visit_taskgrad_half_ds_core_runner_up_32_s.yaml`
- `configs/v64_hmix_visit_taskgrad_half_ds_randdepth_32_m.yaml`
- `configs/v64_hmix_visit_taskgrad_half_ds_randdepth_32_s.yaml`
- `configs/v64_hmix_visitonly_d_32_m.yaml`
- `configs/v64_hmix_visitonly_d_32_s.yaml`
- `configs/v64_hmix_visitonly_ds_core_runner_up_32_s.yaml`
- `configs/v64_hmix_visitonly_ds_randdepth_32_m.yaml`
- `configs/v64_hmix_visitonly_ds_randdepth_32_s.yaml`
- `configs/v64_t1_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_t1_visit_taskgrad_half_d_32_m.yaml`
- `configs/v64_t1_visit_taskgrad_half_d_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_d_32_s.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_auxanneal_025_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_auxanneal_050_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_auxanneal_32_s.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_core_best_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_core_best_32_s.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_core_runner_up_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_core_runner_up_32_s.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_randdepth_32_m.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_randdepth_32_p.yaml`
- `configs/v64_t1_visit_taskgrad_half_ds_randdepth_32_s.yaml`
- `configs/v64_t1_visitonly_d_32_l.yaml`
- `configs/v64_t1_visitonly_d_32_m.yaml`
- `configs/v64_t1_visitonly_d_32_p.yaml`
- `configs/v64_t1_visitonly_d_32_s.yaml`
- `configs/v64_t1_visitonly_ds_auxanneal_025_32_p.yaml`
- `configs/v64_t1_visitonly_ds_auxanneal_050_32_p.yaml`
- `configs/v64_t1_visitonly_ds_auxanneal_32_s.yaml`
- `configs/v64_t1_visitonly_ds_core_best_32_p.yaml`
- `configs/v64_t1_visitonly_ds_core_best_32_s.yaml`
- `configs/v64_t1_visitonly_ds_core_runner_up_32_p.yaml`
- `configs/v64_t1_visitonly_ds_core_runner_up_32_s.yaml`
- `configs/v64_t1_visitonly_ds_fixed1step_32_p.yaml`
- `configs/v64_t1_visitonly_ds_fixed2step_32_p.yaml`
- `configs/v64_t1_visitonly_ds_p005_32_p.yaml`
- `configs/v64_t1_visitonly_ds_p010_32_p.yaml`
- `configs/v64_t1_visitonly_ds_p025_32_p.yaml`
- `configs/v64_t1_visitonly_ds_p040_32_p.yaml`
- `configs/v64_t1_visitonly_ds_randdepth_32_m.yaml`
- `configs/v64_t1_visitonly_ds_randdepth_32_p.yaml`
- `configs/v64_t1_visitonly_ds_randdepth_32_s.yaml`
- `configs/v64_t1r_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_t1r_visitonly_d_32_l.yaml`
- `configs/v64_t2a_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_t2a_visit_taskgrad_half_d_32_m.yaml`
- `configs/v64_t2a_visit_taskgrad_half_d_32_s.yaml`
- `configs/v64_t2a_visit_taskgrad_half_ds_auxanneal_32_s.yaml`
- `configs/v64_t2a_visit_taskgrad_half_ds_core_best_32_s.yaml`
- `configs/v64_t2a_visit_taskgrad_half_ds_core_runner_up_32_s.yaml`
- `configs/v64_t2a_visit_taskgrad_half_ds_randdepth_32_m.yaml`
- `configs/v64_t2a_visit_taskgrad_half_ds_randdepth_32_s.yaml`
- `configs/v64_t2a_visitonly_d_32_l.yaml`
- `configs/v64_t2a_visitonly_d_32_m.yaml`
- `configs/v64_t2a_visitonly_d_32_s.yaml`
- `configs/v64_t2a_visitonly_ds_auxanneal_32_s.yaml`
- `configs/v64_t2a_visitonly_ds_core_best_32_s.yaml`
- `configs/v64_t2a_visitonly_ds_core_runner_up_32_s.yaml`
- `configs/v64_t2a_visitonly_ds_randdepth_32_m.yaml`
- `configs/v64_t2a_visitonly_ds_randdepth_32_s.yaml`
- `configs/v64_t2b_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_t2b_visitonly_d_32_l.yaml`
- `configs/v64_t2c_visit_taskgrad_half_d_32_l.yaml`
- `configs/v64_t2c_visitonly_d_32_l.yaml`

## Screening Summary

| Pair | Dense | Last | Last5 | Drop | Composite |
| --- | --- | --- | --- | --- | --- |
| VT-0.5/DS-core-runner-up | 0.0405 | 0.0694 | 0.0694 | 0.0057 | 0.0466 |
| V/DS-core-runner-up | 0.0405 | 0.0694 | 0.0694 | 0.0057 | 0.0466 |
| VT-0.5/D | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| V/D | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| VT-0.5/DS+RandDepth | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| V/DS+RandDepth | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| V/DS-core-best | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |
| VT-0.5/DS-core-best | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |
| VT-0.5/DS+AuxAnneal | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |
| V/DS+AuxAnneal | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |

### Contract Ranking

| Contract | Composite | Std |
| --- | --- | --- |
| DS-core-runner-up | 0.0466 | 0.0000 |
| D | 0.0384 | 0.0000 |
| DS+RandDepth | 0.0384 | 0.0000 |
| DS-core-best | 0.0381 | 0.0000 |
| DS+AuxAnneal | 0.0381 | 0.0000 |

Promoted contracts to `Hmix` tiebreak: `ds_core_runner_up, d, ds_randdepth`

## Hmix Contract Tiebreak

| Contract | Dense | Last | Composite |
| --- | --- | --- | --- |
| DS-core-runner-up | 0.0368 | 0.0000 | 0.0147 |
| D | 0.0372 | 0.0417 | 0.0324 |
| DS+RandDepth | 0.0366 | 0.0417 | 0.0334 |

Top 2 contracts after `Hmix`: `ds_randdepth, d`

## Confirmation Summary

| Pair | Dense | Last | Last5 | Drop | Composite |
| --- | --- | --- | --- | --- | --- |
| VT-0.5/D | 0.0320 | 0.0365 | 0.0355 | 0.0269 | 0.0250 |
| V/D | 0.0313 | 0.0365 | 0.0355 | 0.0269 | 0.0247 |
| V/DS+RandDepth | 0.0326 | 0.0312 | 0.0323 | 0.0260 | 0.0234 |
| VT-0.5/DS+RandDepth | 0.0320 | 0.0312 | 0.0323 | 0.0260 | 0.0231 |

## Holdout Verification

| Regime | Pair | Dense | Last | Composite |
| --- | --- | --- | --- | --- |
| t1r | VT-0.5/D | 0.0339 | 0.0417 | 0.0323 |
| t1r | V/D | 0.0339 | 0.0417 | 0.0323 |
| t2b | VT-0.5/D | 0.0365 | 0.0417 | 0.0333 |
| t2b | V/D | 0.0339 | 0.0417 | 0.0323 |
| t2c | VT-0.5/D | 0.0417 | 0.0417 | 0.0354 |
| t2c | V/D | 0.0417 | 0.0417 | 0.0354 |
| hmid | VT-0.5/D | 0.0312 | 0.0000 | 0.0100 |
| hmid | V/D | 0.0352 | 0.0000 | 0.0103 |

## Extra Compute / Settling

| Pair | Regime | Settle Dense | Settle Rate | Steps | Accept-on-settle |
| --- | --- | --- | --- | --- | --- |
| VT-0.5/D | core | 0.0391 | 1.0000 | 2.00 | 0.0312 |
| VT-0.5/D | t1 | 0.0443 | 1.0000 | 2.00 | 0.0391 |
| VT-0.5/D | t2a | 0.0182 | 1.0000 | 2.00 | 0.0312 |
| VT-0.5/D | hmix | 0.0312 | 1.0000 | 2.00 | 0.0234 |
| V/D | core | 0.0391 | 1.0000 | 2.00 | 0.0312 |
| V/D | t1 | 0.0443 | 1.0000 | 2.00 | 0.0391 |
| V/D | t2a | 0.0182 | 1.0000 | 2.00 | 0.0312 |
| V/D | hmix | 0.0312 | 1.0000 | 2.00 | 0.0234 |

## Fresh Reruns

Chosen rerun regimes: `t1, t2a`

| Regime | Pair | Composite |
| --- | --- | --- |
| t1 | VT-0.5/D | 0.0267 |
| t1 | V/D | 0.0267 |
| t2a | VT-0.5/D | 0.0279 |
| t2a | V/D | 0.0287 |

## Ambiguity Breaker

Not triggered yet.

## Final Diagnosis

- outcome: `universal_selector_under_contract`
- winner: `visit_taskgrad_half_d`

## Best Next Experiment

If the result is still unresolved, keep the best DS-like contract family and continue refining temporal credit assignment rather than reopening selector-family search.
