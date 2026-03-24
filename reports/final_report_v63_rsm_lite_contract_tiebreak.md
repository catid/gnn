# APSGNN v63: RSM-lite Contract Tie-Break

## What Changed From v62

v63 stops slicing selector weights and instead tests whether temporal-credit and train-shallow/test-deep contracts can stabilize the unresolved `V` vs `VT-0.5` tie from v62.

## Budgets

- `P = 420`
- `S = 1134`
- `M = 2268`
- `L = 3024`
- visible GPUs used: `2`
- rolling late-stage window: `5` evals

## Calibration Settings

| Pair | LR x | p_keep_prev | Pilot Composite |
| --- | --- | --- | --- |
| VT-0.5/B | 0.80 | 0.00 | 0.0517 |
| VT-0.5/D | 0.80 | 0.00 | 0.0524 |
| VT-0.5/DS | 1.00 | 0.10 | 0.0536 |
| VT-0.5/DSG | 0.60 | 0.10 | 0.0518 |
| V/B | 0.80 | 0.00 | 0.0517 |
| V/D | 0.80 | 0.00 | 0.0524 |
| V/DS | 1.00 | 0.10 | 0.0536 |
| V/DSG | 0.60 | 0.10 | 0.0518 |

## Screening Summary

| Pair | Dense | Last | Last5 | Drop | Composite |
| --- | --- | --- | --- | --- | --- |
| VT-0.5/DSG | 0.0405 | 0.0556 | 0.0556 | 0.0175 | 0.0386 |
| V/DSG | 0.0405 | 0.0556 | 0.0556 | 0.0175 | 0.0386 |
| V/D | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| VT-0.5/D | 0.0394 | 0.0556 | 0.0556 | 0.0159 | 0.0384 |
| VT-0.5/DS | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |
| V/DS | 0.0394 | 0.0556 | 0.0557 | 0.0175 | 0.0381 |
| VT-0.5/B | 0.0307 | 0.0694 | 0.0639 | 0.0972 | 0.0281 |
| V/B | 0.0307 | 0.0694 | 0.0639 | 0.0972 | 0.0281 |

Promoted pairs: `visit_taskgrad_half_dsg, visitonly_dsg, visitonly_d, visit_taskgrad_half_d`

## Confirmation Summary

| Pair | Anchor Total | Dense | Last | Drop |
| --- | --- | --- | --- | --- |
| VT-0.5/DSG | 0.1087 | 0.0333 | 0.0556 | 0.0139 |
| V/DSG | 0.1087 | 0.0333 | 0.0556 | 0.0139 |
| VT-0.5/D | 0.0895 | 0.0313 | 0.0486 | 0.0290 |
| V/D | 0.0885 | 0.0304 | 0.0486 | 0.0290 |

## Holdout Verification

| Pair | Holdout Total |
| --- | --- |
| VT-0.5/DSG | 0.1444 |
| V/DSG | 0.1402 |

## Extra Compute / Settling

| Pair | Hmix settle dense | Settle Rate | Steps |
| --- | --- | --- | --- |
| VT-0.5/DSG | 0.0273 | 1.0000 | 2.00 |
| V/DSG | 0.0234 | 1.0000 | 2.00 |

## Fresh Reruns

| Regime | Pair | Dense | Last | Composite |
| --- | --- | --- | --- | --- |
| core | VT-0.5/DSG | 0.0219 | 0.0000 | 0.0087 |
| core | V/DSG | 0.0219 | 0.0000 | 0.0087 |
| t1 | VT-0.5/DSG | 0.0208 | 0.0417 | 0.0258 |
| t1 | V/DSG | 0.0208 | 0.0417 | 0.0258 |

## Ambiguity Breaker

| Regime | Pair | Dense | Last | Composite |
| --- | --- | --- | --- | --- |
| core | VT-0.5/DSG | 0.0281 | 0.0417 | 0.0300 |
| core | V/DSG | 0.0281 | 0.0417 | 0.0300 |

## Final Diagnosis

- Outcome: `unresolved`
- Top pair: `visit_taskgrad_half_dsg` score `0.3177`
- Runner-up: `visitonly_dsg` score `0.3135`
