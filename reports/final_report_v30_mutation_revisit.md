# APSGNN v30: Mutation Revisit on VT-0.5

## What Changed

v30 revisits the older mutation ideas after the selector line stabilized around `VT-0.5`.
The baseline stays mutation-free `VT-0.5`, and the mutation variants are narrow rebases of the old high-confidence, component-agreement, and stagnation gates.

## Selectors

- `VT-0.5`: clone baseline
- `VT-0.5 HCM`: high-confidence mutation gate
- `VT-0.5 CAG`: component-agreement mutation gate
- `VT-0.5 SGM`: stagnation-gated mutation gate

## Promotion Rule

The mutation arms are ranked on `Core-L` using the same screening composite as the selector funnel:

- `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`
- only the strongest mutation arm is promoted against the mutation-free baseline

## Completed Runs

| Phase | Sel | Seed | Best | Last | Last5 | Dense | Comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-L Screen | VT-0.5 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Screen | VT-0.5 | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-L Screen | VT-0.5 CAG | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0469 | 0.0440 |
| Core-L Screen | VT-0.5 CAG | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-L Screen | VT-0.5 HCM | 1234 | 0.1250 | 0.0417 | 0.0250 | 0.0391 | 0.0372 |
| Core-L Screen | VT-0.5 HCM | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-L Screen | VT-0.5 SGM | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Screen | VT-0.5 SGM | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 | 4234 | 0.1250 | 0.0417 | 0.0667 | 0.0547 | 0.0525 |
| Core-XL Confirmation | VT-0.5 CAG | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 CAG | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| T1-L Transfer | VT-0.5 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Transfer | VT-0.5 | 2234 | 0.1250 | 0.0000 | 0.0167 | 0.0365 | 0.0197 |
| T1-L Transfer | VT-0.5 CAG | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Transfer | VT-0.5 CAG | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0469 | 0.0586 |
| T1-XL Confirmation | VT-0.5 | 4234 | 0.0833 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T1-XL Confirmation | VT-0.5 CAG | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0573 | 0.0633 |
| T1-XL Confirmation | VT-0.5 CAG | 4234 | 0.0833 | 0.0417 | 0.0250 | 0.0625 | 0.0477 |
| T2a-XL Stress | VT-0.5 | 5234 | 0.0417 | 0.0000 | 0.0083 | 0.0573 | 0.0274 |
| T2a-XL Stress | VT-0.5 | 6234 | 0.0833 | 0.0417 | 0.0333 | 0.0312 | 0.0353 |
| T2a-XL Stress | VT-0.5 CAG | 5234 | 0.0833 | 0.0000 | 0.0250 | 0.0521 | 0.0284 |
| T2a-XL Stress | VT-0.5 CAG | 6234 | 0.0833 | 0.0417 | 0.0250 | 0.0417 | 0.0383 |

## Core-L Screening

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 HCM | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0352 ± 0.0055 | 0.0354 ± 0.0025 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0388 ± 0.0073 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 SGM | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |

Promoted mutation arm: `VT-0.5 CAG`

## Core-XL Confirmation

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0625 ± 0.0295 | 0.0625 ± 0.0110 | 0.0600 ± 0.0106 | 0.0703 ± 0.0110 | 0.0547 ± 0.0110 |
| VT-0.5 CAG | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |

## T1-L Transfer

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0239 ± 0.0059 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |

## T1-XL Confirmation

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0521 ± 0.0074 | 0.0520 ± 0.0094 | 0.0547 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |
| VT-0.5 CAG | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0599 ± 0.0037 | 0.0555 ± 0.0110 | 0.0703 ± 0.0110 | 0.0469 ± 0.0000 | 0.0625 ± 0.0221 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0625 ± 0.0295 | 0.0208 ± 0.0295 | 0.0443 ± 0.0184 | 0.0314 ± 0.0056 | 0.0547 ± 0.0110 | 0.0312 ± 0.0221 | 0.0469 ± 0.0442 |
| VT-0.5 CAG | 0.0833 ± 0.0000 | 0.0208 ± 0.0295 | 0.0469 ± 0.0074 | 0.0334 ± 0.0070 | 0.0469 ± 0.0221 | 0.0391 ± 0.0110 | 0.0547 ± 0.0331 |

## Current Recommendation

Current best supported default: `VT-0.5 CAG`.

This recommendation updates automatically from whichever completed phases exist. If only `Core-L` is complete, treat it as a screen rather than a final decision.

## Outputs

- Summary JSON: [summary_metrics_v30.json](/home/catid/gnn/reports/summary_metrics_v30.json)
- Report: [final_report_v30_mutation_revisit.md](/home/catid/gnn/reports/final_report_v30_mutation_revisit.md)
- Plots: [reports](/home/catid/gnn/reports)
