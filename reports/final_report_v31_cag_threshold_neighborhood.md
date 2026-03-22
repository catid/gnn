# APSGNN v31: CAG Mutation Threshold Neighborhood

## What Changed

v31 follows the v30 mutation revisit by focusing only on the surviving component-agreement family.
The question is whether the v30 win is robust to local agreement-threshold changes, or whether it was only one lucky threshold setting.

## Selectors

- `VT-0.5`: mutation-free baseline
- `VT-0.5 CAG-z0.0`: loose agreement thresholds
- `VT-0.5 CAG-z0.25`: v30 default CAG threshold
- `VT-0.5 CAG-z0.5`: tight agreement thresholds

## Promotion Rule

The mutation thresholds are ranked on `Core-L` using the v30 screening composite:

- `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`
- only the strongest threshold is promoted against the mutation-free baseline

## Completed Runs

| Phase | Sel | Seed | Best | Last | Last5 | Dense | Comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-L Threshold Screen | VT-0.5 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Threshold Screen | VT-0.5 | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.0 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0469 | 0.0440 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.0 | 2234 | 0.1667 | 0.0833 | 0.0333 | 0.0312 | 0.0499 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.25 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0469 | 0.0440 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.25 | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.5 | 1234 | 0.1250 | 0.0417 | 0.0333 | 0.0469 | 0.0423 |
| Core-L Threshold Screen | VT-0.5 CAG-z0.5 | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 | 4234 | 0.1250 | 0.0417 | 0.0667 | 0.0547 | 0.0525 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0 | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| T1-L Threshold Screen | VT-0.5 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Threshold Screen | VT-0.5 | 2234 | 0.1250 | 0.0000 | 0.0167 | 0.0365 | 0.0197 |
| T1-L Threshold Screen | VT-0.5 CAG-z0.0 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Threshold Screen | VT-0.5 CAG-z0.0 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0469 | 0.0586 |
| T1-XL Confirmation | VT-0.5 | 4234 | 0.0833 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0573 | 0.0633 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0 | 4234 | 0.0417 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T2a-XL Stress | VT-0.5 | 5234 | 0.0417 | 0.0000 | 0.0083 | 0.0573 | 0.0274 |
| T2a-XL Stress | VT-0.5 | 6234 | 0.0833 | 0.0417 | 0.0333 | 0.0312 | 0.0353 |
| T2a-XL Stress | VT-0.5 CAG-z0.0 | 5234 | 0.0417 | 0.0000 | 0.0000 | 0.0521 | 0.0234 |
| T2a-XL Stress | VT-0.5 CAG-z0.0 | 6234 | 0.0833 | 0.0417 | 0.0250 | 0.0417 | 0.0383 |

## Core-L Threshold Screen

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0 | 0.1458 ± 0.0295 | 0.0625 ± 0.0295 | 0.0391 ± 0.0110 | 0.0470 ± 0.0042 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.25 | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0388 ± 0.0073 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.5 | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0380 ± 0.0062 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |

Promoted mutation threshold: `VT-0.5 CAG-z0.0`

## T1-L Threshold Screen

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0239 ± 0.0059 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |

## Core-XL Confirmation

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0625 ± 0.0295 | 0.0625 ± 0.0110 | 0.0600 ± 0.0106 | 0.0703 ± 0.0110 | 0.0547 ± 0.0110 |
| VT-0.5 CAG-z0.0 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |

## T1-XL Confirmation

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0521 ± 0.0074 | 0.0520 ± 0.0094 | 0.0547 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |
| VT-0.5 CAG-z0.0 | 0.0625 ± 0.0295 | 0.0625 ± 0.0295 | 0.0573 ± 0.0000 | 0.0543 ± 0.0127 | 0.0703 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0625 ± 0.0295 | 0.0208 ± 0.0295 | 0.0443 ± 0.0184 | 0.0314 ± 0.0056 | 0.0547 ± 0.0110 | 0.0312 ± 0.0221 | 0.0469 ± 0.0442 |
| VT-0.5 CAG-z0.0 | 0.0625 ± 0.0295 | 0.0208 ± 0.0295 | 0.0469 ± 0.0074 | 0.0309 ± 0.0105 | 0.0469 ± 0.0221 | 0.0391 ± 0.0110 | 0.0547 ± 0.0331 |

## Current Recommendation

Current best supported default: `VT-0.5 CAG-z0.0`.

This recommendation updates automatically from whichever completed phases exist. If only `Core-L` is complete, treat it as a screen rather than a final decision.

## Outputs

- Summary JSON: [summary_metrics_v31.json](/home/catid/gnn/reports/summary_metrics_v31.json)
- Report: [final_report_v31_cag_threshold_neighborhood.md](/home/catid/gnn/reports/final_report_v31_cag_threshold_neighborhood.md)
- Plots: [reports](/home/catid/gnn/reports)
