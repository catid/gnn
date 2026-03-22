# APSGNN v32: CAG-z0.0 Mutation Margin Neighborhood

## What Changed

v32 follows v31 by locking the winning loose agreement threshold and varying only the mutation score margin.
The question is whether `CAG-z0.0` is robust to local margin changes, or whether the v31 gain depends on the old `0.75` margin.

## Selectors

- `VT-0.5`: mutation-free baseline
- `VT-0.5 CAG-z0.0-m0.50`: looser mutation acceptance margin
- `VT-0.5 CAG-z0.0-m0.75`: v31 carry-over default margin
- `VT-0.5 CAG-z0.0-m1.00`: tighter mutation acceptance margin

## Promotion Rule

- rank mutation margins on `Core-L` using `0.45 * dense_mean + 0.35 * last_val + 0.20 * last5_val_mean`
- promote the strongest margin against the mutation-free baseline

## Completed Runs

| Phase | Sel | Seed | Best | Last | Last5 | Dense | Comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-L Margin Screen | VT-0.5 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Margin Screen | VT-0.5 | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m0.50 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m0.50 | 2234 | 0.1667 | 0.0833 | 0.0333 | 0.0312 | 0.0499 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m0.75 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0469 | 0.0440 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m0.75 | 2234 | 0.1667 | 0.0833 | 0.0333 | 0.0312 | 0.0499 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m1.00 | 1234 | 0.1250 | 0.0417 | 0.0333 | 0.0469 | 0.0423 |
| Core-L Margin Screen | VT-0.5 CAG-z0.0-m1.00 | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 | 4234 | 0.1250 | 0.0417 | 0.0667 | 0.0547 | 0.0525 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0-m0.50 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0-m0.50 | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0-m0.75 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Confirmation | VT-0.5 CAG-z0.0-m0.75 | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| T1-L Margin Screen | VT-0.5 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Margin Screen | VT-0.5 | 2234 | 0.1250 | 0.0000 | 0.0167 | 0.0365 | 0.0197 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m0.50 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m0.50 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m0.75 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m0.75 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m1.00 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Margin Screen | VT-0.5 CAG-z0.0-m1.00 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-XL Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0469 | 0.0586 |
| T1-XL Confirmation | VT-0.5 | 4234 | 0.0833 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0-m0.50 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0573 | 0.0633 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0-m0.50 | 4234 | 0.0417 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0-m0.75 | 3234 | 0.0833 | 0.0833 | 0.0417 | 0.0573 | 0.0633 |
| T1-XL Confirmation | VT-0.5 CAG-z0.0-m0.75 | 4234 | 0.0417 | 0.0417 | 0.0250 | 0.0573 | 0.0454 |
| T2a-XL Stress | VT-0.5 | 5234 | 0.0417 | 0.0000 | 0.0083 | 0.0573 | 0.0274 |
| T2a-XL Stress | VT-0.5 | 6234 | 0.0833 | 0.0417 | 0.0333 | 0.0312 | 0.0353 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75 | 5234 | 0.0417 | 0.0000 | 0.0000 | 0.0521 | 0.0234 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75 | 6234 | 0.0833 | 0.0417 | 0.0250 | 0.0417 | 0.0383 |

## Core-L Margin Screen

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.50 | 0.1458 ± 0.0295 | 0.0625 ± 0.0295 | 0.0352 ± 0.0055 | 0.0452 ± 0.0066 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1458 ± 0.0295 | 0.0625 ± 0.0295 | 0.0391 ± 0.0110 | 0.0470 ± 0.0042 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m1.00 | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0380 ± 0.0062 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |

Promoted mutation margin: `VT-0.5 CAG-z0.0-m0.75`

## T1-L Margin Screen

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0239 ± 0.0059 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.50 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0-m1.00 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |

## Core-XL Confirmation

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0625 ± 0.0295 | 0.0625 ± 0.0110 | 0.0600 ± 0.0106 | 0.0703 ± 0.0110 | 0.0547 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.50 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |

## T1-XL Confirmation

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0521 ± 0.0074 | 0.0520 ± 0.0094 | 0.0547 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.50 | 0.0625 ± 0.0295 | 0.0625 ± 0.0295 | 0.0573 ± 0.0000 | 0.0543 ± 0.0127 | 0.0703 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75 | 0.0625 ± 0.0295 | 0.0625 ± 0.0295 | 0.0573 ± 0.0000 | 0.0543 ± 0.0127 | 0.0703 ± 0.0110 | 0.0391 ± 0.0110 | 0.0625 ± 0.0221 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0625 ± 0.0295 | 0.0208 ± 0.0295 | 0.0443 ± 0.0184 | 0.0314 ± 0.0056 | 0.0547 ± 0.0110 | 0.0312 ± 0.0221 | 0.0469 ± 0.0442 |
| VT-0.5 CAG-z0.0-m0.75 | 0.0625 ± 0.0295 | 0.0208 ± 0.0295 | 0.0469 ± 0.0074 | 0.0309 ± 0.0105 | 0.0469 ± 0.0221 | 0.0391 ± 0.0110 | 0.0547 ± 0.0331 |

## Current Recommendation

Current best supported default: `VT-0.5 CAG-z0.0-m0.75`.

## Outputs

- Summary JSON: [summary_metrics_v32.json](/home/catid/gnn/reports/summary_metrics_v32.json)
- Report: [final_report_v32_cag_margin_neighborhood.md](/home/catid/gnn/reports/final_report_v32_cag_margin_neighborhood.md)
- Plots: [reports](/home/catid/gnn/reports)
