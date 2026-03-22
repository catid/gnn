# APSGNN v34: Mutation Sparsity and Stagnation Revisit

## What Changed

v34 follows the v33 reversal of the mutation package.
The question here is whether the component-agreement mutation can be recovered by making it sparser and/or requiring stage stagnation, rather than abandoning mutation entirely.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Last5 | Dense | Comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-L Aggressiveness Screen | VT-0.5 | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Aggressiveness Screen | VT-0.5 | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 1234 | 0.1250 | 0.0417 | 0.0333 | 0.0469 | 0.0423 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50 | 1234 | 0.1250 | 0.0417 | 0.0333 | 0.0469 | 0.0423 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50 | 2234 | 0.1667 | 0.0417 | 0.0250 | 0.0312 | 0.0336 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50-stag | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50-stag | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-stag | 1234 | 0.1250 | 0.0417 | 0.0417 | 0.0391 | 0.0405 |
| Core-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-stag | 2234 | 0.1667 | 0.0000 | 0.0167 | 0.0312 | 0.0174 |
| Core-XL Fresh Confirmation | VT-0.5 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Fresh Confirmation | VT-0.5 | 4234 | 0.1250 | 0.0417 | 0.0667 | 0.0547 | 0.0525 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 3234 | 0.0833 | 0.0833 | 0.0333 | 0.0703 | 0.0675 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 4234 | 0.1250 | 0.0833 | 0.0750 | 0.0547 | 0.0688 |
| T1-L Aggressiveness Screen | VT-0.5 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Aggressiveness Screen | VT-0.5 | 2234 | 0.1250 | 0.0000 | 0.0167 | 0.0365 | 0.0197 |
| T1-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50 | 1234 | 0.1250 | 0.0000 | 0.0000 | 0.0625 | 0.0281 |
| T1-L Aggressiveness Screen | VT-0.5 CAG-z0.0-m0.75-f0.50 | 2234 | 0.1250 | 0.0000 | 0.0250 | 0.0365 | 0.0214 |
| T1-XL Fresh Confirmation | VT-0.5 | 5234 | 0.1250 | 0.0417 | 0.0750 | 0.0312 | 0.0436 |
| T1-XL Fresh Confirmation | VT-0.5 | 6234 | 0.0833 | 0.0417 | 0.0417 | 0.0573 | 0.0487 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 5234 | 0.1250 | 0.0833 | 0.0833 | 0.0312 | 0.0599 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 6234 | 0.0833 | 0.0417 | 0.0417 | 0.0573 | 0.0487 |
| T2a-XL Stress | VT-0.5 | 7234 | 0.1250 | 0.0417 | 0.0583 | 0.0677 | 0.0567 |
| T2a-XL Stress | VT-0.5 | 8234 | 0.1667 | 0.0833 | 0.0917 | 0.0208 | 0.0569 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.50 | 7234 | 0.1250 | 0.0833 | 0.0667 | 0.0573 | 0.0683 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.50 | 8234 | 0.1667 | 0.0417 | 0.0833 | 0.0208 | 0.0406 |

## Core-L Aggressiveness Screen

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0380 ± 0.0062 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1458 ± 0.0295 | 0.0417 ± 0.0000 | 0.0391 ± 0.0110 | 0.0380 ± 0.0062 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-stag | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75-f0.50-stag | 0.1458 ± 0.0295 | 0.0208 ± 0.0295 | 0.0352 ± 0.0055 | 0.0289 ± 0.0163 | 0.0391 ± 0.0331 | 0.0312 ± 0.0221 |

## T1-L Aggressiveness Screen

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0239 ± 0.0059 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1250 ± 0.0000 | 0.0000 ± 0.0000 | 0.0495 ± 0.0184 | 0.0248 ± 0.0048 | 0.0547 ± 0.0331 | 0.0469 ± 0.0221 | 0.0469 ± 0.0000 |

## Core-XL Fresh Confirmation

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0625 ± 0.0295 | 0.0625 ± 0.0110 | 0.0600 ± 0.0106 | 0.0703 ± 0.0110 | 0.0547 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0625 ± 0.0110 | 0.0681 ± 0.0009 | 0.0625 ± 0.0000 | 0.0625 ± 0.0221 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0417 ± 0.0000 | 0.0443 ± 0.0184 | 0.0462 ± 0.0036 | 0.0625 ± 0.0221 | 0.0312 ± 0.0221 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1042 ± 0.0295 | 0.0625 ± 0.0295 | 0.0443 ± 0.0184 | 0.0543 ± 0.0079 | 0.0547 ± 0.0110 | 0.0391 ± 0.0331 | 0.0391 ± 0.0110 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0625 ± 0.0295 | 0.0443 ± 0.0331 | 0.0568 ± 0.0001 | 0.0391 ± 0.0110 | 0.0625 ± 0.0442 | 0.0312 ± 0.0442 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1458 ± 0.0295 | 0.0625 ± 0.0295 | 0.0391 ± 0.0258 | 0.0545 ± 0.0196 | 0.0469 ± 0.0221 | 0.0469 ± 0.0663 | 0.0234 ± 0.0331 |

## Current Recommendation

Current best supported default: `VT-0.5 CAG-z0.0-m0.75-f0.50`.

## Outputs

- Summary JSON: [summary_metrics_v34.json](/home/catid/gnn/reports/summary_metrics_v34.json)
- Report: [final_report_v34_mutation_sparsity_stagnation.md](/home/catid/gnn/reports/final_report_v34_mutation_sparsity_stagnation.md)
- Plots: [reports](/home/catid/gnn/reports)
