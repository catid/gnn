# APSGNN v36: Sparse F0.25 Package Confirmation

## What Changed

v36 is the fresh-seed confirmation round for the more aggressive sparse mutation package that tied the v34 screen but did not get the v35 confirmation slot.
It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75-f0.25` on fresh Core-XL, T1-XL, T2a-XL, plus a final T1 rerun block with both best and last checkpoint evals.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense |
| --- | --- | --- | --- | --- | --- | --- |
| Core-XL Fresh Confirmation | VT-0.5 | 22234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 |
| Core-XL Fresh Confirmation | VT-0.5 | 23234 | 0.1250 | 0.0417 | 0.0312 | 0.0312 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 22234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 23234 | 0.0833 | 0.0833 | 0.0234 | 0.0234 |
| T1-XL Fresh Confirmation | VT-0.5 | 24234 | 0.0417 | 0.0000 | 0.0625 | 0.0625 |
| T1-XL Fresh Confirmation | VT-0.5 | 25234 | 0.2083 | 0.0417 | 0.0885 | 0.0885 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 24234 | 0.0417 | 0.0000 | 0.0573 | 0.0573 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 25234 | 0.2083 | 0.0417 | 0.0885 | 0.0885 |
| T1-XL Final Rerun | VT-0.5 | 28234 | 0.0833 | 0.0000 | 0.0347 | 0.0347 |
| T1-XL Final Rerun | VT-0.5 | 29234 | 0.0833 | 0.0417 | 0.0451 | 0.0451 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.25 | 28234 | 0.1250 | 0.0000 | 0.0382 | 0.0382 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.25 | 29234 | 0.0833 | 0.0833 | 0.0521 | 0.0521 |
| T2a-XL Stress | VT-0.5 | 26234 | 0.1250 | 0.0833 | 0.0469 | 0.0469 |
| T2a-XL Stress | VT-0.5 | 27234 | 0.2500 | 0.2500 | 0.0365 | 0.0365 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 26234 | 0.1250 | 0.0833 | 0.0521 | 0.0521 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 27234 | 0.2083 | 0.2083 | 0.0417 | 0.0417 |

## Core-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0430 ± 0.0166 | 0.0430 ± 0.0166 | 0.0312 ± 0.0000 | 0.0547 ± 0.0331 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.0833 ± 0.0000 | 0.0417 ± 0.0589 | 0.0391 ± 0.0221 | 0.0391 ± 0.0221 | 0.0312 ± 0.0000 | 0.0469 ± 0.0442 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.1179 | 0.0208 ± 0.0295 | 0.0755 ± 0.0184 | 0.0755 ± 0.0184 | 0.0625 ± 0.0221 | 0.0547 ± 0.0331 | 0.1094 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1250 ± 0.1179 | 0.0208 ± 0.0295 | 0.0729 ± 0.0221 | 0.0729 ± 0.0221 | 0.0547 ± 0.0110 | 0.0469 ± 0.0442 | 0.1172 ± 0.0110 |

## T2a-XL Stress

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1875 ± 0.0884 | 0.1667 ± 0.1179 | 0.0417 ± 0.0074 | 0.0417 ± 0.0074 | 0.0234 ± 0.0110 | 0.0312 ± 0.0221 | 0.0703 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1667 ± 0.0589 | 0.1458 ± 0.0884 | 0.0469 ± 0.0074 | 0.0469 ± 0.0074 | 0.0312 ± 0.0221 | 0.0312 ± 0.0221 | 0.0781 ± 0.0221 |

## T1-XL Final Rerun

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0208 ± 0.0295 | 0.0399 ± 0.0074 | 0.0399 ± 0.0074 | 0.0417 ± 0.0295 | 0.0521 ± 0.0295 | 0.0260 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.0417 ± 0.0589 | 0.0451 ± 0.0098 | 0.0451 ± 0.0098 | 0.0417 ± 0.0295 | 0.0625 ± 0.0147 | 0.0312 ± 0.0147 |

## Current Recommendation

Current best supported default: `VT-0.5 CAG-z0.0-m0.75-f0.25`.

## Outputs

- Summary JSON: [summary_metrics_v36.json](/home/catid/gnn/reports/summary_metrics_v36.json)
- Report: [final_report_v36_sparse_f025_package_confirmation.md](/home/catid/gnn/reports/final_report_v36_sparse_f025_package_confirmation.md)
- Plots: [reports](/home/catid/gnn/reports)
