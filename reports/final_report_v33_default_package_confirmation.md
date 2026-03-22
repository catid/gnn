# APSGNN v33: Default Package Confirmation

## What Changed

v33 is a fresh-seed confirmation round for the full recommended package from v32.
It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75` on fresh Core-XL, T1-XL, T2a-XL, plus a final T1 rerun block with both best and last checkpoint evals.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense |
| --- | --- | --- | --- | --- | --- | --- |
| Core-XL Fresh Confirmation | VT-0.5 | 7234 | 0.1250 | 0.0417 | 0.0391 | 0.0391 |
| Core-XL Fresh Confirmation | VT-0.5 | 8234 | 0.1667 | 0.0000 | 0.0547 | 0.0547 |
| Core-XL Fresh Confirmation | VT-0.5 | 9234 | 0.0833 | 0.0000 | 0.0312 | 0.0312 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 7234 | 0.1250 | 0.0417 | 0.0391 | 0.0391 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 8234 | 0.1667 | 0.0000 | 0.0547 | 0.0547 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 9234 | 0.0833 | 0.0000 | 0.0312 | 0.0312 |
| T1-XL Fresh Confirmation | VT-0.5 | 7234 | 0.1667 | 0.0417 | 0.0417 | 0.0417 |
| T1-XL Fresh Confirmation | VT-0.5 | 8234 | 0.1667 | 0.0833 | 0.0417 | 0.0417 |
| T1-XL Fresh Confirmation | VT-0.5 | 9234 | 0.1250 | 0.0417 | 0.0573 | 0.0573 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 7234 | 0.1250 | 0.0417 | 0.0417 | 0.0417 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 8234 | 0.1667 | 0.0417 | 0.0365 | 0.0365 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75 | 9234 | 0.1250 | 0.0417 | 0.0521 | 0.0521 |
| T1-XL Final Rerun | VT-0.5 | 12234 | 0.1250 | 0.0000 | 0.0521 | 0.0521 |
| T1-XL Final Rerun | VT-0.5 | 13234 | 0.0833 | 0.0417 | 0.0208 | 0.0208 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75 | 12234 | 0.1250 | 0.0000 | 0.0521 | 0.0521 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75 | 13234 | 0.0833 | 0.0000 | 0.0417 | 0.0278 |
| T2a-XL Stress | VT-0.5 | 10234 | 0.0833 | 0.0000 | 0.0365 | 0.0365 |
| T2a-XL Stress | VT-0.5 | 11234 | 0.1250 | 0.0000 | 0.0521 | 0.0521 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75 | 10234 | 0.0833 | 0.0000 | 0.0417 | 0.0417 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75 | 11234 | 0.1250 | 0.0000 | 0.0417 | 0.0417 |

## Core-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0417 | 0.0139 ± 0.0241 | 0.0417 ± 0.0119 | 0.0417 ± 0.0119 | 0.0365 ± 0.0090 | 0.0469 ± 0.0156 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1250 ± 0.0417 | 0.0139 ± 0.0241 | 0.0417 ± 0.0119 | 0.0417 ± 0.0119 | 0.0365 ± 0.0090 | 0.0469 ± 0.0156 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1528 ± 0.0241 | 0.0556 ± 0.0241 | 0.0469 ± 0.0090 | 0.0469 ± 0.0090 | 0.0521 ± 0.0239 | 0.0312 ± 0.0413 | 0.0573 ± 0.0090 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1389 ± 0.0241 | 0.0417 ± 0.0000 | 0.0434 ± 0.0080 | 0.0434 ± 0.0080 | 0.0469 ± 0.0271 | 0.0312 ± 0.0312 | 0.0521 ± 0.0180 |

## T2a-XL Stress

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0443 ± 0.0110 | 0.0443 ± 0.0110 | 0.0234 ± 0.0331 | 0.0859 ± 0.0331 | 0.0234 ± 0.0331 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0417 ± 0.0000 | 0.0417 ± 0.0000 | 0.0234 ± 0.0331 | 0.0781 ± 0.0442 | 0.0234 ± 0.0110 |

## T1-XL Final Rerun

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0365 ± 0.0221 | 0.0365 ± 0.0221 | 0.0469 ± 0.0074 | 0.0417 ± 0.0295 | 0.0208 ± 0.0295 |
| VT-0.5 CAG-z0.0-m0.75 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0469 ± 0.0074 | 0.0399 ± 0.0172 | 0.0365 ± 0.0074 | 0.0729 ± 0.0147 | 0.0312 ± 0.0295 |

## Current Recommendation

Current best supported default: `VT-0.5`.

## Outputs

- Summary JSON: [summary_metrics_v33.json](/home/catid/gnn/reports/summary_metrics_v33.json)
- Report: [final_report_v33_default_package_confirmation.md](/home/catid/gnn/reports/final_report_v33_default_package_confirmation.md)
- Plots: [reports](/home/catid/gnn/reports)
