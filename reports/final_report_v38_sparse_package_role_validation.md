# APSGNN v38: Sparse Package Role Validation

## What Changed

v38 is the direct follow-up to the mixed v37 result.
It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75-f0.25` only, using fresh paired `XL` runs on home, transfer, rerun, and stress regimes.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-XL Fresh Confirmation | VT-0.5 | 40234 | 0.0833 | 0.0000 | 0.0391 | 0.0391 | 0.0391 |
| Core-XL Fresh Confirmation | VT-0.5 | 41234 | 0.0833 | 0.0417 | 0.0469 | 0.0469 | 0.0885 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 40234 | 0.0833 | 0.0000 | 0.0391 | 0.0391 | 0.0391 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 41234 | 0.0833 | 0.0417 | 0.0469 | 0.0469 | 0.0885 |
| T1-XL Fresh Confirmation | VT-0.5 | 42234 | 0.1250 | 0.0833 | 0.0104 | 0.0104 | 0.0938 |
| T1-XL Fresh Confirmation | VT-0.5 | 43234 | 0.1250 | 0.0833 | 0.0521 | 0.0521 | 0.1354 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 42234 | 0.0833 | 0.0833 | 0.0417 | 0.0417 | 0.1250 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 43234 | 0.1250 | 0.0833 | 0.0417 | 0.0417 | 0.1250 |
| T1-XL Rerun Validation | VT-0.5 | 44234 | 0.0833 | 0.0833 | 0.0451 | 0.0451 | 0.1285 |
| T1-XL Rerun Validation | VT-0.5 | 45234 | 0.0417 | 0.0000 | 0.0451 | 0.0451 | 0.0451 |
| T1-XL Rerun Validation | VT-0.5 | 46234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 | 0.0729 |
| T1-XL Rerun Validation | VT-0.5 | 47234 | 0.2917 | 0.0833 | 0.0486 | 0.0486 | 0.1319 |
| T1-XL Rerun Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 44234 | 0.0833 | 0.0417 | 0.0417 | 0.0417 | 0.0833 |
| T1-XL Rerun Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 45234 | 0.0417 | 0.0000 | 0.0451 | 0.0451 | 0.0451 |
| T1-XL Rerun Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 46234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 | 0.0729 |
| T1-XL Rerun Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 47234 | 0.2917 | 0.0833 | 0.0451 | 0.0451 | 0.1285 |
| T2a-XL Stress | VT-0.5 | 48234 | 0.1250 | 0.0833 | 0.0243 | 0.0243 | 0.1076 |
| T2a-XL Stress | VT-0.5 | 49234 | 0.1250 | 0.1250 | 0.0521 | 0.0521 | 0.1771 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 48234 | 0.1250 | 0.0833 | 0.0243 | 0.0243 | 0.1076 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 49234 | 0.1250 | 0.1250 | 0.0521 | 0.0521 | 0.1771 |

## Core-XL Fresh Confirmation

| Selector | Best | Last | Dense | Score | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0208 ± 0.0295 | 0.0430 ± 0.0055 | 0.0638 ± 0.0350 | 0.0625 ± 0.0000 | 0.0234 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.0833 ± 0.0000 | 0.0208 ± 0.0295 | 0.0430 ± 0.0055 | 0.0638 ± 0.0350 | 0.0625 ± 0.0000 | 0.0234 ± 0.0110 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.0833 ± 0.0000 | 0.0312 ± 0.0295 | 0.1146 ± 0.0295 | 0.0391 ± 0.0552 | 0.0234 ± 0.0110 | 0.0312 ± 0.0221 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.0833 ± 0.0000 | 0.0417 ± 0.0000 | 0.1250 ± 0.0000 | 0.0469 ± 0.0221 | 0.0391 ± 0.0110 | 0.0391 ± 0.0110 |

## T1-XL Rerun Validation

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.1128 | 0.0521 ± 0.0399 | 0.0425 ± 0.0077 | 0.0946 ± 0.0427 | 0.0391 ± 0.0131 | 0.0339 ± 0.0131 | 0.0547 ± 0.0156 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1250 ± 0.1128 | 0.0417 ± 0.0340 | 0.0408 ± 0.0066 | 0.0825 ± 0.0346 | 0.0365 ± 0.0104 | 0.0391 ± 0.0100 | 0.0469 ± 0.0060 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0000 | 0.1042 ± 0.0295 | 0.0382 ± 0.0196 | 0.1424 ± 0.0491 | 0.0417 ± 0.0147 | 0.0312 ± 0.0147 | 0.0417 ± 0.0295 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1250 ± 0.0000 | 0.1042 ± 0.0295 | 0.0382 ± 0.0196 | 0.1424 ± 0.0491 | 0.0417 ± 0.0147 | 0.0312 ± 0.0147 | 0.0417 ± 0.0295 |

## Current Recommendation

Mutation-free `VT-0.5` remains the safer default on the completed fresh v38 validation.
`VT-0.5 CAG-z0.0-m0.75-f0.25` stays alive only if it keeps a targeted rerun-style edge that does not generalize cleanly.

## Outputs

- Summary JSON: [summary_metrics_v38.json](/home/catid/gnn/reports/summary_metrics_v38.json)
- Report: [final_report_v38_sparse_package_role_validation.md](/home/catid/gnn/reports/final_report_v38_sparse_package_role_validation.md)
- Plots: [reports](/home/catid/gnn/reports)
