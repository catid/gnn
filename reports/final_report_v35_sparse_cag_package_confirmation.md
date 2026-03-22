# APSGNN v35: Sparse CAG Package Confirmation

## What Changed

v35 is a fresh-seed confirmation round for the recovered sparse mutation package from v34.
It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75-f0.50` on fresh Core-XL, T1-XL, T2a-XL, plus a final T1 rerun block with both best and last checkpoint evals.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense |
| --- | --- | --- | --- | --- | --- | --- |
| Core-XL Fresh Confirmation | VT-0.5 | 14234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 |
| Core-XL Fresh Confirmation | VT-0.5 | 15234 | 0.1250 | 0.0000 | 0.0547 | 0.0547 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 14234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 |
| Core-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 15234 | 0.1250 | 0.0000 | 0.0547 | 0.0547 |
| T1-XL Fresh Confirmation | VT-0.5 | 16234 | 0.1250 | 0.0417 | 0.0312 | 0.0312 |
| T1-XL Fresh Confirmation | VT-0.5 | 17234 | 0.1667 | 0.1667 | 0.0208 | 0.0208 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 16234 | 0.1250 | 0.0417 | 0.0312 | 0.0312 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.50 | 17234 | 0.1667 | 0.1667 | 0.0156 | 0.0156 |
| T1-XL Final Rerun | VT-0.5 | 20234 | 0.0417 | 0.0000 | 0.0382 | 0.0382 |
| T1-XL Final Rerun | VT-0.5 | 21234 | 0.0417 | 0.0000 | 0.0694 | 0.0694 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.50 | 20234 | 0.0417 | 0.0000 | 0.0417 | 0.0417 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.50 | 21234 | 0.0417 | 0.0000 | 0.0660 | 0.0660 |
| T2a-XL Stress | VT-0.5 | 18234 | 0.0833 | 0.0417 | 0.0573 | 0.0573 |
| T2a-XL Stress | VT-0.5 | 19234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.50 | 18234 | 0.0833 | 0.0417 | 0.0573 | 0.0573 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.50 | 19234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 |

## Core-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0430 ± 0.0166 | 0.0430 ± 0.0166 | 0.0312 ± 0.0000 | 0.0547 ± 0.0331 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0430 ± 0.0166 | 0.0430 ± 0.0166 | 0.0312 ± 0.0000 | 0.0547 ± 0.0331 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.1042 ± 0.0884 | 0.0260 ± 0.0074 | 0.0260 ± 0.0074 | 0.0234 ± 0.0110 | 0.0156 ± 0.0000 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.1458 ± 0.0295 | 0.1042 ± 0.0884 | 0.0234 ± 0.0110 | 0.0234 ± 0.0110 | 0.0156 ± 0.0221 | 0.0156 ± 0.0000 | 0.0391 ± 0.0110 |

## T2a-XL Stress

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0417 ± 0.0000 | 0.0443 ± 0.0184 | 0.0443 ± 0.0184 | 0.0234 ± 0.0110 | 0.0625 ± 0.0221 | 0.0469 ± 0.0442 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.0833 ± 0.0000 | 0.0417 ± 0.0000 | 0.0443 ± 0.0184 | 0.0443 ± 0.0184 | 0.0234 ± 0.0110 | 0.0625 ± 0.0221 | 0.0469 ± 0.0442 |

## T1-XL Final Rerun

| Selector | Best | Last | Dense | LastDense | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0417 ± 0.0000 | 0.0000 ± 0.0000 | 0.0538 ± 0.0221 | 0.0538 ± 0.0221 | 0.0573 ± 0.0516 | 0.0521 ± 0.0295 | 0.0521 ± 0.0147 |
| VT-0.5 CAG-z0.0-m0.75-f0.50 | 0.0417 ± 0.0000 | 0.0000 ± 0.0000 | 0.0538 ± 0.0172 | 0.0538 ± 0.0172 | 0.0625 ± 0.0589 | 0.0417 ± 0.0147 | 0.0573 ± 0.0221 |

## Current Recommendation

Current best supported default: `VT-0.5`.

## Outputs

- Summary JSON: [summary_metrics_v35.json](/home/catid/gnn/reports/summary_metrics_v35.json)
- Report: [final_report_v35_sparse_cag_package_confirmation.md](/home/catid/gnn/reports/final_report_v35_sparse_cag_package_confirmation.md)
- Plots: [reports](/home/catid/gnn/reports)
