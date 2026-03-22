# APSGNN v37: Sparse Fraction Neighborhood

## What Changed

v37 is the fine-grained neighborhood check around the recovered sparse CAG package.
It compares mutation-free `VT-0.5` against `VT-0.5 CAG-z0.0-m0.75` with mutation-selected fractions `0.125`, `0.25`, and `0.375`.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense | Comp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Core-L Sparse Fraction Screen | VT-0.5 | 30234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 | 0.0263 |
| Core-L Sparse Fraction Screen | VT-0.5 | 31234 | 0.1250 | 0.0833 | 0.0547 | 0.0547 | 0.0688 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.125 | 30234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 | 0.0263 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.125 | 31234 | 0.1250 | 0.0417 | 0.0703 | 0.0703 | 0.0596 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 30234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 | 0.0263 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 31234 | 0.1250 | 0.0417 | 0.0703 | 0.0703 | 0.0596 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.375 | 30234 | 0.0833 | 0.0000 | 0.0547 | 0.0547 | 0.0263 |
| Core-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.375 | 31234 | 0.1250 | 0.0417 | 0.0703 | 0.0703 | 0.0596 |
| T1-L Sparse Fraction Screen | VT-0.5 | 30234 | 0.0833 | 0.0000 | 0.0260 | 0.0260 | 0.0151 |
| T1-L Sparse Fraction Screen | VT-0.5 | 31234 | 0.0833 | 0.0000 | 0.0365 | 0.0365 | 0.0214 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.125 | 30234 | 0.0833 | 0.0000 | 0.0260 | 0.0260 | 0.0151 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.125 | 31234 | 0.1250 | 0.0000 | 0.0417 | 0.0417 | 0.0271 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 30234 | 0.0833 | 0.0000 | 0.0260 | 0.0260 | 0.0151 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.25 | 31234 | 0.1250 | 0.0000 | 0.0417 | 0.0417 | 0.0271 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.375 | 30234 | 0.0833 | 0.0000 | 0.0260 | 0.0260 | 0.0151 |
| T1-L Sparse Fraction Screen | VT-0.5 CAG-z0.0-m0.75-f0.375 | 31234 | 0.1250 | 0.0000 | 0.0417 | 0.0417 | 0.0271 |
| T1-XL Fresh Confirmation | VT-0.5 | 34234 | 0.0833 | 0.0833 | 0.0625 | 0.0625 | 0.0690 |
| T1-XL Fresh Confirmation | VT-0.5 | 35234 | 0.0833 | 0.0417 | 0.0260 | 0.0260 | 0.0296 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.125 | 34234 | 0.0833 | 0.0833 | 0.0573 | 0.0573 | 0.0666 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.125 | 35234 | 0.0833 | 0.0417 | 0.0260 | 0.0260 | 0.0296 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 34234 | 0.0833 | 0.0833 | 0.0573 | 0.0573 | 0.0666 |
| T1-XL Fresh Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 35234 | 0.0833 | 0.0417 | 0.0260 | 0.0260 | 0.0296 |
| T1-XL Final Rerun | VT-0.5 | 36234 | 0.1250 | 0.1250 | 0.0417 | 0.0417 | 0.0725 |
| T1-XL Final Rerun | VT-0.5 | 37234 | 0.0833 | 0.0833 | 0.0556 | 0.0556 | 0.0642 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.25 | 36234 | 0.1250 | 0.1250 | 0.0417 | 0.0417 | 0.0742 |
| T1-XL Final Rerun | VT-0.5 CAG-z0.0-m0.75-f0.25 | 37234 | 0.0833 | 0.0833 | 0.0590 | 0.0590 | 0.0657 |
| T2a-XL Stress | VT-0.5 | 38234 | 0.1667 | 0.0417 | 0.0417 | 0.0417 | 0.0434 |
| T2a-XL Stress | VT-0.5 | 39234 | 0.1250 | 0.1250 | 0.0208 | 0.0208 | 0.0665 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 38234 | 0.1667 | 0.0417 | 0.0417 | 0.0417 | 0.0434 |
| T2a-XL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 39234 | 0.1250 | 0.1250 | 0.0208 | 0.0208 | 0.0665 |

## Core-L Sparse Fraction Screen

| Selector | Best | Last | Dense | Comp | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.0417 ± 0.0589 | 0.0547 ± 0.0000 | 0.0475 ± 0.0301 | 0.0469 ± 0.0000 | 0.0625 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.125 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0625 ± 0.0110 | 0.0429 ± 0.0235 | 0.0547 ± 0.0110 | 0.0703 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0625 ± 0.0110 | 0.0429 ± 0.0235 | 0.0547 ± 0.0110 | 0.0703 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.375 | 0.1042 ± 0.0295 | 0.0208 ± 0.0295 | 0.0625 ± 0.0110 | 0.0429 ± 0.0235 | 0.0547 ± 0.0110 | 0.0703 ± 0.0110 |

## T1-L Sparse Fraction Screen

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0000 ± 0.0000 | 0.0312 ± 0.0074 | 0.0182 ± 0.0045 | 0.0234 ± 0.0331 | 0.0391 ± 0.0110 | 0.0312 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.125 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0339 ± 0.0110 | 0.0211 ± 0.0085 | 0.0312 ± 0.0442 | 0.0391 ± 0.0110 | 0.0312 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0339 ± 0.0110 | 0.0211 ± 0.0085 | 0.0312 ± 0.0442 | 0.0391 ± 0.0110 | 0.0312 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.375 | 0.1042 ± 0.0295 | 0.0000 ± 0.0000 | 0.0339 ± 0.0110 | 0.0211 ± 0.0085 | 0.0312 ± 0.0442 | 0.0391 ± 0.0110 | 0.0312 ± 0.0000 |

## T1-XL Fresh Confirmation

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0443 ± 0.0258 | 0.0493 ± 0.0278 | 0.0391 ± 0.0110 | 0.0547 ± 0.0552 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.125 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0417 ± 0.0221 | 0.0481 ± 0.0261 | 0.0312 ± 0.0000 | 0.0547 ± 0.0552 | 0.0391 ± 0.0110 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.0833 ± 0.0000 | 0.0625 ± 0.0295 | 0.0417 ± 0.0221 | 0.0481 ± 0.0261 | 0.0312 ± 0.0000 | 0.0547 ± 0.0552 | 0.0391 ± 0.0110 |

## T2a-XL Stress

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1458 ± 0.0295 | 0.0833 ± 0.0589 | 0.0312 ± 0.0147 | 0.0549 ± 0.0163 | 0.0312 ± 0.0221 | 0.0312 ± 0.0221 | 0.0312 ± 0.0000 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1458 ± 0.0295 | 0.0833 ± 0.0589 | 0.0312 ± 0.0147 | 0.0549 ± 0.0163 | 0.0312 ± 0.0221 | 0.0312 ± 0.0221 | 0.0312 ± 0.0000 |

## T1-XL Final Rerun

| Selector | Best | Last | Dense | Comp | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0295 | 0.1042 ± 0.0295 | 0.0486 ± 0.0098 | 0.0683 ± 0.0059 | 0.0312 ± 0.0000 | 0.0677 ± 0.0221 | 0.0469 ± 0.0516 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0295 | 0.1042 ± 0.0295 | 0.0503 ± 0.0123 | 0.0699 ± 0.0060 | 0.0417 ± 0.0147 | 0.0677 ± 0.0221 | 0.0417 ± 0.0442 |

## Current Recommendation

The sparse-fraction neighborhood stays flat.
`f0.125` adds no measurable gain over `f0.25`, and `f0.375` never separates from the weaker screen-only pattern.
Across the completed phases, mutation-free `VT-0.5` remains the safest default because it leads on `Core-L`,
holds the slight edge on fresh `T1-XL`, and ties `f0.25` on fresh `T2a-XL`.
`VT-0.5 CAG-z0.0-m0.75-f0.25` remains a live rerun-specialist variant because it keeps the small `T1r-XL` dense edge,
but v37 does not show a strong enough cross-regime advantage to replace the mutation-free default.

## Outputs

- Summary JSON: [summary_metrics_v37.json](/home/catid/gnn/reports/summary_metrics_v37.json)
- Report: [final_report_v37_sparse_fraction_neighborhood.md](/home/catid/gnn/reports/final_report_v37_sparse_fraction_neighborhood.md)
- Plots: [reports](/home/catid/gnn/reports)
