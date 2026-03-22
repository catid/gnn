# APSGNN v39: Sparse Transfer Specialist Validation

## What Changed

v39 follows the mixed v38 result with a transfer-only validation.
It drops home confirmation and asks a narrower question: does `VT-0.5 CAG-z0.0-m0.75-f0.25` win enough fresh transfer-like evidence to deserve a transfer-specialist role, or should it be retired.

## Completed Runs

| Phase | Sel | Seed | Best | Last | Dense | LastDense | Score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1-XL Fresh Transfer Confirmation | VT-0.5 | 50234 | 0.0417 | 0.0000 | 0.0243 | 0.0243 | 0.0243 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 | 51234 | 0.1250 | 0.0417 | 0.0556 | 0.0556 | 0.0972 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 | 52234 | 0.1250 | 0.0417 | 0.0243 | 0.0243 | 0.0660 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 | 53234 | 0.1250 | 0.0833 | 0.0486 | 0.0486 | 0.1319 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 50234 | 0.0417 | 0.0000 | 0.0278 | 0.0278 | 0.0278 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 51234 | 0.1250 | 0.0417 | 0.0590 | 0.0590 | 0.1007 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 52234 | 0.1250 | 0.0417 | 0.0243 | 0.0243 | 0.0660 |
| T1-XL Fresh Transfer Confirmation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 53234 | 0.1250 | 0.0833 | 0.0521 | 0.0521 | 0.1354 |
| T1-XL Rerun Transfer Validation | VT-0.5 | 54234 | 0.1250 | 0.0476 | 0.0486 | 0.0523 | 0.0962 |
| T1-XL Rerun Transfer Validation | VT-0.5 | 55234 | 0.0833 | 0.0833 | 0.0417 | 0.0417 | 0.1250 |
| T1-XL Rerun Transfer Validation | VT-0.5 | 56234 | 0.0833 | 0.0417 | 0.0382 | 0.0382 | 0.0799 |
| T1-XL Rerun Transfer Validation | VT-0.5 | 57234 | 0.1250 | 0.0417 | 0.0556 | 0.0556 | 0.0972 |
| T1-XL Rerun Transfer Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 54234 | 0.1250 | 0.0000 | 0.0486 | 0.0593 | 0.0486 |
| T1-XL Rerun Transfer Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 55234 | 0.0833 | 0.0833 | 0.0451 | 0.0451 | 0.1285 |
| T1-XL Rerun Transfer Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 56234 | 0.0833 | 0.0417 | 0.0312 | 0.0312 | 0.0729 |
| T1-XL Rerun Transfer Validation | VT-0.5 CAG-z0.0-m0.75-f0.25 | 57234 | 0.1250 | 0.0417 | 0.0556 | 0.0556 | 0.0972 |
| T2b-XL Tight-TTL Stress | VT-0.5 | 58234 | 0.1667 | 0.0417 | 0.0278 | 0.0278 | 0.0694 |
| T2b-XL Tight-TTL Stress | VT-0.5 | 59234 | 0.0833 | 0.0000 | 0.0451 | 0.0451 | 0.0451 |
| T2b-XL Tight-TTL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 58234 | 0.1667 | 0.0833 | 0.0312 | 0.0312 | 0.1146 |
| T2b-XL Tight-TTL Stress | VT-0.5 CAG-z0.0-m0.75-f0.25 | 59234 | 0.0833 | 0.0000 | 0.0451 | 0.0451 | 0.0451 |

## T1-XL Fresh Transfer Confirmation

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0417 | 0.0417 ± 0.0340 | 0.0382 ± 0.0163 | 0.0799 ± 0.0458 | 0.0391 ± 0.0428 | 0.0365 ± 0.0180 | 0.0391 ± 0.0100 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0417 | 0.0417 ± 0.0340 | 0.0408 ± 0.0173 | 0.0825 ± 0.0462 | 0.0417 ± 0.0371 | 0.0391 ± 0.0260 | 0.0417 ± 0.0085 |

## T1-XL Rerun Transfer Validation

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1042 ± 0.0241 | 0.0536 ± 0.0200 | 0.0460 ± 0.0077 | 0.0996 ± 0.0187 | 0.0391 ± 0.0260 | 0.0365 ± 0.0199 | 0.0625 ± 0.0225 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1042 ± 0.0241 | 0.0417 ± 0.0340 | 0.0451 ± 0.0102 | 0.0868 ± 0.0341 | 0.0365 ± 0.0276 | 0.0365 ± 0.0180 | 0.0625 ± 0.0282 |

## T2b-XL Tight-TTL Stress

| Selector | Best | Last | Dense | Score | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| VT-0.5 | 0.1250 ± 0.0589 | 0.0208 ± 0.0295 | 0.0365 ± 0.0123 | 0.0573 ± 0.0172 | 0.0260 ± 0.0074 | 0.0469 ± 0.0074 | 0.0365 ± 0.0368 |
| VT-0.5 CAG-z0.0-m0.75-f0.25 | 0.1250 ± 0.0589 | 0.0417 ± 0.0589 | 0.0382 ± 0.0098 | 0.0799 ± 0.0491 | 0.0312 ± 0.0000 | 0.0469 ± 0.0074 | 0.0365 ± 0.0368 |

## Current Recommendation

`VT-0.5 CAG-z0.0-m0.75-f0.25` wins the transfer-focused v39 score and stays alive as a real transfer-specialist package.
Mutation-free `VT-0.5` remains the overall default until a broader whole-regime confirmation says otherwise.

## Outputs

- Summary JSON: [summary_metrics_v39.json](/home/catid/gnn/reports/summary_metrics_v39.json)
- Report: [final_report_v39_sparse_transfer_specialist_validation.md](/home/catid/gnn/reports/final_report_v39_sparse_transfer_specialist_validation.md)
- Plots: [reports](/home/catid/gnn/reports)
