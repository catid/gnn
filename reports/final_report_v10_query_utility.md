# APSGNN v10: Query-Aware Utility Refinement

## Purpose

V10 follows directly from v9. The goal is to keep the hard selective-growth benchmark fixed and test a narrower utility score that uses tail query-specific traffic/gradient signal instead of the weak success-conditioned traffic term. The core comparison is against the pushed v9 utility baselines `U` and `US`.

## Core Summary

| Arm | Seeds | Best Val | Last Val | Best K6 | Best K10 |
| --- | --- | --- | --- | --- | --- |
| v9_U | 6 | 0.5694 ± 0.1026 | 0.5556 ± 0.1100 | 0.4714 ± 0.0722 | 0.4141 ± 0.1386 |
| v9_US | 4 | 0.5677 ± 0.0548 | 0.5312 ± 0.0859 | 0.4492 ± 0.0780 | 0.4648 ± 0.1400 |
| querygrad | 4 | 0.6927 ± 0.0667 | 0.5677 ± 0.0462 | 0.5195 ± 0.0630 | 0.4863 ± 0.0400 |
| querymix | 2 | 0.6354 ± 0.0147 | 0.5729 ± 0.0737 | 0.5000 ± 0.0773 | 0.4961 ± 0.0055 |
| querygrad_mutate | 4 | 0.6875 ± 0.0851 | 0.5990 ± 0.0598 | 0.5195 ± 0.0352 | 0.5176 ± 0.0258 |

## Transfer Summary

| Arm | Seeds | Best K4 | Best K8 | Best K12 | Best K14 |
| --- | --- | --- | --- | --- | --- |
| transfer_querygrad | 2 | 0.5352 ± 0.1823 | 0.4453 ± 0.1436 | 0.4258 ± 0.0497 | 0.4141 ± 0.1326 |
| transfer_querygrad_mutate | 2 | 0.4258 ± 0.0387 | 0.3945 ± 0.0829 | 0.3867 ± 0.0055 | 0.4023 ± 0.0276 |

## Key Findings

- `querygrad` materially improved over the pushed v9 `U` and `US` baselines on the core regime.
- `querymix` was retained only as a redundancy check. On the two-seed score-selection round it produced the same split parents and the same `K2/K6/K10` checkpoint accuracies as `querygrad`.
- `querygrad_mutate` did not improve mean best validation over `querygrad`, but it did improve mean last-checkpoint stability and mean `K10` on the core regime.
- The H1 transfer round favored `querygrad` over `querygrad_mutate`, so mutation is not yet a reliable default upgrade.

## Notes

- `querygrad`: `z(visits) + z(grad) + z(query_grad)`
- `querymix`: `z(visits) + z(grad) + 0.5*z(query_visit) + z(query_grad)`
- `querygrad_mutate`: same score as `querygrad` with the existing modest local mutation policy
- The v9 baselines are read from [summary_metrics_v9.json](/home/catid/gnn/reports/summary_metrics_v9.json)

Artifacts:
- [summary_metrics_v10.json](/home/catid/gnn/reports/summary_metrics_v10.json)
- [v10_query_utility_comparison.png](/home/catid/gnn/reports/v10_query_utility_comparison.png)
