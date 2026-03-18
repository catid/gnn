# APSGNN v17 Querygrad Component Ablation

## What Changed

This round keeps the selective-growth architecture fixed and tests only the utility selector components behind the current `querygrad` default.

New additive mechanism:

- `utility_visit_weight`: explicit weight on the base task-visit term, so `visit-only` and `querygrad-only` are both honest score variants.

Arms:

- `querygrad`: existing v10 reference, `visit + query_grad`
- `visitonly`: `visit` only
- `querygradonly`: `query_grad` only

Core regime stays on the long selective benchmark with `writers_per_episode=2` and eval at `2/6/10`. Transfer regime stays on H1 with training at `4` writers and eval at `4/8/12/14`.

## Core Summary

| Arm | Count | Best Val | Last Val | K2 | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| querygrad | 4 | 0.6927 ± 0.0667 | 0.5677 ± 0.0462 | 0.6133 ± 0.0614 | 0.5195 ± 0.0630 | 0.4863 ± 0.0400 |
| visitonly | 3 | 0.6944 ± 0.0842 | 0.6736 ± 0.0789 | 0.6797 ± 0.0341 | 0.5964 ± 0.0119 | 0.5182 ± 0.0471 |
| querygradonly | 3 | 0.7014 ± 0.0120 | 0.6319 ± 0.0732 | 0.5964 ± 0.0508 | 0.4844 ± 0.0135 | 0.4323 ± 0.0674 |

## H1 Transfer Summary

| Arm | Count | Best Val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- |
| querygrad_h1 | 3 | 0.5903 ± 0.1356 | 0.5182 ± 0.1322 | 0.4427 ± 0.1017 | 0.4271 ± 0.0352 | 0.4141 ± 0.0938 |

## Interpretation

`querygrad` still looks like the best default once the score is decomposed cleanly. `visitonly` tests whether simple traffic concentration is already enough; `querygradonly` tests whether the query-side signal alone is enough. If both fall below `querygrad`, the selector really is using a useful combination rather than one redundant term.

In practice this round is aimed at validating the current score, not replacing the model family. The right next step after this should depend on whether the better challenger is `visitonly` or `querygradonly`.

## Outputs

- summary JSON: [`summary_metrics_v17.json`](/home/catid/gnn/reports/summary_metrics_v17.json)
- core plot: [`v17_core_component_ablation.png`](/home/catid/gnn/reports/v17_core_component_ablation.png)
- H1 plot: [`v17_h1_component_ablation.png`](/home/catid/gnn/reports/v17_h1_component_ablation.png)
