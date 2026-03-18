# APSGNN v18 Selector Transfer Campaign

## What Changed

V18 keeps the selective-growth APSGNN family fixed and tests selector choice only.

This round focuses on selector choice rather than mutation because v10-v17 already established
that mutation was not a robust default. The sharp unresolved question after v17 was whether
`visitonly` was truly better on the home benchmark or whether `querygrad` remained the safer
choice once transfer was matched carefully.

Selectors:

- `visitonly`: `z(task_visits)`
- `querygrad`: `z(task_visits) + z(task_grad) + z(query_grad)`
- `querygradonly`: `z(query_grad)`

Visible GPU count actually used: `2`

Why `T2a`:

- `T2a` reduces `start_node_pool_size` from `2` to `1` while keeping writer density fixed, which stresses routing/retrieval robustness without changing both ingress diversity and writer density at once.

## Regimes

- Core: `writers/train=2`, eval at `2/6/10`, `start_node_pool_size=2`, `query_ttl=2..3`, `max_rollout_steps=12`, `steps=[250, 250, 300, 300, 400, 600, 4900]`
- T1: `writers/train=4`, eval at `4/8/12/14`, `start_node_pool_size=2`, `steps=[250, 250, 300, 300, 400, 600, 6400]`
- T2a: `writers/train=4`, eval at `4/8/12/14`, `start_node_pool_size=1`, `steps=[250, 250, 300, 300, 400, 600, 6400]`

## Configs

- Core V/Q/G: [v18_core_visitonly_long.yaml](/home/catid/gnn/configs/v18_core_visitonly_long.yaml), [v18_core_querygrad_long.yaml](/home/catid/gnn/configs/v18_core_querygrad_long.yaml), [v18_core_querygradonly_long.yaml](/home/catid/gnn/configs/v18_core_querygradonly_long.yaml)
- T1 V/Q/G: [v18_transfer_t1_visitonly_long.yaml](/home/catid/gnn/configs/v18_transfer_t1_visitonly_long.yaml), [v18_transfer_t1_querygrad_long.yaml](/home/catid/gnn/configs/v18_transfer_t1_querygrad_long.yaml), [v18_transfer_t1_querygradonly_long.yaml](/home/catid/gnn/configs/v18_transfer_t1_querygradonly_long.yaml)
- T2a V/Q/G: [v18_transfer_t2a_visitonly_long.yaml](/home/catid/gnn/configs/v18_transfer_t2a_visitonly_long.yaml), [v18_transfer_t2a_querygrad_long.yaml](/home/catid/gnn/configs/v18_transfer_t2a_querygrad_long.yaml), [v18_transfer_t2a_querygradonly_long.yaml](/home/catid/gnn/configs/v18_transfer_t2a_querygradonly_long.yaml)

## Core Summary

| Arm | Count | Best Val | Last Val | K2 | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| visitonly | 6 | 0.1042 ± 0.0437 | 0.0764 ± 0.0487 | 0.0885 ± 0.0520 | 0.1042 ± 0.0365 | 0.0677 ± 0.0352 |
| querygrad | 4 | 0.1250 ± 0.0340 | 0.0625 ± 0.0538 | 0.0781 ± 0.0403 | 0.0977 ± 0.0267 | 0.0508 ± 0.0197 |
| querygradonly | 4 | 0.1458 ± 0.0417 | 0.0729 ± 0.0625 | 0.0977 ± 0.0449 | 0.0820 ± 0.0150 | 0.0508 ± 0.0197 |

## T1 Summary

| Arm | Count | Best Val | Last Val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| visitonly_t1 | 6 | 0.1609 ± 0.0616 | 0.0972 ± 0.0340 | 0.1120 ± 0.0208 | 0.1016 ± 0.0718 | 0.1146 ± 0.0653 | 0.1120 ± 0.0269 |
| querygrad_t1 | 6 | 0.1597 ± 0.0554 | 0.0764 ± 0.0170 | 0.1094 ± 0.0221 | 0.1016 ± 0.0690 | 0.1120 ± 0.0588 | 0.1016 ± 0.0450 |
| querygradonly_t1 | 4 | 0.1771 ± 0.0399 | 0.0729 ± 0.0208 | 0.0859 ± 0.0325 | 0.1133 ± 0.0830 | 0.0859 ± 0.0705 | 0.0859 ± 0.0372 |

## T2a Summary

| Arm | Count | Best Val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- |
| visitonly_t2a | 2 | 0.2083 ± 0.0589 | 0.1719 ± 0.0442 | 0.1250 ± 0.0000 | 0.1641 ± 0.0331 | 0.1328 ± 0.0773 |
| querygrad_t2a | 2 | 0.1458 ± 0.0295 | 0.1484 ± 0.0110 | 0.1328 ± 0.0994 | 0.1484 ± 0.0110 | 0.1484 ± 0.0110 |

## Key Differences

- Core `V - Q` on `K6`: `0.0065` with bootstrap CI [`-0.0273`, `0.0443`]
- T1 `V - Q` on `K12`: `0.0026` with bootstrap CI [`-0.0625`, `0.0625`]
- Core `Q - G` on `K6`: `0.0156` with bootstrap CI [`-0.0117`, `0.0391`]
- T1 `Q - G` on `K12`: `0.0260` with bootstrap CI [`-0.0534`, `0.0924`]
- T2a `V - Q` on `K12`: `0.0156` with bootstrap CI [`-0.0156`, `0.0469`]

## Mechanism Notes

- Core selector predictiveness:
  - `visitonly` score->usefulness correlation: `0.4924 ± 0.1010`
  - `querygrad` score->usefulness correlation: `0.5088 ± 0.1403`
  - `querygradonly` score->usefulness correlation: `0.3882 ± 0.0787`
- T1 selector predictiveness:
  - `visitonly`: `0.4880 ± 0.0893`
  - `querygrad`: `0.4710 ± 0.1492`
  - `querygradonly`: `0.3178 ± 0.0998`
- Core task-only coverage at step 10 is already high for both main selectors: `visitonly` visit/grad = `0.9583`/`0.9583`, `querygrad` visit/grad = `1.0000`/`1.0000`. That means the selector gap is not mainly an early-coverage effect.
- T1 task-only coverage at step 10 is saturated for both main selectors: `visitonly` visit/grad = `1.0000`/`1.0000`, `querygrad` visit/grad = `1.0000`/`1.0000`. Transfer differences therefore come from late-stage split selection and stability, not from broader early exploration.

## Follow-Up Choice

The initial 24-run matrix looked like case 1: `visitonly` was ahead on the core benchmark, while
T1 was too close to call between `visitonly` and `querygrad`. I therefore ran:

- 2 more `visitonly` core seeds to test whether the core win was stable rather than a 4-seed blip
- 2 more `querygrad` T1 seeds because it was the strongest transfer runner-up
- 2 T2a seeds each for `visitonly` and `querygrad` to stress selector robustness with `start_node_pool_size=1`
- then 2 more `visitonly` T1 seeds to balance the T1 comparison at 6-vs-6 before making a final selector call

This was the smallest follow-up set that made the transfer conclusion fair instead of comparing a
6-seed `querygrad` result to a 4-seed `visitonly` result.

## Selector Diagnosis

- Core benchmark: `visitonly` is the better selector on the dense home metrics that matter most for this task. It keeps the best `K6` and `K10` means after the follow-up seeds.
- T1 transfer: once the comparison is balanced at 6-vs-6 seeds, `visitonly` is no worse than `querygrad` and is slightly better on best/last validation and on `K4`, `K12`, and `K14`, with `K8` effectively tied.
- `querygradonly`: it remains mostly a best-val artifact. It can peak well in training, but it loses on stability and on the denser evals that matter for default-selector choice.
- T2a stress: the two-seed stress round is mixed and noisy, but it does not restore a clear transfer-safety advantage for `querygrad`.

## Recommendation

The balanced v18 evidence supports `visitonly` as the new default selector. It wins the core long benchmark on the main dense-eval metrics and does not lose the matched transfer rounds.

The best next experiment after v18 is to adopt the winning selector on the larger benchmark rather
than returning to more mutation variants immediately.

## Per-Seed Core Runs

| Arm | Seed | Best Val | Last Val | Last5 | Drop | k2 | k6 | k10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| querygrad | 1234 | 0.1250 | 0.0417 | 0.0750 | 0.0833 | 0.0625 | 0.1250 | 0.0312 |
| querygrad | 2234 | 0.1667 | 0.1250 | 0.1000 | 0.0417 | 0.1250 | 0.0625 | 0.0469 |
| querygrad | 3234 | 0.0833 | 0.0000 | 0.0417 | 0.0833 | 0.0312 | 0.0938 | 0.0781 |
| querygrad | 4234 | 0.1250 | 0.0833 | 0.0917 | 0.0417 | 0.0938 | 0.1094 | 0.0469 |
| querygradonly | 1234 | 0.1667 | 0.1250 | 0.1083 | 0.0417 | 0.1250 | 0.0938 | 0.0312 |
| querygradonly | 2234 | 0.1667 | 0.1250 | 0.1000 | 0.0417 | 0.1250 | 0.0625 | 0.0469 |
| querygradonly | 3234 | 0.0833 | 0.0000 | 0.0417 | 0.0833 | 0.0312 | 0.0938 | 0.0781 |
| querygradonly | 4234 | 0.1667 | 0.0417 | 0.0667 | 0.1250 | 0.1094 | 0.0781 | 0.0469 |
| visitonly | 1234 | 0.1250 | 0.0417 | 0.0750 | 0.0833 | 0.0625 | 0.1250 | 0.0312 |
| visitonly | 2234 | 0.1667 | 0.1667 | 0.1000 | 0.0000 | 0.1719 | 0.1562 | 0.1250 |
| visitonly | 3234 | 0.0417 | 0.0417 | 0.0083 | 0.0000 | 0.0156 | 0.0625 | 0.0938 |
| visitonly | 4234 | 0.1250 | 0.0833 | 0.0917 | 0.0417 | 0.0938 | 0.1094 | 0.0469 |
| visitonly | 5234 | 0.0833 | 0.0417 | 0.0333 | 0.0417 | 0.0781 | 0.1094 | 0.0469 |
| visitonly | 6234 | 0.0833 | 0.0833 | 0.0583 | 0.0000 | 0.1094 | 0.0625 | 0.0625 |

## Per-Seed T1 Runs

| Arm | Seed | Best Val | Last Val | Last5 | Drop | k4 | k8 | k12 | k14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| querygrad_t1 | 1234 | 0.1250 | 0.0833 | 0.0583 | 0.0417 | 0.1094 | 0.0938 | 0.0938 | 0.0781 |
| querygrad_t1 | 2234 | 0.2500 | 0.0833 | 0.1083 | 0.1667 | 0.1250 | 0.2344 | 0.2031 | 0.1562 |
| querygrad_t1 | 3234 | 0.1250 | 0.0833 | 0.0670 | 0.0417 | 0.1406 | 0.0781 | 0.0469 | 0.1406 |
| querygrad_t1 | 4234 | 0.1250 | 0.0417 | 0.0500 | 0.0833 | 0.0938 | 0.0312 | 0.1562 | 0.1250 |
| querygrad_t1 | 5234 | 0.2083 | 0.0833 | 0.1000 | 0.1250 | 0.1094 | 0.0781 | 0.0625 | 0.0469 |
| querygrad_t1 | 6234 | 0.1250 | 0.0833 | 0.0833 | 0.0417 | 0.0781 | 0.0938 | 0.1094 | 0.0625 |
| querygradonly_t1 | 1234 | 0.1250 | 0.0833 | 0.0583 | 0.0417 | 0.0781 | 0.0156 | 0.0312 | 0.0625 |
| querygradonly_t1 | 2234 | 0.2083 | 0.0417 | 0.1167 | 0.1667 | 0.0938 | 0.2188 | 0.1875 | 0.1250 |
| querygradonly_t1 | 3234 | 0.2083 | 0.0833 | 0.1417 | 0.1250 | 0.0469 | 0.1094 | 0.0469 | 0.1094 |
| querygradonly_t1 | 4234 | 0.1667 | 0.0833 | 0.0833 | 0.0833 | 0.1250 | 0.1094 | 0.0781 | 0.0469 |
| visitonly_t1 | 1234 | 0.2083 | 0.1250 | 0.1583 | 0.0833 | 0.0938 | 0.0625 | 0.0469 | 0.1094 |
| visitonly_t1 | 2234 | 0.2500 | 0.0833 | 0.1083 | 0.1667 | 0.1250 | 0.2344 | 0.2031 | 0.1562 |
| visitonly_t1 | 3234 | 0.1739 | 0.1250 | 0.0841 | 0.0489 | 0.1406 | 0.1094 | 0.0312 | 0.1094 |
| visitonly_t1 | 4234 | 0.1250 | 0.0417 | 0.0500 | 0.0833 | 0.0938 | 0.0312 | 0.1562 | 0.1250 |
| visitonly_t1 | 5234 | 0.0833 | 0.0833 | 0.0583 | 0.0000 | 0.0938 | 0.0625 | 0.1250 | 0.0781 |
| visitonly_t1 | 6234 | 0.1250 | 0.1250 | 0.0917 | 0.0000 | 0.1250 | 0.1094 | 0.1250 | 0.0938 |

## Per-Seed T2a Runs

| Arm | Seed | Best Val | Last Val | Last5 | Drop | k4 | k8 | k12 | k14 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| querygrad_t2a | 5234 | 0.1667 | 0.0833 | 0.1167 | 0.0833 | 0.1562 | 0.0625 | 0.1562 | 0.1406 |
| querygrad_t2a | 6234 | 0.1250 | 0.0833 | 0.0500 | 0.0417 | 0.1406 | 0.2031 | 0.1406 | 0.1562 |
| visitonly_t2a | 5234 | 0.2500 | 0.2083 | 0.1750 | 0.0417 | 0.2031 | 0.1250 | 0.1875 | 0.1875 |
| visitonly_t2a | 6234 | 0.1667 | 0.1250 | 0.0667 | 0.0417 | 0.1406 | 0.1250 | 0.1406 | 0.0781 |

## Outputs

- summary JSON: [`summary_metrics_v18.json`](/home/catid/gnn/reports/summary_metrics_v18.json)
- core plot: [`v18_core_selector_bars.png`](/home/catid/gnn/reports/v18_core_selector_bars.png)
- T1 plot: [`v18_t1_selector_bars.png`](/home/catid/gnn/reports/v18_t1_selector_bars.png)
- T2 plot: [`v18_t2_selector_bars.png`](/home/catid/gnn/reports/v18_t2_selector_bars.png)
- core stability/predictiveness: [`v18_core_stability_predictiveness.png`](/home/catid/gnn/reports/v18_core_stability_predictiveness.png)
- T1 stability/predictiveness: [`v18_t1_stability_predictiveness.png`](/home/catid/gnn/reports/v18_t1_stability_predictiveness.png)
- T2 stability/predictiveness: [`v18_t2_stability_predictiveness.png`](/home/catid/gnn/reports/v18_t2_stability_predictiveness.png)
