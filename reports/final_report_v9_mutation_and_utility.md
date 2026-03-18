# APSGNN v9: Mutation And Utility

## What Changed From v8

V9 keeps the v8 hard selective-growth benchmark and asks two narrower questions: whether utility+mutation (UM) is a real upgrade over utility-only selective growth (U), and which parts of the utility score actually matter. The core architecture is unchanged from the v8 winner: the v3 key-centric first-hop router, the v4 implicit learned retrieval path, and the v8 selective-growth machinery with task-only coverage/utility accounting.

The main long schedule uses 6500 optimizer steps with stage steps `[250, 250, 300, 300, 400, 600, 4400]` for the selective active-node schedule `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`. Follow-up and transfer confirmation runs use an 8000-step `longplus` schedule with the final stage extended to 5900 steps. Bootstrap remains 75 steps at each stage start and is excluded from task-only coverage and utility scoring.

## Benchmark

- Final compute leaves: `32`
- Output node: `0`
- Task: write-then-query memory routing
- Core train writers/episode: `2`
- Core eval writers/episode: `2, 6, 10`
- Start node pool size: `2`
- Query TTL: `2..3`
- Max rollout steps: `12`
- Visible GPUs used: `2`

## Utility Formulas

- Full score `U`: `z(visits) + z(grad) + 0.75 * z(success)`
- No-success `US`: `z(visits) + z(grad)`
- No-grad `UG`: `z(visits) + 0.75 * z(success)`

Mutation policy for `UM`: one child of each selected split is mutated with a modest local perturbation to child-local routing/delay heads and small MLP weights, preserving overall function as much as possible.

## Exact Configs Used

- Core anchors: `configs/v9_staged_static_selective_long.yaml`, `configs/v9_clone_selective_long.yaml`
- Core utility family: `configs/v9_utility_selective_long.yaml`, `configs/v9_utility_mutate_long.yaml`, `configs/v9_utility_nosuccess_long.yaml`, `configs/v9_utility_nograd_long.yaml`
- Longplus follow-up: `configs/v9_utility_selective_longplus.yaml`, `configs/v9_utility_mutate_longplus.yaml`, `configs/v9_utility_nosuccess_longplus.yaml`, `configs/v9_utility_nograd_longplus.yaml`
- Transfer H1: `configs/v9_transfer_h1_utility_longplus.yaml`, `configs/v9_transfer_h1_utility_mutate_longplus.yaml`

## Core Mean/Std Summary

| Arm | Seeds | Best Val | Last Val | Best K2 | Best K6 | Best K10 |
| --- | --- | --- | --- | --- | --- | --- |
| B | 2 | 0.5417 ± 0.0884 | 0.5417 ± 0.0884 | 0.4844 ± 0.0884 | 0.4531 ± 0.0442 | 0.4688 ± 0.1768 |
| C | 4 | 0.4948 ± 0.0462 | 0.4948 ± 0.0462 | 0.4688 ± 0.0556 | 0.4453 ± 0.0578 | 0.4102 ± 0.1334 |
| U | 6 | 0.5694 ± 0.1026 | 0.5556 ± 0.1100 | 0.5365 ± 0.0660 | 0.4714 ± 0.0722 | 0.4141 ± 0.1386 |
| UM | 6 | 0.5694 ± 0.1500 | 0.5243 ± 0.1516 | 0.5365 ± 0.1332 | 0.4635 ± 0.0622 | 0.4323 ± 0.0888 |
| US | 4 | 0.5677 ± 0.0548 | 0.5312 ± 0.0859 | 0.5234 ± 0.0942 | 0.4492 ± 0.0780 | 0.4648 ± 0.1400 |
| UG | 4 | 0.5260 ± 0.0922 | 0.5104 ± 0.0876 | 0.5156 ± 0.1460 | 0.4648 ± 0.0603 | 0.4570 ± 0.1310 |

## Per-Seed Core And Follow-Up Runs

| Arm | Seed | Best Val | Last Val | Best K2 | Best K6 | Best K10 |
| --- | --- | --- | --- | --- | --- | --- |
| B | 1234 | 0.4792 | 0.4792 | 0.4219 | 0.4844 | 0.3438 |
| B | 2234 | 0.6042 | 0.6042 | 0.5469 | 0.4219 | 0.5938 |
| C | 1234 | 0.4792 | 0.4792 | 0.4688 | 0.3750 | 0.2656 |
| C | 2234 | 0.5208 | 0.5208 | 0.4219 | 0.4219 | 0.5156 |
| C | 3234 | 0.4375 | 0.4375 | 0.4375 | 0.5000 | 0.5312 |
| C | 4234 | 0.5417 | 0.5417 | 0.5469 | 0.4844 | 0.3281 |
| U | 1234 | 0.4792 | 0.4792 | 0.5000 | 0.4844 | 0.2500 |
| U | 2234 | 0.5417 | 0.5208 | 0.4688 | 0.3438 | 0.4062 |
| U | 3234 | 0.5625 | 0.5208 | 0.5781 | 0.5000 | 0.5000 |
| U | 4234 | 0.4583 | 0.4375 | 0.4688 | 0.4375 | 0.3281 |
| UG | 1234 | 0.4583 | 0.4583 | 0.4375 | 0.4688 | 0.2812 |
| UG | 2234 | 0.4375 | 0.4167 | 0.3594 | 0.4062 | 0.4375 |
| UG | 3234 | 0.5833 | 0.5625 | 0.5781 | 0.5469 | 0.5312 |
| UG_followup | 5234 | 0.6250 | 0.6042 | 0.6875 | 0.4375 | 0.5781 |
| UM | 1234 | 0.3750 | 0.3750 | 0.4219 | 0.4375 | 0.2812 |
| UM | 2234 | 0.5000 | 0.3958 | 0.3750 | 0.3594 | 0.4531 |
| UM | 3234 | 0.6250 | 0.5208 | 0.5625 | 0.4844 | 0.5000 |
| UM | 4234 | 0.4583 | 0.4375 | 0.4844 | 0.4531 | 0.3906 |
| UM_followup | 5234 | 0.6875 | 0.6875 | 0.6719 | 0.5312 | 0.5312 |
| UM_followup | 6234 | 0.7708 | 0.7292 | 0.7031 | 0.5156 | 0.4375 |
| US | 1234 | 0.5208 | 0.4375 | 0.4219 | 0.4375 | 0.3125 |
| US | 2234 | 0.5417 | 0.5208 | 0.4688 | 0.3438 | 0.4062 |
| US | 3234 | 0.5625 | 0.5208 | 0.5781 | 0.5000 | 0.5000 |
| US_followup | 5234 | 0.6458 | 0.6458 | 0.6250 | 0.5156 | 0.6406 |
| U_followup | 5234 | 0.6458 | 0.6458 | 0.6250 | 0.5156 | 0.6406 |
| U_followup | 6234 | 0.7292 | 0.7292 | 0.5781 | 0.5469 | 0.3594 |

## Effect Summaries

- `U - C`: last-val +0.0608 [-0.0226, +0.1510], K6 +0.0260 [-0.0495, +0.0951], K10 +0.0039 [-0.1458, +0.1589]
- `UM - U`: last-val -0.0312 [-0.1632, +0.1076], K6 -0.0078 [-0.0755, +0.0625], K10 +0.0182 [-0.1016, +0.1380]
- `U - US`: last-val +0.0243 [-0.0799, +0.1319], K6 +0.0221 [-0.0612, +0.1081], K10 -0.0508 [-0.2044, +0.1003]
- `U - UG`: last-val +0.0451 [-0.0608, +0.1562], K6 +0.0065 [-0.0716, +0.0755], K10 -0.0430 [-0.1888, +0.1133]

## Follow-Up Round

The initial 20-run matrix left `U`, `UM`, `US`, and `UG` too close to settle from the 6500-step schedule alone. I therefore chose the “late-emerging” follow-up path: two additional `U` seeds and two additional `UM` seeds on the 8000-step longplus schedule, plus one extra `US` seed and one extra `UG` seed. This isolates whether the utility-score and mutation effects mainly appear deep into the final 32-node stage.

The longplus follow-up raised the ceiling for both `U` and `UM`, but it did not flip the overall diagnosis. `UM` showed higher upside on some longplus seeds, yet the combined six-seed `U` aggregate remained more stable from best to last checkpoint (`0.0139 ± 0.0170` vs `0.0451 ± 0.0483` drop), and `U` retained a small edge on combined last-checkpoint mean.

I did not rerun random-mutate (`RM`) in v9. The initial matrix never established a clean `UM > U` advantage, so the most diagnostic follow-up was to extend `U/UM/US/UG` into a longer final-stage regime rather than spend the follow-up budget on a mutation-without-utility arm before mutation itself had cleared the stronger utility-only baseline.

## Transfer / Stress Round

I used stress regime `H1` because it increases retrieval and late-stage generalization pressure directly while keeping the selective schedule fixed:

- Train writers/episode: `4`
- Eval writers/episode: `4, 8, 12, 14`
- Same longplus stage schedule and bootstrap logic

The two strongest core contenders were `U` and `UM`, so the transfer round compares exactly those.

| Arm | Seeds | Best Val | Best K4 | Best K8 | Best K12 | Best K14 |
| --- | --- | --- | --- | --- | --- | --- |
| transfer_U | 2 | 0.6146 ± 0.0442 | 0.4766 ± 0.1215 | 0.4844 ± 0.1989 | 0.4453 ± 0.0110 | 0.4062 ± 0.0663 |
| transfer_UM | 2 | 0.5312 ± 0.0442 | 0.4219 ± 0.0221 | 0.4375 ± 0.1768 | 0.4062 ± 0.0884 | 0.4609 ± 0.0994 |

Under H1, `U` generalizes more reliably than `UM`: the mean best-checkpoint transfer accuracies are higher for `U` at `K4`, `K8`, and `K12`, while `UM` only catches up at `K14`.

## Mechanism Diagnostics

Early task-only coverage is already saturated across the selective-growth arms by step 10, so the core `U/UM/US/UG` differences are not being driven by broader early exploration. The remaining signal is late-stage allocation.

For `U`, selected parents have much higher utility than unselected eligible parents, and they also lead to more useful children:

- Selected parent utility: 1.8673 ± 0.9388
- Unselected parent utility: -1.7606 ± 0.6203
- Selected parent child usefulness: 2.2326 ± 1.6773
- Unselected parent child usefulness: 0.4072 ± 0.2505
- Parent utility vs later child usefulness correlation: 0.8044 ± 0.2219

Component-level usefulness correlations in `U`:

- Visit component: `0.9601`
- Grad component: `0.8549`
- Success component: `0.4967`
- Full score: `0.5965`

These diagnostics match the ablation results: visit and gradient signals are doing most of the work, while the success-conditioned traffic term is comparatively weak. Removing the success term (`US`) does not hurt materially. Removing the gradient term (`UG`) is somewhat worse overall, especially on late-stage stability.

Mutation diagnostics across `UM` runs:

- Mutated child usefulness win rate over its sibling: 0.2125 ± 0.2567
- Mutated child visit share: 0.2897 ± 0.1978
- Mutated child grad share: 0.3075 ± 0.1579
- Mutated child usefulness share: 0.2930 ± 0.1919
- Sibling divergence: 0.0088 ± 0.0015

Mutation is real, but it is not yet a reliable average win: mutated children do diverge, and some seeds exploit that well, but the aggregate usefulness win rate stays below 0.5 and the stability penalty is still visible in `UM`.

## Conclusions

1. **UM vs U:** `UM` is not yet a reliable upgrade over `U`. On combined means they tie on best validation (`0.5694 ± 0.1026` vs `0.5694 ± 0.1500`) but `U` is slightly better on last-checkpoint stability and transfer.
2. **Utility score parts:** the gradient term appears more important than the success-conditioned term. `US` stays essentially tied with `U`, while `UG` is weaker overall. The raw component correlations also point the same way: gradient and visit predict later child usefulness better than success traffic.
3. **Mechanism:** the `U` advantage is late-stage allocation, not early task coverage. Early task-only coverage saturates quickly for all selective arms, but the selected-vs-unselected utility and child-usefulness gaps remain strong, especially in later transitions.
4. **Stress transfer:** the core winner remains `U`. Under the harder H1 regime, `U` beats `UM` on most transfer writer densities and degrades more gracefully. There is still no evidence that mutation helps more generally without utility selection; v9 never established a strong enough `UM > U` margin to justify elevating mutation-first variants ahead of score refinement.

## Best Next Experiment

The next best move is **refining the utility score**, not switching to crossover or pruning/merge yet. The score is already doing real work, but the current success-conditioned term is not pulling its weight and mutation is still too noisy. The clean next experiment is to keep the same hard benchmark and long final-stage schedule, refine the utility target around later child usefulness or final-stage contribution, and only then reconsider conditional mutation.

## Artifacts

- Summary JSON: [summary_metrics_v9.json](/home/catid/gnn/reports/summary_metrics_v9.json)
- Mean/std bars: [v9_mean_std_accuracy_bars.png](/home/catid/gnn/reports/v9_mean_std_accuracy_bars.png)
- Stability: [v9_best_to_last_stability.png](/home/catid/gnn/reports/v9_best_to_last_stability.png)
- Visit coverage: [v9_task_visit_coverage_curves.png](/home/catid/gnn/reports/v9_task_visit_coverage_curves.png)
- Gradient coverage: [v9_task_gradient_coverage_curves.png](/home/catid/gnn/reports/v9_task_gradient_coverage_curves.png)
- Late-stage rolling validation: [v9_late_stage_rolling_validation.png](/home/catid/gnn/reports/v9_late_stage_rolling_validation.png)
- Selected vs unselected utility: [v9_selected_vs_unselected_utility.png](/home/catid/gnn/reports/v9_selected_vs_unselected_utility.png)
- Child usefulness vs parent utility: [v9_child_usefulness_vs_parent_utility.png](/home/catid/gnn/reports/v9_child_usefulness_vs_parent_utility.png)
- Mutated child diagnostics: [v9_mutated_child_vs_sibling_usefulness.png](/home/catid/gnn/reports/v9_mutated_child_vs_sibling_usefulness.png)
- Transfer comparison: [v9_transfer_comparison.png](/home/catid/gnn/reports/v9_transfer_comparison.png)
