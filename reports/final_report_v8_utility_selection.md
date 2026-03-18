# APSGNN v8: Utility-Based Split Selection On The Hard Benchmark

## Summary

V8 tests whether utility-based split selection adds value beyond the strongest v7 controls on the hard 32-leaf benchmark. The key change from v7 is that the growth schedule is now selective rather than pure doubling:

- active schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`
- not every eligible parent splits at each transition
- this makes split choice itself meaningful

The base architecture stays fixed:

- v3-style key-centric first-hop router
- v4 implicit learned retrieval
- v7 hard-benchmark task and task-only coverage accounting

Visible GPU count during v8 was `2`.

## Why The Selective Schedule Is Necessary

Pure doubling schedules such as `4 -> 8 -> 16 -> 32` force every eligible parent to split. Under that schedule, a “utility-based” policy cannot differ from a uniform policy. The selective schedule introduces real choice by allocating only a subset of splits at each transition:

- `4 -> 6`: `+2`
- `6 -> 8`: `+2`
- `8 -> 12`: `+4`
- `12 -> 16`: `+4`
- `16 -> 24`: `+8`
- `24 -> 32`: `+8`

Projected home targets were extended to support irregular contiguous leaf intervals. Unsplit parents keep their interval. Split parents refine their interval into two contiguous children.

## Benchmark And Training Budget

Hard benchmark:

- final compute leaves: `32`
- output node: `0`
- task: write-then-query memory routing
- train writers per episode: `2`
- eval writers per episode: `2`, `6`, `10`
- `start_node_pool_size = 2`
- `query_ttl = 2..3`
- `max_rollout_steps = 12`
- task-only coverage excludes bootstrap packets

Initial training budget:

- total steps: `4500`
- stage steps: `[250, 250, 300, 300, 400, 600, 2400]`
- final stage share: `2400 / 4500 = 53.3%`

Follow-up long budget:

- total steps: `5500`
- same stage schedule plus `1000` extra steps in the final `32`-node stage
- effective final stage share: `3400 / 5500 = 61.8%`

Bootstrap:

- `75` steps at each stage start
- strong clockwise transport bias and delay-zero bias
- bootstrap packets excluded from the task-only coverage metrics

## Arms

- `A`: static full-size + bootstrap
- `B`: staged-static selective curriculum, no inheritance
- `C`: deterministic clone growth, balanced split policy
- `R`: random-select clone growth
- `U`: utility-select clone growth
- `UM`: utility-select mutate growth, follow-up only

The comparisons are intentionally matched:

- `A vs B`: full static vs staged curriculum
- `B vs C`: inheritance benefit
- `C vs U`: utility selection benefit beyond deterministic clone growth
- `R vs U`: utility signal vs arbitrary selective growth

## Utility Score

The stage-tail utility score for eligible parents is:

`u_i = z(EMA task visits_i) + z(EMA gradient norm_i) + 0.75 * z(EMA success-conditioned traffic_i)`

Selection uses only normal task packets from the tail of the previous stage. Bootstrap traffic is excluded.

## Configs Used

- `configs/v8_static_bootstrap_hard.yaml`
- `configs/v8_staged_static_selective_hard.yaml`
- `configs/v8_clone_selective_hard.yaml`
- `configs/v8_random_selective_hard.yaml`
- `configs/v8_utility_selective_hard.yaml`
- `configs/v8_utility_mutate_hard.yaml`

## Initial 14-Run Matrix

Initial matrix summary:

| Arm | Seeds | Best Val Acc Mean ± Std | Last Val Acc Mean ± Std |
| --- | --- | ---: | ---: |
| A | 3 | `0.1319 ± 0.0120` | `0.1250 ± 0.0208` |
| B | 3 | `0.2014 ± 0.0434` | `0.1806 ± 0.0636` |
| C | 3 | `0.1736 ± 0.0241` | `0.1667 ± 0.0361` |
| R | 2 | `0.1458 ± 0.0000` | `0.1250 ± 0.0295` |
| U | 3 | `0.2153 ± 0.0434` | `0.1875 ± 0.0908` |

Takeaways from the initial matrix:

- `A -> B` is large, so staged curriculum remains a major effect.
- `U` is ahead of `C` and `R` already, but the `U vs C` gap is still modest enough to justify longer confirmatory runs.
- `R` is not enough by itself; arbitrary selective growth is weaker than utility selection.

## Follow-Up Runs

I chose a long `C/U` confirmatory pair first because the initial `U > C` signal was real but modest. After that I ran `UM` for two long seeds to answer whether mutation helps once utility selection is in place.

| Arm | Seeds | Best Val Acc Mean ± Std | Last Val Acc Mean ± Std | Best Eval K2 | Best Eval K6 | Best Eval K10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| C long | 2 | `0.3542 ± 0.0000` | `0.3021 ± 0.0737` | `0.3125 ± 0.0331` | `0.2969 ± 0.0773` | `0.3125 ± 0.0331` |
| U long | 2 | `0.3333 ± 0.0000` | `0.3021 ± 0.0442` | `0.3281 ± 0.0442` | `0.3164 ± 0.0939` | `0.2969 ± 0.0331` |
| UM long | 2 | `0.3958 ± 0.0589` | `0.3958 ± 0.0589` | `0.3398 ± 0.0166` | `0.3438 ± 0.1105` | `0.3125 ± 0.0331` |

## All Completed Runs

| Arm | Seed | Best Val | Last Val | Best K2 | Best K6 | Best K10 | Run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 1234 | `0.1458` | `0.1458` | `0.0938` | `0.1250` | `0.0781` | `20260317-233956-v8-static-bootstrap-hard-s1234` |
| A | 2234 | `0.1250` | `0.1250` | `-` | `-` | `-` | `20260317-234250-v8-static-bootstrap-hard-s2234` |
| A | 3234 | `0.1250` | `0.1042` | `-` | `-` | `-` | `20260317-234545-v8-static-bootstrap-hard-s3234` |
| B | 1234 | `0.1875` | `0.1250` | `-` | `-` | `-` | `20260317-234851-v8-staged-static-selective-hard-s1234` |
| B | 2234 | `0.1667` | `0.1667` | `-` | `-` | `-` | `20260317-235320-v8-staged-static-selective-hard-s2234` |
| B | 3234 | `0.2500` | `0.2500` | `0.1641` | `0.1406` | `0.0938` | `20260317-235750-v8-staged-static-selective-hard-s3234` |
| C | 1234 | `0.1875` | `0.1875` | `-` | `-` | `-` | `20260318-000229-v8-clone-selective-hard-s1234` |
| C | 2234 | `0.1458` | `0.1250` | `-` | `-` | `-` | `20260318-000657-v8-clone-selective-hard-s2234` |
| C | 3234 | `0.1875` | `0.1875` | `-` | `-` | `-` | `20260318-001125-v8-clone-selective-hard-s3234` |
| C long | 4234 | `0.3542` | `0.3542` | `0.3359` | `0.2422` | `0.2891` | `20260318-005603-v8-clone-selective-hard-long-s4234` |
| C long | 5234 | `0.3542` | `0.2500` | `0.2891` | `0.3516` | `0.3359` | `20260318-010635-v8-clone-selective-hard-long-s5234` |
| R | 4234 | `0.1458` | `0.1042` | `-` | `-` | `-` | `20260318-003231-v8-random-selective-hard-s4234` |
| R | 5234 | `0.1458` | `0.1458` | `0.1953` | `0.2031` | `0.1016` | `20260318-003700-v8-random-selective-hard-s5234` |
| U | 1234 | `0.2292` | `0.2292` | `-` | `-` | `-` | `20260318-004141-v8-utility-selective-hard-s1234` |
| U | 2234 | `0.2500` | `0.2500` | `-` | `-` | `-` | `20260318-004610-v8-utility-selective-hard-s2234` |
| U | 3234 | `0.1667` | `0.0833` | `-` | `-` | `-` | `20260318-005038-v8-utility-selective-hard-s3234` |
| U long | 4234 | `0.3333` | `0.2708` | `0.2969` | `0.2500` | `0.2734` | `20260318-010120-v8-utility-selective-hard-long-s4234` |
| U long | 5234 | `0.3333` | `0.3333` | `0.3594` | `0.3828` | `0.3203` | `20260318-011150-v8-utility-selective-hard-long-s5234` |
| UM long | 4234 | `0.3542` | `0.3542` | `0.3281` | `0.2656` | `0.2891` | `20260318-011918-v8-utility-mutate-hard-long-s4234` |
| UM long | 5234 | `0.4375` | `0.4375` | `0.3516` | `0.4219` | `0.3359` | `20260318-012433-v8-utility-mutate-hard-long-s5234` |

## Combined Mean ± Std Summary

For `C` and `U`, the combined summary includes the initial three seeds plus the two long follow-up seeds. `R` remains a two-seed arm by design.

| Arm | Count | Best Val | Last Val | Query First Hop | Writer First Hop | Best Eval K2 | Best Eval K6 | Best Eval K10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 3 | `0.1319 ± 0.0120` | `0.1250 ± 0.0208` | `0.6181 ± 0.0732` | `0.6215 ± 0.0262` | `0.0938 ± 0.0000` | `0.1250 ± 0.0000` | `0.0781 ± 0.0000` |
| B | 3 | `0.2014 ± 0.0434` | `0.1806 ± 0.0636` | `0.5417 ± 0.0751` | `0.5764 ± 0.0782` | `0.1641 ± 0.0000` | `0.1406 ± 0.0000` | `0.0938 ± 0.0000` |
| C | 5 | `0.2458 ± 0.1003` | `0.2208 ± 0.0867` | `0.5167 ± 0.1416` | `0.5188 ± 0.1006` | `0.3125 ± 0.0331` | `0.2969 ± 0.0773` | `0.3125 ± 0.0331` |
| R | 2 | `0.1458 ± 0.0000` | `0.1250 ± 0.0295` | `0.5625 ± 0.1179` | `0.5885 ± 0.0368` | `0.1953 ± 0.0000` | `0.2031 ± 0.0000` | `0.1016 ± 0.0000` |
| U | 5 | `0.2625 ± 0.0716` | `0.2333 ± 0.0925` | `0.5875 ± 0.0771` | `0.6000 ± 0.0785` | `0.3281 ± 0.0442` | `0.3164 ± 0.0939` | `0.2969 ± 0.0331` |

`UM` is follow-up only: best/last `0.3958 ± 0.0589`, best eval `K2=0.3398 ± 0.0166`, `K6=0.3438 ± 0.1105`, `K10=0.3125 ± 0.0331`.

## Coverage And Mechanism Diagnostics

Early task-only coverage:

- `A` is the only arm with clearly sparse early task coverage:
  - task visit coverage @10: `0.5208 ± 0.0651`
  - task gradient coverage @10: `0.5104 ± 0.0477`
- `B`, `C`, `R`, and `U` all reach task-only visit and gradient coverage `1.0` by step `10`

This matters because it means:

- `A -> B` is strongly about staged curriculum and early access
- `C -> U` is **not** explainable by broader early task coverage, since `C` and `U` already saturate the task-only coverage metric equally early

Utility diagnostics:

- `R` selected-parent utility mean: `0.3919`
- `R` unselected-parent utility mean: `-0.3254`
- `R` selected child usefulness mean: `1.5883`
- `R` unselected child usefulness mean: `0.8012`
- `R` utility/usefulness correlation mean: `0.5296`

- `U` selected-parent utility mean: `1.5947`
- `U` unselected-parent utility mean: `-1.5891`
- `U` selected child usefulness mean: `1.8274`
- `U` unselected child usefulness mean: `0.3336`
- `U` utility/usefulness correlation mean: `0.5697`

- `UM` selected-parent utility mean: `1.5410`
- `UM` unselected-parent utility mean: `-1.5617`
- `UM` selected child usefulness mean: `1.8343`
- `UM` unselected child usefulness mean: `0.3654`
- `UM` utility/usefulness correlation mean: `0.6320`

Interpretation:

- the utility score does predict later child usefulness
- random selective growth also sometimes picks good parents, but the separation is weaker
- utility selection creates a much larger selected-vs-unselected usefulness gap than random selection

## Answers To The V8 Questions

### Q1. Does utility-based split selection beat deterministic clone growth?

Yes, but modestly rather than dramatically.

Evidence:

- initial matrix: `U` best val `0.2153 ± 0.0434` vs `C` `0.1736 ± 0.0241`
- combined across all `C/U` seeds: `U` best val `0.2625 ± 0.0716` vs `C` `0.2458 ± 0.1003`
- combined last-checkpoint val: `U` `0.2333 ± 0.0925` vs `C` `0.2208 ± 0.0867`
- representative long-run checkpoint evals: `U` edges `C` on `K2` and `K6`, while `C` slightly edges `U` on `K10`

So utility selection does beat deterministic clone growth overall, but the margin is not huge.

### Q2. Is the gain mainly better final-stage allocation rather than early coverage?

Yes.

`C` and `U` both hit full task-only coverage by step `10`, so the remaining `C -> U` difference cannot be blamed on broader early exploration. The stronger explanation is better allocation of limited splits late in training. The selected-vs-unselected utility gaps and the higher child usefulness gap under `U` support that interpretation.

### Q3. How does utility compare with the controls?

- `A -> B`: staged selective curriculum is a major improvement
- `B -> C`: inheritance still helps beyond staged curriculum
- `C -> U`: utility helps beyond deterministic inheritance, but modestly
- `R -> U`: utility clearly beats arbitrary selective growth

The strongest control result is `R vs U`: if the gain were only from using any non-uniform split pattern, `R` should be close to `U`. It is not.

### Q4. Does mutation help on top of utility?

Tentatively yes, but only on limited evidence.

The two long `UM` seeds are stronger than the two long `U` seeds on best val, last val, and `K6`, and roughly tied on `K10`. That is interesting enough to keep, but it is still only a two-seed follow-up. It is not strong enough yet to replace clone-only utility growth as the default conclusion.

## Plots

The main v8 plots are:

- `reports/v8_mean_std_accuracy_bars.png`
- `reports/v8_task_visit_coverage_curves.png`
- `reports/v8_task_gradient_coverage_curves.png`
- `reports/v8_stagewise_training_curves.png`
- `reports/v8_final_stage_coverage.png`
- `reports/v8_selected_vs_unselected_utility.png`
- `reports/v8_child_usefulness_vs_parent_utility.png`

## Bottom Line

V8 answers the main question positively:

- utility-based split selection is better than staged-static curriculum
- better than deterministic clone growth on the full seeded campaign, though by a modest margin
- and better than random selective growth by a clearer margin

The evidence also suggests the utility score is capturing something real about later usefulness, not just early traffic.

## Best Next Experiment

The best next move is **refining the utility score**, not pruning/merge or crossover.

Reason:

- v8 finally shows a real selection signal
- the current score is simple and already useful
- the next highest-value experiment is to strengthen that score with stage-final utility targets or better success-conditioned terms before scaling the method up further
