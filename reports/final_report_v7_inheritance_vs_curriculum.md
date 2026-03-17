# APSGNN v7 Inheritance vs Curriculum Report

## Summary

APSGNN v7 keeps the v3 first-hop router, the v4 implicit retrieval path, and the v6 hard 32-leaf benchmark, but adds the missing staged-static curriculum control:

- static full-size + bootstrap (`A`)
- staged static curriculum + bootstrap without inheritance (`B`)
- growth clone with split inheritance (`C`)
- long growth mutate follow-ups (`D`)

The central v7 result is:

- curriculum and active-set growth matter a lot: `A -> B` is a real improvement
- inheritance is not a dominant early coverage effect: `B` and `C` have essentially identical task-only coverage in the initial 3-seed matrix
- inheritance still appears to help once the final 32-node stage is long enough:
  - `C` beats `B` on mean checkpoint evals at the 3-seed budget
  - `C long` beats `B long` on best val, last val, `K2`, `K6`, and `K10`
- mutation is not a clean upgrade over clone:
  - it stays below clone on best val and `K10`
  - it becomes competitive on `K6`
  - but the evidence is not strong enough to replace clone growth with mutate growth

Visible GPU count on this machine was `2`, so all DDP runs used `torchrun --standalone --nproc_per_node=2`.

## What Changed From v6

v6 showed that a harder benchmark and task-only coverage accounting were necessary, but it still lacked the key control:

- a staged-static curriculum that activates `4 -> 8 -> 16 -> 32` nodes on the same schedule as growth
- identical projected targets and coverage bootstrap
- no split inheritance and no mutation

v7 adds that control and extends the hard-regime study in three ways:

- `3` seeded runs each for `A`, `B`, and `C`
- a longer confirmatory pair for `B` and `C` with a much larger final-stage budget
- two long mutate runs to answer whether mutation becomes useful once inheritance itself starts to look plausible

## Hard Benchmark

Common task setup:

- final compute leaves: `32`
- output node: `0`
- task family: write-then-query memory routing
- projected targets by stage:
  - `4` active nodes: each node owns `8` final leaves
  - `8` active nodes: each node owns `4` final leaves
  - `16` active nodes: each node owns `2` final leaves
  - `32` active nodes: each node owns `1` final leaf
- writers per episode during training: `2`
- start-node pool size: `2`
- query TTL: `2..3`
- max rollout steps: `12`
- task-only coverage metrics exclude bootstrap packets

Why this benchmark is harder than v5:

- final leaf space doubled again relative to the reduced v5 setting
- task packets enter through only `2` ingress nodes
- task packet count stays low
- TTL is tight enough that task traffic is sparse without curriculum/coverage support

## Stage Schedule and Budget

Initial 9-run matrix:

- total steps: `3000`
- schedule: `250 + 350 + 500 + 1900`
- stages: `4 -> 8 -> 16 -> 32`
- eval interval: `300`

Long follow-up runs:

- total steps: `4000`
- schedule: `250 + 350 + 500 + 2900`
- same stage order and same hard benchmark

The final `32`-node stage is therefore:

- `63%` of training in the initial matrix
- `72.5%` of training in the long follow-up

That longer final stage is what made the inheritance question more interpretable.

## Why Staged Static Is The Key Baseline

`B` isolates:

- active-set curriculum
- projected stage targets
- clockwise stage bootstrap

without:

- copying parent weights into children
- preserving function through splits
- mutation

So:

- `A vs B` tests curriculum/coverage
- `B vs C` tests inheritance
- `C vs D` tests mutation

That was the missing control in v6.

## Exact Configs

Initial 3-seed matrix:

- [v7_static_bootstrap_hard.yaml](/home/catid/gnn/configs/v7_static_bootstrap_hard.yaml)
- [v7_staged_static_hard.yaml](/home/catid/gnn/configs/v7_staged_static_hard.yaml)
- [v7_growth_clone_hard.yaml](/home/catid/gnn/configs/v7_growth_clone_hard.yaml)

Long follow-up:

- [v7_staged_static_hard_long.yaml](/home/catid/gnn/configs/v7_staged_static_hard_long.yaml)
- [v7_growth_clone_hard_long.yaml](/home/catid/gnn/configs/v7_growth_clone_hard_long.yaml)
- [v7_growth_mutate_hard_long.yaml](/home/catid/gnn/configs/v7_growth_mutate_hard_long.yaml)

## Initial 9-Run Matrix

| Run | Best val acc | Last val acc | Eval `K2` | Eval `K6` | Best val query 1-hop | Task visit @10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A s1234 | `0.1042` | `0.0833` | `0.0547` | `0.0547` | `0.4375` | `0.4688` |
| A s2234 | `0.0625` | `0.0417` | `0.0391` | `0.0625` | `0.2292` | `0.5938` |
| A s3234 | `0.0625` | `0.0208` | `0.0234` | `0.0781` | `0.5625` | `0.5000` |
| B s1234 | `0.1250` | `0.1042` | `0.0781` | `0.0938` | `0.3542` | `1.0000` |
| B s2234 | `0.1042` | `0.0833` | `0.0781` | `0.0547` | `0.5833` | `1.0000` |
| B s3234 | `0.0833` | `0.0208` | `0.0312` | `0.0469` | `0.5000` | `1.0000` |
| C s1234 | `0.1250` | `0.1250` | `0.0703` | `0.0938` | `0.5208` | `1.0000` |
| C s2234 | `0.1042` | `0.1042` | `0.1094` | `0.0859` | `0.5000` | `1.0000` |
| C s3234 | `0.0625` | `0.0625` | `0.0781` | `0.1016` | `0.4375` | `1.0000` |

## A/B/C Mean±Std Across Seeds

| Arm | Best val acc | Last val acc | Eval `K2` | Eval `K6` | Best val query 1-hop |
| --- | ---: | ---: | ---: | ---: | ---: |
| A static+bootstrap | `0.0764 ± 0.0241` | `0.0486 ± 0.0318` | `0.0391 ± 0.0156` | `0.0651 ± 0.0119` | `0.4097 ± 0.1684` |
| B staged-static | `0.1042 ± 0.0208` | `0.0694 ± 0.0434` | `0.0625 ± 0.0271` | `0.0651 ± 0.0251` | `0.4792 ± 0.1160` |
| C growth-clone | `0.0972 ± 0.0318` | `0.0972 ± 0.0318` | `0.0859 ± 0.0207` | `0.0938 ± 0.0078` | `0.4861 ± 0.0434` |

What this means:

- `A -> B` is the big jump
- `B -> C` is not a clear win on mean **best val**
- `B -> C` is a cleaner win on **last-checkpoint stability** and matched checkpoint evals

That is why the follow-up branch was a longer final-stage confirmatory pair rather than mutation-first.

## Follow-Up Runs

| Run | Best val acc | Last val acc | Eval `K2` | Eval `K6` | Eval `K10` | Best val query 1-hop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| B long s4234 | `0.1250` | `0.0417` | `0.0703` | `0.0938` | `0.0547` | `0.5833` |
| C long s4234 | `0.1458` | `0.1458` | `0.1094` | `0.1016` | `0.1250` | `0.6875` |
| D long s4234 | `0.1042` | `0.1042` | `0.1094` | `0.1406` | `0.1094` | `0.4792` |
| D long s5234 | `0.1250` | `0.1250` | `0.1016` | `0.1016` | `0.1172` | `0.5000` |

Long mutate mean across `2` seeds:

- best val acc: `0.1146 ± 0.0147`
- eval `K2`: `0.1055 ± 0.0055`
- eval `K6`: `0.1211 ± 0.0276`
- eval `K10`: `0.1133 ± 0.0055`

## Coverage Diagnostics

### Coverage explains most of `A -> B`

At step `10`, mean task-only coverage for the initial matrix was:

- `A` task visit coverage: `0.5208`
- `B` task visit coverage: `1.0000`
- `C` task visit coverage: `1.0000`

The same pattern holds for task gradient coverage and `>= 5` task visits.

So curriculum plus active-set growth is doing real work. The missing staged-static control from v6 was essential.

### Coverage does **not** explain all of `B -> C`

In the initial seeded matrix:

- `B` and `C` have effectively identical task-only coverage curves
- both reach `1.0` visit and gradient coverage almost immediately at the small-node stages
- yet `C` has better last-checkpoint accuracy and better mean checkpoint evals

In the longer final-stage comparison:

- final-stage task visit coverage at local step `10`
  - `B long`: `0.5313`
  - `C long`: `0.6875`
  - `D long mean`: `0.5156`
- final-stage task visit coverage at local step `100`
  - `B long`: `1.0000`
  - `C long`: `1.0000`
  - `D long mean`: `0.9844`

So clone inheritance does get a modest early final-stage traffic advantage, but that edge is small and disappears by step `100`. The stronger difference is downstream optimization:

- `C long` keeps improving through the final stage and finishes at its best checkpoint
- `B long` peaks earlier and degrades by the last checkpoint

That is the clearest v7 sign of value from function-preserving inheritance.

### Mutation diagnostics

Mutation was real, not a no-op:

- mutated children received traffic in all `16 / 16` final-stage sibling pairs in both long runs
- mutated child traffic exceeded exact sibling traffic in:
  - `4 / 16` pairs on seed `4234`
  - `6 / 16` pairs on seed `5234`
- mutated child gradient exceeded exact sibling gradient in:
  - `6 / 16` pairs on both seeds
- sibling divergence:
  - seed `4234`: `0.00923`
  - seed `5234`: `0.00828`

So mutate does create meaningful diversity. It just does not yet produce a clear across-metric win over clone inheritance.

## Interpretation

### Q1. Curriculum vs growth

Yes, growth clone beats static full-size bootstrap, but only partially and only once the right baseline is in place.

The important comparison is `B vs C`, not `A vs C`:

- on the initial 3-seed matrix, `C` does **not** beat `B` on mean best val accuracy
- but `C` does beat `B` on mean `K2`, mean `K6`, and last-checkpoint stability
- on the longer confirmatory seed, `C long` beats `B long` on best val, last val, `K2`, `K6`, and `K10`

So the answer is not “growth crushes curriculum,” but it is also no longer “staged curriculum is enough.”

### Q2. Inheritance

The inheritance gain is real but modest.

The cleanest diagnosis is:

- inheritance does **not** dominate early coverage
- inheritance does appear to improve late final-stage optimization and retention
- that benefit is large enough to show up in checkpoint eval means and in the longer final-stage follow-up

The current evidence supports a narrow claim:

- function-preserving split inheritance helps beyond staged curriculum + bootstrap
- the gain is late-emerging and not huge

### Q3. Coverage

Coverage explains most of the `A -> B` improvement, but not all of `B -> C`.

Evidence:

- `A -> B` sharply improves task-only visit and gradient coverage
- `B` and `C` have nearly identical initial task-only coverage curves
- `C` still ends up better on eval means and on the longer final-stage run

So coverage is a major mechanism, but it does **not** fully explain the inheritance effect.

### Q4. Mutation

Not cleanly.

Mutation became more competitive on the extended regime than it looked in v6:

- long mutate mean `K6` (`0.1211`) is above `C long` (`0.1016`)
- long mutate `K2` is close to `C long`

But mutation still does not clearly beat clone growth where it matters most:

- best val remains below `C long`
- `K10` remains below `C long`
- the evidence is only `2` mutate seeds versus `1` long clone seed

So the v7 answer is:

- mutation is no longer clearly useless
- but it is not yet a justified replacement for clone inheritance

## Artifacts

- Summary metrics: [summary_metrics_v7.json](/home/catid/gnn/reports/summary_metrics_v7.json)
- Mean/std accuracy bars: [v7_mean_std_accuracy_bars.png](/home/catid/gnn/reports/v7_mean_std_accuracy_bars.png)
- Task visit coverage curves: [v7_task_visit_coverage_curves.png](/home/catid/gnn/reports/v7_task_visit_coverage_curves.png)
- Task gradient coverage curves: [v7_task_gradient_coverage_curves.png](/home/catid/gnn/reports/v7_task_gradient_coverage_curves.png)
- Final-stage coverage: [v7_final_stage_coverage.png](/home/catid/gnn/reports/v7_final_stage_coverage.png)
- Stage-wise training curves: [v7_stagewise_training_curves.png](/home/catid/gnn/reports/v7_stagewise_training_curves.png)
- Final-stage utility histograms: [v7_final_stage_utility_histograms.png](/home/catid/gnn/reports/v7_final_stage_utility_histograms.png)
- Inheritance comparison: [v7_inheritance_comparison.png](/home/catid/gnn/reports/v7_inheritance_comparison.png)

Primary run directories:

- [20260317-210440-v7-static-bootstrap-hard-s1234](/home/catid/gnn/runs/20260317-210440-v7-static-bootstrap-hard-s1234)
- [20260317-210702-v7-static-bootstrap-hard-s2234](/home/catid/gnn/runs/20260317-210702-v7-static-bootstrap-hard-s2234)
- [20260317-210924-v7-static-bootstrap-hard-s3234](/home/catid/gnn/runs/20260317-210924-v7-static-bootstrap-hard-s3234)
- [20260317-211145-v7-staged-static-hard-s1234](/home/catid/gnn/runs/20260317-211145-v7-staged-static-hard-s1234)
- [20260317-211442-v7-staged-static-hard-s2234](/home/catid/gnn/runs/20260317-211442-v7-staged-static-hard-s2234)
- [20260317-211739-v7-staged-static-hard-s3234](/home/catid/gnn/runs/20260317-211739-v7-staged-static-hard-s3234)
- [20260317-212036-v7-growth-clone-hard-s1234](/home/catid/gnn/runs/20260317-212036-v7-growth-clone-hard-s1234)
- [20260317-212333-v7-growth-clone-hard-s2234](/home/catid/gnn/runs/20260317-212333-v7-growth-clone-hard-s2234)
- [20260317-212630-v7-growth-clone-hard-s3234](/home/catid/gnn/runs/20260317-212630-v7-growth-clone-hard-s3234)
- [20260317-213303-v7-staged-static-hard-long-s4234](/home/catid/gnn/runs/20260317-213303-v7-staged-static-hard-long-s4234)
- [20260317-213629-v7-growth-clone-hard-long-s4234](/home/catid/gnn/runs/20260317-213629-v7-growth-clone-hard-long-s4234)
- [20260317-214058-v7-growth-mutate-hard-long-s4234](/home/catid/gnn/runs/20260317-214058-v7-growth-mutate-hard-long-s4234)
- [20260317-214422-v7-growth-mutate-hard-long-s5234](/home/catid/gnn/runs/20260317-214422-v7-growth-mutate-hard-long-s5234)

## Best Next Step

The best next experiment is **utility-based split selection**.

v7 is now strong enough to say:

- staged curriculum alone is not the full story
- clone inheritance has a modest late-stage benefit
- mutation is not yet clearly worth the extra instability

That makes utility-based split selection the most defensible next move:

- keep the hard 32-leaf benchmark
- keep the staged-static baseline
- keep the long final-stage budget
- use the final-stage visit/gradient utility signals already logged in v7
- ask whether targeted splitting beats uniform clone growth

Pruning/merge and crossover should wait until that sharper growth mechanism is validated.
