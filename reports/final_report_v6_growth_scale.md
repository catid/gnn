# APSGNN v6 Growth Scale Report

## Summary

APSGNN v6 keeps the v3 first-hop router family, the v4 implicit learned retrieval path, and the v5 growth machinery, but scales the topology-growth study to a harder 32-leaf benchmark with stronger task-only coverage metrics.

This round answered four questions:

1. Does growth improve task-packet coverage on a benchmark where coverage is not trivially saturated?
2. Does growth clone still beat a matched static baseline once the benchmark is harder?
3. If growth helps, is the gain mostly just a coverage-bootstrap effect?
4. On the harder regime, does split+mutate help more than pure clone growth?

Visible GPU count on this machine was `2`, so all DDP runs used `torchrun --standalone --nproc_per_node=2`.

## What Changed From v5

The core v5 idea stayed intact, but the benchmark and diagnostics were strengthened:

- final leaf-home space increased from `16` to `32`
- stage growth extended from `4 -> 8 -> 16` to `4 -> 8 -> 16 -> 32`
- projected home targets were extended to match the 32-leaf tree
- clockwise transport prior and stage bootstraps were retained
- task start nodes were hardened with a restricted ingress pool so task traffic no longer covers all active nodes immediately
- coverage accounting was split into:
  - all-packet coverage
  - bootstrap-only coverage
  - task-packet-only coverage
  - query-packet-only coverage
- new harder coverage metrics were logged:
  - task visit coverage at steps `10/50/100/200`
  - task gradient coverage at steps `10/50/100/200`
  - fraction of nodes with at least `5` task visits
  - post-bootstrap coverage slope
  - time to `50% / 75% / 100%` task visit and gradient coverage
  - task-traffic entropy and Gini

No new retrieval redesign or router redesign was introduced in v6.

## Benchmark Regimes

### Common structure

- final compute leaves: `32`
- output node: `0`
- task: write-then-query memory routing
- projected home targets across stages:
  - `4` active nodes: each node owns `8` final leaves
  - `8` active nodes: each node owns `4` final leaves
  - `16` active nodes: each node owns `2` final leaves
  - `32` active nodes: each node owns `1` final leaf
- growth schedule: `4 -> 8 -> 16 -> 32`
- static schedule: `32` from the start
- stage bootstrap:
  - clockwise forced or strongly biased traffic
  - zero-delay bias
  - bootstrap packets counted separately from task coverage

### Regime M

- `writers_per_episode = 6`
- `start_node_pool_size = 8`
- `query_ttl = 3..5`
- `max_rollout_steps = 16`
- `batch_size_per_gpu = 2`

### Regime H

- `writers_per_episode = 2`
- `start_node_pool_size = 2`
- `query_ttl = 2..3`
- `max_rollout_steps = 12`
- `batch_size_per_gpu = 1`

Why Regime H is harder than v5:

- twice the final leaf count
- twice as many growth stages
- task packets enter through only `2` ingress nodes instead of all nodes
- lower task packet count and tighter TTL budget make task-only coverage genuinely sparse

## Exact Configs

Initial matrix:

- [v6_static_moderate.yaml](/home/catid/gnn/configs/v6_static_moderate.yaml)
- [v6_static_bootstrap_moderate.yaml](/home/catid/gnn/configs/v6_static_bootstrap_moderate.yaml)
- [v6_growth_clone_moderate.yaml](/home/catid/gnn/configs/v6_growth_clone_moderate.yaml)
- [v6_static_hard.yaml](/home/catid/gnn/configs/v6_static_hard.yaml)
- [v6_static_bootstrap_hard.yaml](/home/catid/gnn/configs/v6_static_bootstrap_hard.yaml)
- [v6_growth_clone_hard.yaml](/home/catid/gnn/configs/v6_growth_clone_hard.yaml)

Follow-up:

- [v6_growth_mutate_followup.yaml](/home/catid/gnn/configs/v6_growth_mutate_followup.yaml)
- `v6_static_bootstrap_hard.yaml` with `--train-steps 1800`
- `v6_growth_clone_hard.yaml` with `--train-steps 1800`

## Initial 6-Run Matrix

| Run | Best val acc | Query 1-hop | Writer 1-hop | Task visit @10 | Task visit @50 | Nodes >=5 visits @50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static / M | `0.0703` | `0.2734` | `0.5924` | `0.9375` | `1.0000` | `1.0000` |
| Static+bootstrap / M | `0.0625` | `0.3984` | `0.6497` | `0.9375` | `1.0000` | `1.0000` |
| Growth clone / M | `0.0703` | `0.0469` | `0.0273` | `0.9375` | `1.0000` | `1.0000` |
| Static / H | `0.0625` | `0.0781` | `0.1641` | `0.5938` | `0.9688` | `0.5938` |
| Static+bootstrap / H | `0.0781` | `0.3594` | `0.4609` | `0.4688` | `0.9688` | `0.5000` |
| Growth clone / H | `0.0625` | `0.0781` | `0.2031` | `0.5625` | `1.0000` | `0.5313` |

## Follow-Up Runs

The initial matrix made Regime H the more diagnostic regime:

- Regime M still saturated task coverage by step `50`
- Regime H preserved nontrivial task-only sparsity at steps `10` and `50`
- static+bootstrap vs growth clone was no longer trivially dominated by immediate full coverage

So the follow-up round focused on Regime H.

| Run | Best val acc | Query 1-hop | Writer 1-hop | Eval `K=2` acc | Eval `K=6` acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| Growth mutate / H | `0.0625` | `0.0938` | `0.2109` | `0.0391` | `0.0625` |
| Static+bootstrap / H long | `0.0781` | `0.3594` | `0.4609` | `0.0312` | `0.0391` |
| Growth clone / H long | `0.0781` | `0.1875` | `0.2109` | `0.0469` | `0.1094` |

## Coverage Diagnostics

### Task-only coverage is no longer trivial on Regime H

The v5 reduced benchmark failed because task coverage saturated immediately. Regime H fixed that:

- Static / H:
  - task visit coverage at step `10`: `0.5938`
  - nodes with at least `5` task visits at step `10`: `0.0625`
  - time to `100%` task visit coverage after bootstrap: `65`
- Static+bootstrap / H:
  - task visit coverage at step `10`: `0.4688`
  - nodes with at least `5` task visits at step `10`: `0.0625`
  - time to `100%` task visit coverage after bootstrap: `2`
- Growth clone / H:
  - task visit coverage at step `10`: `0.5625`
  - nodes with at least `5` task visits at step `10`: `0.0625`
  - time to `100%` task visit coverage after bootstrap: `0`

The main difference from v5 is that coverage is now meaningfully sparse early in training, especially under the stricter `>= 5 visits` metric.

### Growth does not dominate bootstrap on coverage alone

On the harder regime:

- growth clone beat static scratch on early task coverage
- but growth clone did **not** beat static+bootstrap cleanly on the stronger task-only coverage metrics
- the static+bootstrap run often reached the same or better post-bootstrap coverage milestones

That matters because it weakens the simple “growth helps because it covers more nodes earlier” story.

## Mutation Diagnostics

Mutation was real, not a no-op:

- mutated children received traffic in all `16` sibling pairs
- mutated children had more traffic in `7 / 16` pairs
- mutated children had more gradient in `7 / 16` pairs
- sibling divergence at the final stage was `0.00808`

So mutate did create function-preserving diversity. It just did not improve the outcome.

## Interpretation

### Q1. Coverage

Yes, but only relative to static scratch. On Regime H, growth clone improved early task-only coverage over static full-size from scratch. However, once the static full-size model got the same style of stage bootstrap, growth did not show a clean coverage advantage.

### Q2. Performance

Not cleanly. In the initial hard matrix, growth clone did **not** beat static+bootstrap:

- Static+bootstrap / H best val acc: `0.0781`
- Growth clone / H best val acc: `0.0625`

In the longer confirmatory pair, growth clone recovered to a tie on best validation accuracy:

- Static+bootstrap / H long best val acc: `0.0781`
- Growth clone / H long best val acc: `0.0781`

But the direct checkpoint evaluations show a mixed picture:

- on native hard `K=2`, static+bootstrap long remained stronger on routing and no worse on accuracy
- on heavier `K=6`, growth clone long generalized better:
  - Static+bootstrap / H long eval `K=6`: `0.0391`
  - Growth clone / H long eval `K=6`: `0.1094`

So growth did not produce a broad final win, but it may have improved robustness to higher writer load once trained longer.

### Q3. Mechanism

The best evidence now says coverage explains most of the v5-style story.

Static+bootstrap closes the gap to growth on the harder regime, and the harder task-only metrics show that the main difference between static scratch and the other regimes is better early traffic and gradient distribution. Growth does not currently show a strong extra gain once static full-size training is given the same coverage treatment.

### Q4. Mutation

No. On the harder regime, split+mutate still did not beat clone-based growth.

- best val acc stayed at `0.0625`
- eval `K=2` stayed at `0.0391`
- eval `K=6` stayed at `0.0625`

Mutation created diversity and traffic, but not a better model.

## Does v6 Support “Small Animal First”?

Not strongly enough yet.

The v6 evidence supports a narrower claim:

- task-only coverage genuinely mattered, and v5 had hidden that by using a benchmark that saturated coverage too easily
- static full-size training benefits materially from a coverage bootstrap
- once that bootstrap is present, growth clone is no longer clearly better on the hard regime

So the current evidence does **not** yet justify a strong claim that “small animal first” is intrinsically better than a well-bootstrapped full-size model. At most, v6 leaves open a weaker possibility that growth may help generalization under higher downstream load, as suggested by the `K=6` hard-checkpoint eval.

## Artifacts

- Summary metrics: [summary_metrics_v6.json](/home/catid/gnn/reports/summary_metrics_v6.json)
- Task visit coverage curves: [v6_task_visit_coverage_curves.png](/home/catid/gnn/reports/v6_task_visit_coverage_curves.png)
- Task gradient coverage curves: [v6_task_gradient_coverage_curves.png](/home/catid/gnn/reports/v6_task_gradient_coverage_curves.png)
- Accuracy bars: [v6_accuracy_comparison_bars.png](/home/catid/gnn/reports/v6_accuracy_comparison_bars.png)
- Stage-wise curves: [v6_stagewise_training_curves.png](/home/catid/gnn/reports/v6_stagewise_training_curves.png)
- Task visit histograms: [v6_task_visit_histograms.png](/home/catid/gnn/reports/v6_task_visit_histograms.png)
- Task gradient histograms: [v6_task_gradient_histograms.png](/home/catid/gnn/reports/v6_task_gradient_histograms.png)
- Hard checkpoint eval bars: [v6_checkpoint_eval_bars.png](/home/catid/gnn/reports/v6_checkpoint_eval_bars.png)

Main run directories:

- [20260317-195436-v6-static-moderate](/home/catid/gnn/runs/20260317-195436-v6-static-moderate)
- [20260317-195601-v6-static-bootstrap-moderate](/home/catid/gnn/runs/20260317-195601-v6-static-bootstrap-moderate)
- [20260317-195823-v6-growth-clone-moderate](/home/catid/gnn/runs/20260317-195823-v6-growth-clone-moderate)
- [20260317-200126-v6-static-hard](/home/catid/gnn/runs/20260317-200126-v6-static-hard)
- [20260317-200213-v6-static-bootstrap-hard](/home/catid/gnn/runs/20260317-200213-v6-static-bootstrap-hard)
- [20260317-200340-v6-growth-clone-hard](/home/catid/gnn/runs/20260317-200340-v6-growth-clone-hard)
- [20260317-200615-v6-growth-mutate-hard](/home/catid/gnn/runs/20260317-200615-v6-growth-mutate-hard)
- [20260317-200802-v6-static-bootstrap-hard-long](/home/catid/gnn/runs/20260317-200802-v6-static-bootstrap-hard-long)
- [20260317-200943-v6-growth-clone-hard-long](/home/catid/gnn/runs/20260317-200943-v6-growth-clone-hard-long)

## Best Next Step

The next best move is **something else**: tighten the benchmark around stage-final utility, not around more elaborate structure changes yet.

The v6 result is strong enough to reject “growth obviously wins because coverage is better,” but not strong enough to justify crossover, pruning/merge, or utility-based split selection immediately. The most defensible next experiment is:

- keep the hard 32-leaf setup
- keep task-only coverage metrics
- make the final stage longer and more important
- keep ingress sparse
- then test whether growth plus utility-based split selection beats the best static+bootstrap baseline

Only after that does it make sense to spend time on pruning/merge or crossover.
