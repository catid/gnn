# APSGNN v5 Growth Curriculum Report

## Summary

APSGNN v5 keeps the v4 memory path and the strong learned first-hop router family, but moves to a reduced 16-leaf benchmark to test a new question: whether a small-animal growth curriculum helps compared with training the final sparse model from scratch.

This round tested four conditions on the same reduced write-then-query task:

1. Static sparse from scratch at 16 active compute nodes
2. Static sparse with the same early clockwise/bootstrap traffic treatment as the growth runs
3. Growth by clone split: `4 -> 8 -> 16`
4. Growth by mutate split: `4 -> 8 -> 16`

Visible GPU count on this machine was `2`, so all DDP runs used `torchrun --standalone --nproc_per_node=2`.

## What Changed From v4

The v4 retrieval path and the strong learned first-hop router were kept intact. The new intervention was topology curriculum only:

- final active compute leaves: `16`
- fixed final leaf-home space with projected ancestor targets at earlier stages
- clockwise transport prior over active nodes via route-logit bias toward the successor node
- delay-logit bias toward `0`
- stage bootstrap after stage creation, with injected coverage packets, `TTL = active_node_count`, and optional forced clockwise / zero-delay execution
- deterministic splitting from `4 -> 8 -> 16`
- two split modes:
  - clone: exact inherited copy
  - mutate: copy plus small perturbation on child-local node cell feed-forward weights, start-node embedding row, and first-hop router output row
- explicit visit and gradient coverage instrumentation per stage

The cache, retrieval design, and later-hop routing machinery were not redesigned in this round.

## Reduced Benchmark

- compute leaves: `16`
- output node: `0`
- train task: same write-then-query memory-routing family as v4
- train `writers_per_episode = 6`
- evaluate at `writers_per_episode in {6, 10}`
- home assignment: fixed frozen `H16` over final leaves
- projected target semantics:
  - stage `4`: each active node owns `4` final leaves
  - stage `8`: each active node owns `2` final leaves
  - stage `16`: each active node owns `1` final leaf

This made early-stage supervision compatible with later stages instead of redefining the task after every split.

## Exact Configs

- Static sparse: [v5_static_sparse.yaml](/home/catid/gnn/configs/v5_static_sparse.yaml)
- Static bootstrap: [v5_static_bootstrap.yaml](/home/catid/gnn/configs/v5_static_bootstrap.yaml)
- Growth clone: [v5_growth_clone.yaml](/home/catid/gnn/configs/v5_growth_clone.yaml)
- Growth mutate: [v5_growth_mutate.yaml](/home/catid/gnn/configs/v5_growth_mutate.yaml)

Common settings:

- `nodes_total = 17`
- `train_steps = 1200`
- `writers_per_episode = 6`
- eval every `150` steps
- learned first-hop router enabled
- implicit learned retrieval enabled
- reduced benchmark only, no new no-cache rerun in this round

Stage schedule:

- static runs: `16`
- growth runs: `4` for `300` steps, `8` for `300` steps, `16` for `600` steps
- bootstrap window: `100` local stage steps

## Results

### Main table

| Run | K=6 acc | K=10 acc | Query 1-hop home | Writer 1-hop home | Delivery | Home->out |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Static sparse | `0.4234` | `0.3781` | `0.8578` | `0.8870` | `1.0000` | `0.9320` |
| Static + bootstrap | `0.3906` | `0.3609` | `0.8578` | `0.8888` | `1.0000` | `0.9323` |
| Growth clone | `0.4734` | `0.4422` | `0.8313` | `0.8956` | `1.0000` | `0.9300` |
| Growth mutate | `0.4516` | `0.3781` | `0.8281` | `0.8938` | `1.0000` | `0.9299` |

### Coverage diagnostics

| Run / stage | Visit @10 | Grad @10 | Time to 100% visit | Time to 100% grad |
| --- | ---: | ---: | ---: | ---: |
| Static sparse / 16 | `1.0` | `1.0` | `1` | `1` |
| Static + bootstrap / 16 | `1.0` | `1.0` | `1` | `1` |
| Growth clone / 4 | `1.0` | `1.0` | `1` | `1` |
| Growth clone / 8 | `1.0` | `1.0` | `1` | `1` |
| Growth clone / 16 | `1.0` | `1.0` | `1` | `1` |
| Growth mutate / 4 | `1.0` | `1.0` | `1` | `1` |
| Growth mutate / 8 | `1.0` | `1.0` | `1` | `1` |
| Growth mutate / 16 | `1.0` | `1.0` | `1` | `1` |

### Mutate diagnostics

The mutate split did create real diversity, but not a better final model:

- stage `8`: mutated children got more traffic in `2 / 4` sibling pairs and more gradient in `3 / 4`
- stage `16`: mutated children got more traffic in `5 / 8` sibling pairs and more gradient in `3 / 8`
- all mutated children received traffic after split
- sibling divergence stayed modest:
  - `0.00877` at stage `8`
  - `0.00833` at stage `16`

So the perturbation was real and preserved function, but it did not beat clone-only growth on task performance.

## Interpretation

### Q1. Coverage

No. On this reduced benchmark, the small-animal curriculum did **not** improve the measured cumulative visit or gradient coverage metrics relative to static sparse training. All runs reached full active-node visit coverage and full active-node gradient coverage at local stage step `1`, and were still at `1.0` by the required step-`10` measurement.

That means this benchmark is too easy for the proposed coverage metric. Random writer/query injection plus the packet budget already give broad immediate coverage, so the clockwise bootstrap does not buy additional measurable coverage here.

### Q2. Performance

Yes. Despite the flat coverage story, growth-by-splitting still improved final performance on the reduced benchmark.

- Growth clone beat static sparse at `K=6`: `0.4734` vs `0.4234`
- Growth clone beat static sparse at `K=10`: `0.4422` vs `0.3781`
- Static + bootstrap did **not** explain the gain:
  - `0.3906` at `K=6`
  - `0.3609` at `K=10`

So the evidence from this reduced setup is:

- simple early clockwise/bootstrap traffic by itself is not enough
- the growth curriculum itself is helping final performance

One notable detail is that growth clone did **not** win by producing the best final first-hop home rate. Static sparse reached slightly higher query first-hop home rate (`0.8578` vs `0.8313`), but growth clone still produced better final accuracy. The likely interpretation is that low-stage pretraining over coarse ancestor targets shapes a better memory-routing computation once the model reaches the full `16`-leaf stage, even if final first-hop accuracy is not strictly maximal.

### Q3. Exploration

No strong evidence that mutate helps. The mutate run did create sibling divergence and traffic asymmetry, so the perturbation was not a no-op, but pure clone splitting still won:

- Growth clone `K=6`: `0.4734`
- Growth mutate `K=6`: `0.4516`
- Growth clone `K=10`: `0.4422`
- Growth mutate `K=10`: `0.3781`

On this benchmark, function-preserving clone growth already seems sufficient, and the small targeted perturbation adds noise without improving the final result.

## Training Stability Notes

- Static sparse and static bootstrap improved smoothly through training.
- Growth runs stayed weak through the coarse stages and then improved sharply after the final `16`-node split.
- Clone and mutate behaved similarly until the final stage; clone recovered more cleanly.
- Delivery remained saturated at `1.0` across all completed runs.

## Artifacts

- Summary metrics: [summary_metrics_v5.json](/home/catid/gnn/reports/summary_metrics_v5.json)
- Accuracy bars: [v5_growth_accuracy_bars.png](/home/catid/gnn/reports/v5_growth_accuracy_bars.png)
- Stage curves: [v5_growth_stage_curves.png](/home/catid/gnn/reports/v5_growth_stage_curves.png)
- Coverage summary: [v5_growth_coverage_summary.png](/home/catid/gnn/reports/v5_growth_coverage_summary.png)
- Visit histograms: [v5_growth_visit_histograms.png](/home/catid/gnn/reports/v5_growth_visit_histograms.png)

Main run directories:

- Static sparse: [20260317-180520-v5-static-sparse](/home/catid/gnn/runs/20260317-180520-v5-static-sparse)
- Static bootstrap: [20260317-180818-v5-static-bootstrap](/home/catid/gnn/runs/20260317-180818-v5-static-bootstrap)
- Growth clone: [20260317-181248-v5-growth-clone](/home/catid/gnn/runs/20260317-181248-v5-growth-clone)
- Growth mutate: [20260317-181723-v5-growth-mutate](/home/catid/gnn/runs/20260317-181723-v5-growth-mutate)

## Best Next Step

The next best move is **something else**: make the coverage question harder before adding more evolutionary machinery.

The reduced benchmark saturated coverage immediately, so utility-based split selection, pruning/merge, or crossover would be premature. The more defensible next experiment is to scale the same growth curriculum to a harder regime where static sparse training does **not** already get full early coverage for free. That could mean:

- larger final leaf count
- lower effective packet budget per step
- stricter TTL budget relative to topology size
- fewer packets injected per batch relative to active nodes

Only after the coverage problem is genuinely nontrivial does it make sense to test utility-based split selection or crossover.
