# APSGNN v11: Conditional Mutation Follow-Up

Date: 2026-03-18

## What Changed From v10

v10 established that the simplified `querygrad` utility score was a stronger default than the earlier success-weighted score, and that unconditional mutation remained mixed: it helped some late-stage core metrics but hurt H1 transfer.

v11 tests one narrow change:

- keep the v10 `querygrad` utility score fixed
- keep the same hard selective-growth benchmark and longplus schedule
- change mutation from unconditional to conditional

Conditional mutation policy:

- `split_mode: mutate`
- mutate only on late transitions with `next_stage_index >= 5`
- mutate only the top half of selected parents by selected-parent utility score

This means only high-utility late splits mutate, while earlier and lower-ranked splits remain pure clones.

## Benchmark And Schedule

Base architecture:

- v3 key-centric first-hop router
- v4 implicit learned retrieval
- v8/v10 selective growth family

Core regime:

- final compute leaves: `32`
- output node: `0`
- train writers/episode: `2`
- eval writers/episode: `{2, 6, 10}`
- `start_node_pool_size = 2`
- `query_ttl = 2..3`
- `max_rollout_steps = 12`
- task-only coverage excludes bootstrap packets

Selective active schedule:

- `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`

Longplus training schedule:

- total steps: `8000`
- stage steps: `[250, 250, 300, 300, 400, 600, 5900]`
- final-stage share: `73.75%`
- stage bootstrap: `75` steps per stage

Transfer regime H1:

- same schedule and topology
- train writers/episode: `4`
- eval writers/episode: `{4, 8, 12, 14}`

Visible GPU count actually used by PyTorch: `2`.

## Exact Configs

Core:

- `configs/v10_utility_querygrad_longplus.yaml`
- `configs/v10_utility_querygrad_mutate_longplus.yaml`
- `configs/v11_utility_querygrad_condmut_longplus.yaml`

Transfer:

- `configs/v10_transfer_h1_utility_querygrad_longplus.yaml`
- `configs/v10_transfer_h1_utility_querygrad_mutate_longplus.yaml`
- `configs/v11_transfer_h1_utility_querygrad_condmut_longplus.yaml`

## Core Runs

v11 conditional-mutation runs:

- `runs/20260318-083657-v11-utility-querygrad-condmut-longplus-s1234`
- `runs/20260318-084332-v11-utility-querygrad-condmut-longplus-s2234`
- `runs/20260318-085009-v11-utility-querygrad-condmut-longplus-s3234`
- `runs/20260318-085646-v11-utility-querygrad-condmut-longplus-s4234`

## Transfer Runs

v11 H1 transfer runs:

- `runs/20260318-090851-v11-transfer-h1-utility-querygrad-condmut-longplus-s1234`
- `runs/20260318-091548-v11-transfer-h1-utility-querygrad-condmut-longplus-s2234`

## Core Summary

Mean ± std across seeds:

| Arm | Seeds | Best val | Last val | K2 | K6 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 `querygrad` | 4 | `0.6927 ± 0.0578` | `0.5677 ± 0.0400` | `0.6133 ± 0.0531` | `0.5195 ± 0.0545` | `0.4863 ± 0.0347` |
| v10 `querygrad+mutate` | 4 | `0.6875 ± 0.0737` | `0.5990 ± 0.0518` | `0.6465 ± 0.0453` | `0.5195 ± 0.0305` | `0.5176 ± 0.0224` |
| v11 conditional mutation | 4 | `0.6667 ± 0.0442` | `0.5625 ± 0.1062` | `0.6191 ± 0.0903` | `0.5332 ± 0.0501` | `0.4941 ± 0.0178` |

Per-seed v11 conditional-mutation evals:

| Seed | Best val | Last val | K2 | K6 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| `1234` | `0.6458` | `0.6458` | `0.6719` | `0.5938` | `0.4922` |
| `2234` | `0.6042` | `0.4792` | `0.5000` | `0.4609` | `0.4766` |
| `3234` | `0.7083` | `0.6875` | `0.7344` | `0.5625` | `0.5234` |
| `4234` | `0.7083` | `0.4375` | `0.5703` | `0.5156` | `0.4844` |

## Transfer Summary

Mean ± std across seeds:

| Arm | Seeds | Best val | Last val | K4 | K8 | K12 | K14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 `querygrad` H1 | 2 | `0.6562 ± 0.0729` | `0.5833 ± 0.1458` | `0.5352 ± 0.1289` | `0.4453 ± 0.1016` | `0.4258 ± 0.0352` | `0.4141 ± 0.0938` |
| v10 `querygrad+mutate` H1 | 2 | `0.5521 ± 0.0104` | `0.4375 ± 0.0625` | `0.4258 ± 0.0273` | `0.3945 ± 0.0586` | `0.3867 ± 0.0039` | `0.4023 ± 0.0195` |
| v11 conditional mutation H1 | 2 | `0.5833 ± 0.1042` | `0.5833 ± 0.1042` | `0.5430 ± 0.1055` | `0.4258 ± 0.0664` | `0.4180 ± 0.0039` | `0.4062 ± 0.0625` |

Per-seed v11 H1 evals:

| Seed | Best val | Last val | K4 | K8 | K12 | K14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `1234` | `0.4792` | `0.4792` | `0.4375` | `0.3594` | `0.4219` | `0.3438` |
| `2234` | `0.6875` | `0.6875` | `0.6484` | `0.4922` | `0.4141` | `0.4688` |

## Interpretation

Conditional mutation is a useful refinement of the mutate idea, but it is not yet a clear new default.

What v11 did achieve:

- it preserved the v10 `querygrad` core behavior
- it improved mean `K6` over both pushed v10 baselines
- it recovered most of the H1 transfer damage seen in unconditional mutation

What it did not achieve:

- it did not beat unconditional mutation on mean core `K10`
- it did not beat `querygrad` on mean best-val or mean last-val
- it still has high seed sensitivity on late-stage stability

So the late mutation gate helped robustness relative to unconditional mutation, but not enough to claim a real upgrade over utility-only growth.

## Mechanistic Read

This result is consistent with the v10 conclusion:

- early task-only coverage is already saturated in these selective-growth arms
- the remaining differences come from late-stage allocation and late-stage perturbation
- unconditional mutation is too blunt
- late, high-utility-only mutation is safer, but still noisy

The gate improves the mutate side of the tradeoff without clearly surpassing clone-only utility selection.

## Recommended Next Move

The next best move is **adaptive mutation**, not more unconditional mutation.

Specifically:

- keep the `querygrad` utility score as the default selector
- mutate only when the parent utility margin is large enough
- or mutate only when sibling usefulness is likely to diversify usefully
- otherwise clone

That would test whether mutation needs a confidence trigger rather than a fixed top-half rule.
