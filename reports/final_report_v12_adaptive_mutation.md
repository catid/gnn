# APSGNN v12 Adaptive Mutation

## Goal

Validate a narrower mutation policy after v11.

The v12 rule keeps the v10 `querygrad` utility selector fixed and only mutates late-stage selected parents when their utility clears a score margin:

- mutate only on late transitions: `16 -> 24` and `24 -> 32`
- mutate only selected parents with utility score at least `reference + 0.25`
- `reference = max(unselected eligible score)` when unselected eligible parents exist
- `reference = mean(selected score)` when all remaining eligible parents must split

This tests whether a confidence-gated mutation policy can keep the core late-stage benefits of mutation while avoiding the transfer regressions seen with unconditional mutation.

## Configs

- Core: `configs/v12_utility_querygrad_adaptmut_longplus.yaml`
- Transfer H1: `configs/v12_transfer_h1_utility_querygrad_adaptmut_longplus.yaml`
- Visible GPUs actually used: `2`

Core v12 run directories:

- `runs/20260318-093538-v12-utility-querygrad-adaptmut-longplus-s1234`
- `runs/20260318-094213-v12-utility-querygrad-adaptmut-longplus-s2234`
- `runs/20260318-094848-v12-utility-querygrad-adaptmut-longplus-s3234`
- `runs/20260318-095526-v12-utility-querygrad-adaptmut-longplus-s4234`

Transfer v12 run directories:

- `runs/20260318-100439-v12-transfer-h1-utility-querygrad-adaptmut-longplus-s1234`
- `runs/20260318-101136-v12-transfer-h1-utility-querygrad-adaptmut-longplus-s2234`

## Core Results

| Arm | Seeds | Best val | Last val | K2 | K6 | K10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 `querygrad` | 4 | `0.6927 ± 0.0578` | `0.5677 ± 0.0400` | `0.6133 ± 0.0531` | `0.5195 ± 0.0545` | `0.4863 ± 0.0347` |
| v10 `querygrad+mutate` | 4 | `0.6875 ± 0.0737` | `0.5990 ± 0.0518` | `0.6465 ± 0.0453` | `0.5195 ± 0.0305` | `0.5176 ± 0.0224` |
| v11 conditional mutation | 4 | `0.6667 ± 0.0442` | `0.5625 ± 0.1062` | `0.6191 ± 0.0903` | `0.5332 ± 0.0501` | `0.4941 ± 0.0178` |
| v12 adaptive mutation | 4 | `0.6667 ± 0.0329` | `0.5885 ± 0.0648` | `0.6289 ± 0.0419` | `0.5195 ± 0.0334` | `0.5098 ± 0.0384` |

## H1 Transfer Results

H1 keeps the selective-growth mechanism fixed and raises retrieval pressure:

- train writers per episode: `4`
- eval writers per episode: `4, 8, 12, 14`

| Arm | Seeds | Best val | K4 | K8 | K12 | K14 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v10 `querygrad` H1 | 2 | `0.6562 ± 0.0729` | `0.5352 ± 0.1289` | `0.4453 ± 0.1016` | `0.4258 ± 0.0352` | `0.4141 ± 0.0938` |
| v10 `querygrad+mutate` H1 | 2 | `0.5521 ± 0.0104` | `0.4258 ± 0.0273` | `0.3945 ± 0.0586` | `0.3867 ± 0.0039` | `0.4023 ± 0.0195` |
| v11 conditional mutation H1 | 2 | `0.5833 ± 0.1042` | `0.5430 ± 0.1055` | `0.4258 ± 0.0664` | `0.4180 ± 0.0039` | `0.4062 ± 0.0625` |
| v12 adaptive mutation H1 | 2 | `0.5833 ± 0.0625` | `0.5039 ± 0.0742` | `0.4336 ± 0.0977` | `0.3945 ± 0.0039` | `0.3789 ± 0.0430` |

## Interpretation

Adaptive mutation is a mild refinement, not a new default.

On the core long schedule, v12 matched v11 on mean best validation, improved mean last-checkpoint stability over v11, and raised mean `K10` versus both v10 `querygrad` and v11 conditional mutation. It did **not** improve mean `K6`; it landed exactly on the v10 `querygrad` / v10 `querygrad+mutate` mean and below v11.

Under H1 transfer, v12 adaptive mutation did not preserve the strongest baseline. It remained clearly better than unconditional v10 mutation on best-val and on `K4/K8/K12`, but it fell below the plain v10 `querygrad` transfer baseline and below v11 conditional mutation on most metrics. So the v12 gate reduces some mutation damage, but it still does not turn mutation into a robust default upgrade.

## Conclusion

The most defensible current default remains **utility-only `querygrad` selective growth**.

Adaptive mutation is useful as a safety-oriented variant:

- better core stability than v11
- better transfer behavior than unconditional mutation
- but not enough to replace the mutation-free baseline

## Next Move

If mutation is revisited, it should be **more selective than v12**:

- trigger only when the selected parent has a large utility margin over unselected peers
- or gate on disagreement / uncertainty between visit and gradient components
- or make mutation conditional on late-stage stagnation rather than always-on for a subset of selected parents

The next clean experiment is therefore a higher-confidence mutation gate, not broader mutation.
