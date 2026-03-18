# APSGNN v14 Component-Agreement Mutation

## Goal

Validate whether mutation should require agreement between the two active `querygrad` utility components rather than a score margin alone.

The v14 rule keeps the v10 `querygrad` selector fixed and mutates only on the final `24 -> 32` transition, only when the selected parent clears three gates:

- mutate only at stage index `6` (`24 -> 32`)
- require `selected_score >= reference + 0.75`
- require both `visit_z >= 0.25` and `query_grad_z >= 0.25`

This tests whether component agreement improves over the v13 high-confidence margin gate without reopening broader mutation.

## Configs

- Core: `configs/v14_utility_querygrad_agree_longplus.yaml`
- Transfer H1: `configs/v14_transfer_h1_utility_querygrad_agree_longplus.yaml`
- Visible GPUs actually used: `2`

Core v14 run directories:

- `runs/20260318-113852-v14-utility-querygrad-agree-longplus-s1234`
- `runs/20260318-114530-v14-utility-querygrad-agree-longplus-s2234`
- `runs/20260318-115206-v14-utility-querygrad-agree-longplus-s3234`
- `runs/20260318-115843-v14-utility-querygrad-agree-longplus-s4234`

Transfer v14 run directories:

- `runs/20260318-121124-v14-transfer-h1-utility-querygrad-agree-longplus-s1234`
- `runs/20260318-121818-v14-transfer-h1-utility-querygrad-agree-longplus-s2234`

## Core Results

| Arm | Seeds | Best val | Last val | K2 | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| v10 `querygrad` | 4 | `0.6927 ± 0.0667` | `0.5677 ± 0.0462` | `0.6133 ± 0.0614` | `0.5195 ± 0.0630` | `0.4863 ± 0.0400` |
| v10 `querygrad+mutate` | 4 | `0.6875 ± 0.0851` | `0.5990 ± 0.0598` | `0.6465 ± 0.0524` | `0.5195 ± 0.0352` | `0.5176 ± 0.0258` |
| v11 conditional mutation | 4 | `0.6667 ± 0.0442` | `0.5625 ± 0.1062` | `0.6191 ± 0.0903` | `0.5332 ± 0.0501` | `0.4941 ± 0.0178` |
| v12 adaptive mutation | 4 | `0.6667 ± 0.0329` | `0.5885 ± 0.0648` | `0.6289 ± 0.0419` | `0.5195 ± 0.0334` | `0.5098 ± 0.0384` |
| v13 high-confidence mutation | 4 | `0.6823 ± 0.0462` | `0.5833 ± 0.1227` | `0.6250 ± 0.0965` | `0.5234 ± 0.0506` | `0.5137 ± 0.0117` |
| v14 component-agreement mutation | 4 | `0.6823 ± 0.0574` | `0.6094 ± 0.0521` | `0.6484 ± 0.0456` | `0.5176 ± 0.0483` | `0.5059 ± 0.0173` |

## H1 Transfer Results

H1 keeps the selective-growth mechanism fixed and raises retrieval pressure:

- train writers per episode: `4`
- eval writers per episode: `4, 8, 12, 14`

| Arm | Seeds | Best val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- |
| v10 `querygrad` | 2 | `0.6562 ± 0.1031` | `0.5352 ± 0.1823` | `0.4453 ± 0.1436` | `0.4258 ± 0.0497` | `0.4141 ± 0.1326` |
| v10 `querygrad+mutate` | 2 | `0.5521 ± 0.0147` | `0.4258 ± 0.0387` | `0.3945 ± 0.0829` | `0.3867 ± 0.0055` | `0.4023 ± 0.0276` |
| v11 conditional mutation | 2 | `0.5833 ± 0.1042` | `0.5430 ± 0.1055` | `0.4258 ± 0.0664` | `0.4180 ± 0.0039` | `0.4062 ± 0.0625` |
| v12 adaptive mutation | 2 | `0.5833 ± 0.0625` | `0.5039 ± 0.0742` | `0.4336 ± 0.0977` | `0.3945 ± 0.0039` | `0.3789 ± 0.0430` |
| v13 high-confidence mutation | 2 | `0.6042 ± 0.1768` | `0.5312 ± 0.1547` | `0.4648 ± 0.0829` | `0.4258 ± 0.0166` | `0.3711 ± 0.0608` |
| v14 component-agreement mutation | 2 | `0.6146 ± 0.1620` | `0.4961 ± 0.2044` | `0.4023 ± 0.1713` | `0.3633 ± 0.0718` | `0.3281 ± 0.1215` |

## Interpretation

v14 is a negative but useful result.

On the core long schedule, v14 component-agreement mutation is credible: it matches v13 on mean best-val, improves mean last-val over v13, and keeps strong `K2/K10`. But it does not improve `K6` over plain `querygrad`, so the core case was already marginal.

The H1 transfer pair settles the question. v14 falls below both plain `querygrad` and v13 high-confidence mutation on mean `K8`, `K12`, and `K14`, with only one strong seed carrying the pair. That means the extra component-agreement gate is still not enough to turn mutation into a robust default. It narrows the damage relative to broader mutation, but it does not beat staying mutation-free.

## Conclusion

The most defensible default remains **utility-only `querygrad` selective growth**.

v14 component-agreement mutation should be treated as an informative failure mode:

- stronger late-core stability than some earlier mutation gates
- but still worse transfer means than both v10 `querygrad` and v13 high-confidence mutation
- so mutation should remain opt-in and highly constrained, not the default growth policy

Plot: `reports/v14_component_agreement_comparison.png`
