# APSGNN v13 High-Confidence Mutation

## Goal

Validate a stricter mutation gate after v12.

The v13 rule keeps the v10 `querygrad` selector fixed and mutates only on the final `24 -> 32` transition, only when the selected parent clears a large utility margin:

- mutate only at stage index `6` (`24 -> 32`)
- require `selected_score >= reference + 0.75`
- `reference = max(unselected eligible score)` when available
- otherwise fall back to the mean selected score

This tests whether a very conservative, final-stage-only mutation gate can preserve late-stage core gains without repeating the H1 transfer regressions seen with broader mutation.

## Configs

- Core: `configs/v13_utility_querygrad_hiconf_longplus.yaml`
- Transfer H1: `configs/v13_transfer_h1_utility_querygrad_hiconf_longplus.yaml`
- Visible GPUs actually used: `2`

Core v13 run directories:

- `runs/20260318-102658-v13-utility-querygrad-hiconf-longplus-s1234`
- `runs/20260318-103333-v13-utility-querygrad-hiconf-longplus-s2234`
- `runs/20260318-104010-v13-utility-querygrad-hiconf-longplus-s3234`
- `runs/20260318-104647-v13-utility-querygrad-hiconf-longplus-s4234`

Transfer v13 run directories:

- `runs/20260318-105603-v13-transfer-h1-utility-querygrad-hiconf-longplus-s1234`
- `runs/20260318-110259-v13-transfer-h1-utility-querygrad-hiconf-longplus-s2234`
- `runs/20260318-112106-v13-transfer-h1-utility-querygrad-hiconf-longplus-s3234`

## Core Results

| Arm | Seeds | Best val | Last val | K2 | K6 | K10 |
| --- | --- | --- | --- | --- | --- | --- |
| v10 `querygrad` | 4 | `0.6927 ± 0.0667` | `0.5677 ± 0.0462` | `0.6133 ± 0.0614` | `0.5195 ± 0.0630` | `0.4863 ± 0.0400` |
| v10 `querygrad+mutate` | 4 | `0.6875 ± 0.0851` | `0.5990 ± 0.0598` | `0.6465 ± 0.0524` | `0.5195 ± 0.0352` | `0.5176 ± 0.0258` |
| v11 conditional mutation | 4 | `0.6667 ± 0.0442` | `0.5625 ± 0.1062` | `0.6191 ± 0.0903` | `0.5332 ± 0.0501` | `0.4941 ± 0.0178` |
| v12 adaptive mutation | 4 | `0.6667 ± 0.0329` | `0.5885 ± 0.0648` | `0.6289 ± 0.0419` | `0.5195 ± 0.0334` | `0.5098 ± 0.0384` |
| v13 high-confidence mutation | 4 | `0.6823 ± 0.0462` | `0.5833 ± 0.1227` | `0.6250 ± 0.0965` | `0.5234 ± 0.0506` | `0.5137 ± 0.0117` |

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

## Matched 3-Seed Transfer Confirmation

To reduce the variance in the key comparison, v10 `querygrad` and v13 high-confidence mutation were extended to a matched 3-seed H1 pair.

| Arm | Seeds | Best val | Last val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| v10 `querygrad` | 3 | `0.5903 ± 0.1356` | `0.5417 ± 0.1627` | `0.5182 ± 0.1322` | `0.4427 ± 0.1017` | `0.4271 ± 0.0352` | `0.4141 ± 0.0938` |
| v13 high-confidence mutation | 3 | `0.5625 ± 0.1443` | `0.5278 ± 0.1772` | `0.5312 ± 0.1094` | `0.4453 ± 0.0677` | `0.4323 ± 0.0163` | `0.3672 ± 0.0435` |

## Interpretation

High-confidence mutation is a credible core variant, not a new default.

On the core long schedule, v13 lifted mean `K10` over v12 and v11, stayed close to v10 `querygrad+mutate`, and improved mean best-val versus v11/v12. It did not clearly beat plain v10 `querygrad` on mean best-val or mean last-val.

The matched 3-seed H1 comparison is the deciding result. Extending both v10 `querygrad` and v13 to three seeds did not flip the transfer conclusion: v13 is effectively tied on `K8/K12`, slightly ahead on `K4`, and clearly worse on `K14` and mean best-val. That means the stricter gate reduces some mutation damage, but still does not turn mutation into a robust transfer-default upgrade.

## Conclusion

The most defensible default remains **utility-only `querygrad` selective growth**.

v13 high-confidence mutation is useful as a narrow late-stage variant:

- stronger core `K10` than v11/v12
- better transfer behavior than some broader mutation policies
- but still not enough to replace mutation-free `querygrad` under the matched 3-seed H1 comparison

## Next Move

If mutation is revisited, it should be **more selective than v13**:

- gate on a stronger confidence signal than a fixed margin
- or trigger only under late-stage stagnation
- otherwise keep mutation off and refine the utility score instead

Plot: `reports/v13_high_confidence_comparison.png`
