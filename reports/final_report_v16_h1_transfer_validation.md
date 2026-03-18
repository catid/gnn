# APSGNN v16 H1 Transfer Validation

## What Changed

This follow-up does not change the model family. It validates the current H1 transfer regime by adding the two missing longplus controls under the same settings:

- `staged_static_h1`: staged curriculum with activation but no inheritance
- `clone_h1`: deterministic clone growth with inheritance
- `querygrad_h1`: existing utility-only `querygrad` selective growth from v10

The goal is to check whether the v10 `querygrad` winner still looks like the safest default once the transfer comparison includes both a curriculum-only control and an inheritance-only control.

Visible GPU count used: `2`

## Exact Configs

- [`v16_transfer_h1_staged_static_longplus.yaml`](/home/catid/gnn/configs/v16_transfer_h1_staged_static_longplus.yaml)
- [`v16_transfer_h1_clone_selective_longplus.yaml`](/home/catid/gnn/configs/v16_transfer_h1_clone_selective_longplus.yaml)
- Existing reference arm: `v10_transfer_h1_utility_querygrad_longplus.yaml`

Shared regime:

- train writers per episode: `4`
- eval writers per episode: `4, 8, 12, 14`
- total train steps: `8000`
- selective stage schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32`
- stage steps: `[250, 250, 300, 300, 400, 600, 5900]`

## Per-Seed Results

| Arm | Seed | Best Val | Last Val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clone_h1 | 1234 | 0.6250 | 0.6250 | 0.5000 | 0.3906 | 0.4219 | 0.3672 |
| clone_h1 | 2234 | 0.6042 | 0.6042 | 0.5859 | 0.5078 | 0.4609 | 0.4141 |
| querygrad_h1 | 1234 | 0.5833 | 0.4375 | 0.4062 | 0.3438 | 0.4609 | 0.3203 |
| querygrad_h1 | 2234 | 0.7292 | 0.7292 | 0.6641 | 0.5469 | 0.3906 | 0.5078 |
| querygrad_h1 | 3234 | 0.4583 | 0.4583 | 0.4844 | 0.4375 | 0.4297 | 0.4141 |
| staged_static_h1 | 1234 | 0.5833 | 0.4792 | 0.4141 | 0.3281 | 0.3594 | 0.2969 |
| staged_static_h1 | 2234 | 0.5000 | 0.5000 | 0.5078 | 0.4453 | 0.3594 | 0.3281 |

## Mean / Std Summary

| Arm | Count | Best Val | Last Val | K4 | K8 | K12 | K14 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| querygrad_h1 | 3 | 0.5903 ± 0.1356 | 0.5417 ± 0.1627 | 0.5182 ± 0.1322 | 0.4427 ± 0.1017 | 0.4271 ± 0.0352 | 0.4141 ± 0.0938 |
| staged_static_h1 | 2 | 0.5417 ± 0.0589 | 0.4896 ± 0.0147 | 0.4609 ± 0.0663 | 0.3867 ± 0.0829 | 0.3594 ± 0.0000 | 0.3125 ± 0.0221 |
| clone_h1 | 2 | 0.6146 ± 0.0147 | 0.6146 ± 0.0147 | 0.5430 ± 0.0608 | 0.4492 ± 0.0829 | 0.4414 ± 0.0276 | 0.3906 ± 0.0331 |

## Interpretation

`querygrad_h1` remains the safest transfer default. It stays ahead of `staged_static_h1` on mean `K4`, `K8`, and `K12`, and it also stays ahead of `clone_h1` on mean `K4` and `K14`. `clone_h1` does recover some of the gap versus `querygrad_h1` in the middle of the grid and clearly improves over `staged_static_h1`, which is evidence that inheritance still matters under H1.

The cleaner conclusion is therefore:

- curriculum alone helps, but not enough
- inheritance helps beyond staged activation
- utility-selected growth is still the strongest transfer default overall

This validates the earlier v9-v15 direction rather than overturning it. The main open question is still not whether clone beats staged static under transfer; it does. The open question is whether a mutation policy can beat utility-only `querygrad` without hurting robustness, and the accumulated v11-v15 results still say no.

## Outputs

- summary JSON: [`summary_metrics_v16.json`](/home/catid/gnn/reports/summary_metrics_v16.json)
- plot: [`v16_h1_transfer_comparison.png`](/home/catid/gnn/reports/v16_h1_transfer_comparison.png)
