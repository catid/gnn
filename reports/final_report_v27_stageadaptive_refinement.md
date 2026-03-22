# APSGNN v27 Stage-Adaptive Refinement

This round refines the stage-aware task-grad idea after v26. It screens new stage-switch variants against each other, then compares the surviving stage-adaptive winner back to the fixed v26 baselines `V` and `VT-0.5`.

## Core-S screening

| Selector | Composite | Dense | Last |
| --- | --- | --- | --- |
| StageEarly-0.5 | 0.0574 ± 0.0062 | 0.0312 ± 0.0110 | 0.0833 ± 0.0000 |
| StageLate-0.5 | 0.0574 ± 0.0062 | 0.0312 ± 0.0110 | 0.0833 ± 0.0000 |
| StageLate-0.375 | 0.0574 ± 0.0062 | 0.0312 ± 0.0110 | 0.0833 ± 0.0000 |
| StageLate-0.625 | 0.0574 ± 0.0062 | 0.0312 ± 0.0110 | 0.0833 ± 0.0000 |
| StageFinal-0.5 | 0.0458 ± 0.0103 | 0.0234 ± 0.0000 | 0.0625 ± 0.0295 |

Promoted to `T1-S`: `StageEarly-0.5, StageLate-0.5, StageLate-0.375`

## T1-S screening

| Selector | Composite | Dense | Last |
| --- | --- | --- | --- |
| StageEarly-0.5 | 0.0240 ± 0.0056 | 0.0260 ± 0.0000 | 0.0208 ± 0.0295 |
| StageLate-0.5 | 0.0240 ± 0.0056 | 0.0260 ± 0.0000 | 0.0208 ± 0.0295 |
| StageLate-0.375 | 0.0240 ± 0.0056 | 0.0260 ± 0.0000 | 0.0208 ± 0.0295 |

Promoted to `T1-M`: `StageEarly-0.5, StageLate-0.5`

## T1-M confirmation

| Selector | Best | Last | Dense |
| --- | --- | --- | --- |
| StageEarly-0.5 | 0.0417 ± 0.0000 | 0.0139 ± 0.0241 | 0.0348 ± 0.0131 |
| StageLate-0.5 | 0.0417 ± 0.0000 | 0.0139 ± 0.0241 | 0.0382 ± 0.0150 |

Current stage-adaptive winner: `StageLate-0.5`


## Equivalence Check

`StageLate-0.5` is config-identical to the earlier v26 selector `stageadaptive_vt`: same stage schedule, same base visit-only utility, same `adaptive_selector_stage_index_min`, and the same adaptive task-grad weight. That means v27 did not discover a new late-stage rule here; it revalidated the same late-switch policy under a cleaner focused screen.


## Baseline Comparison

### T1-M

| Selector | Best | Last | Dense |
| --- | --- | --- | --- |
| StageLate-0.5 | 0.0417 ± 0.0000 | 0.0139 ± 0.0241 | 0.0382 ± 0.0150 |
| V | 0.1111 ± 0.0636 | 0.0139 ± 0.0241 | 0.0174 ± 0.0060 |
| VT-0.5 | 0.0833 ± 0.0417 | 0.0139 ± 0.0241 | 0.0209 ± 0.0105 |

### T1-L

| Selector | Best | Last | Dense |
| --- | --- | --- | --- |
| StageLate-0.5 | 0.0833 ± 0.0589 | 0.0208 ± 0.0295 | 0.0399 ± 0.0172 |
| V | 0.1111 ± 0.0241 | 0.0278 ± 0.0241 | 0.0382 ± 0.0159 |
| VT-0.5 | 0.1083 ± 0.0228 | 0.0333 ± 0.0186 | 0.0382 ± 0.0113 |

### Core-M

| Selector | Best | Last | Dense |
| --- | --- | --- | --- |
| StageLate-0.5 | 0.1458 ± 0.0417 | 0.0521 ± 0.0399 | 0.0371 ± 0.0224 |
| V | 0.1458 ± 0.0417 | 0.0521 ± 0.0399 | 0.0371 ± 0.0224 |
| VT-0.5 | 0.1250 ± 0.0481 | 0.0521 ± 0.0399 | 0.0332 ± 0.0148 |

## Completed v27 runs

| Phase | Sel | Seed | Best | Last | Last5 | Dense |
| --- | --- | --- | --- | --- | --- | --- |
| Core-M | StageLate-0.5 | 1234 | 0.1667 | 0.0417 | 0.0583 | 0.0312 |
| Core-M | StageLate-0.5 | 2234 | 0.1667 | 0.0833 | 0.1083 | 0.0234 |
| Core-M | StageLate-0.5 | 3234 | 0.0833 | 0.0000 | 0.0000 | 0.0234 |
| Core-M | StageLate-0.5 | 4234 | 0.1667 | 0.0833 | 0.0250 | 0.0703 |
| Core-S | StageEarly-0.5 | 1234 | 0.1667 | 0.0833 | 0.0667 | 0.0234 |
| Core-S | StageEarly-0.5 | 2234 | 0.1250 | 0.0833 | 0.0750 | 0.0391 |
| Core-S | StageFinal-0.5 | 1234 | 0.1667 | 0.0833 | 0.0667 | 0.0234 |
| Core-S | StageFinal-0.5 | 2234 | 0.1250 | 0.0417 | 0.0667 | 0.0234 |
| Core-S | StageLate-0.375 | 1234 | 0.1667 | 0.0833 | 0.0667 | 0.0234 |
| Core-S | StageLate-0.375 | 2234 | 0.1250 | 0.0833 | 0.0750 | 0.0391 |
| Core-S | StageLate-0.625 | 1234 | 0.1667 | 0.0833 | 0.0667 | 0.0234 |
| Core-S | StageLate-0.625 | 2234 | 0.1250 | 0.0833 | 0.0750 | 0.0391 |
| Core-S | StageLate-0.5 | 1234 | 0.1667 | 0.0833 | 0.0667 | 0.0234 |
| Core-S | StageLate-0.5 | 2234 | 0.1250 | 0.0833 | 0.0750 | 0.0391 |
| T1-M | StageEarly-0.5 | 3234 | 0.0417 | 0.0417 | 0.0083 | 0.0208 |
| T1-M | StageEarly-0.5 | 4234 | 0.0417 | 0.0000 | 0.0000 | 0.0366 |
| T1-M | StageEarly-0.5 | 5234 | 0.0417 | 0.0000 | 0.0167 | 0.0469 |
| T1-M | StageLate-0.5 | 3234 | 0.0417 | 0.0417 | 0.0083 | 0.0208 |
| T1-M | StageLate-0.5 | 4234 | 0.0417 | 0.0000 | 0.0083 | 0.0469 |
| T1-M | StageLate-0.5 | 5234 | 0.0417 | 0.0000 | 0.0167 | 0.0469 |
| T1-S | StageEarly-0.5 | 1234 | 0.0833 | 0.0000 | 0.0417 | 0.0260 |
| T1-S | StageEarly-0.5 | 2234 | 0.0833 | 0.0417 | 0.0083 | 0.0260 |
| T1-S | StageLate-0.375 | 1234 | 0.0833 | 0.0000 | 0.0417 | 0.0260 |
| T1-S | StageLate-0.375 | 2234 | 0.0833 | 0.0417 | 0.0083 | 0.0260 |
| T1-S | StageLate-0.5 | 1234 | 0.0833 | 0.0000 | 0.0417 | 0.0260 |
| T1-S | StageLate-0.5 | 2234 | 0.0833 | 0.0417 | 0.0083 | 0.0260 |
