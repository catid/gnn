# APSGNN v19 Selector Family Scale

## Status

v19 was stopped at the scale-feasibility gate rather than after a full selector-family matrix.
The 64-leaf benchmark was attempted first, then the allowed 48-leaf fallback was profiled after a narrow growth-selection optimization. On the 2 visible GPUs, both remained too slow for the mandated 40-run initial matrix plus 10+ run follow-up campaign.

## What Changed From v18

- Added a v19 selector-family config/script scaffold for `visitonly`, `visit+task_grad`, `visit+query_grad`, `full querygrad`, and `querygrad-only`.
- Added 64-leaf target/schedule support and the allowed 48-leaf fallback configs.
- Added a grouped size-allocation path in `apsgnn/growth.py` so large selective transitions like `32 -> 48` and `48 -> 64` no longer rely on exhaustive parent-subset enumeration.
- Added v19 tests covering selector score computation, bootstrap exclusion, larger target mapping, and schedule matching.

## Larger Benchmark Attempt

- Attempted final compute leaves: `64`
- Fallback final compute leaves: `48`
- Visible GPU count used: `2`
- 64-leaf schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32 -> 48 -> 64`
- 48-leaf fallback schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32 -> 48`

## Completed Profiling Runs

| Leaves | Steps run | Runtime | Stage @50 | Active nodes @50 | PPS @50 | Max GB @50 | Task visit cov @50 | Config | Run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `64` | `50` | `~61s` | `8` | `64` | `1129.5` | `0.618` | `0.828` | `configs/v19_core_visitonly_scale.yaml` | `runs/20260318-213521-v19-core-visitonly-scale-profile64-s1234` |
| `48` | `50` | `~49s` | `7` | `48` | `1143.4` | `0.484` | `0.917` | `configs/v19_core_visitonly_scale48.yaml` | `runs/20260318-213627-v19-core-visitonly-scale48-profile-s1234` |

## Why The Full Matrix Was Not Completed

- The 64-leaf 50-step profile already reached stage index `8` with `64` active compute nodes and took about `61s` for only `50` steps.
- The allowed 48-leaf fallback still took about `49s` for `50` steps.
- Those profiles imply lower-bound runtimes of roughly `3.05h` per Core-L run and `3.39h` per T1-L run at 64 leaves, or `2.45h` and `2.72h` respectively even at the 48-leaf fallback.
- Lower-bound wall-clock for the required initial matrix alone:
  - 64-leaf attempt: `~128.8h`
  - 48-leaf fallback: `~103.4h`
- Lower-bound wall-clock for the full requested campaign:
  - 64-leaf attempt: `~161.0h`
  - 48-leaf fallback: `~129.2h`
- These are optimistic lower bounds and do not include slower later-stage behavior, extra eval sweeps, report generation time, or the required confirmatory reruns.

## Partial Runs Not Counted

- `runs/20260318-212958-v19-core-visitonly-scale-s1234`
- `runs/20260318-213139-v19-core-visit-taskgrad-scale-s1234`
- `runs/20260318-213146-v19-core-visit-querygrad-scale-s1234`

## Tests And Verification

- `python -m compileall apsgnn tests scripts/run_v19_eval_sweep.py`
- `pytest -q tests/test_v19_selector_family.py`
- `pytest -q`
- Final test status before stopping: `83 passed`

## Conclusion

v19 did not produce a valid selector-family comparison because the requested scale-up campaign was not feasible on the available hardware at the mandated budgets. The correct stopping point was to preserve the 64-leaf attempt, the 48-leaf fallback, the growth-selection optimization that made the attempt runnable at all, and the profiling evidence showing why the full matrix should not be faked.

## Recommended Next Step

Keep the v18 `visitonly` selector as the working default and revisit selector-family scale only with either more compute or materially shorter schedules. If selector-family exploration resumes under current hardware, the only defensible version is a reduced-budget campaign rather than the original v19 matrix.
