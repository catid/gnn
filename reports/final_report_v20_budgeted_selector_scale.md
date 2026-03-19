# APSGNN v20 Budgeted Selector Scale

## Status

v20 stopped at the budget gate. The campaign was designed specifically to avoid another fantasy-scale matrix, and the measured 64/48/40/32 profiles showed that the full 50-run plan was still too expensive on the 2-GPU machine under the requested harder regimes and S/M/L schedules.

## What Changed From v19

- Added a v20 config family for 32- and 40-leaf selector-family screening/confirmation/transfer schedules.
- Added 64/48/40/32 gate configs for a clean scale/budget decision.
- Added focused v20 tests for selector scoring, bootstrap exclusion, schedule matching, larger-scale target mapping, and reproducibility.

## Budget Gate

- Candidate scales profiled: `64`, `48`, `40`, `32`
- Visible GPU count used: `2`
- Largest technically runnable scale: `32`
- Feasible campaign scale selected: `none`

| Scale | Runtime @100 | Stage @100 | Active @100 | Task visit cov @100 | Max GB @100 | Config | Run |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `64` | `~115s` | `8` | `64` | `0.969` | `0.619` | `configs/v20_gate_core_visitonly_64.yaml` | `runs/20260319-043232-v20-gate-core-visitonly-64-s1234` |
| `48` | `~90s` | `7` | `48` | `0.979` | `0.485` | `configs/v20_gate_core_visitonly_48.yaml` | `runs/20260319-043433-v20-gate-core-visitonly-48-s1234` |
| `40` | `~77s` | `7` | `40` | `1.000` | `0.417` | `configs/v20_core_visitonly_40_s.yaml` | `runs/20260319-043608-v20-gate-core-visitonly-40-s1234` |
| `32` | `~61s` | `6` | `32` | `0.938` | `0.349` | `configs/v20_core_visitonly_32_s.yaml` | `runs/20260319-043731-v20-gate-core-visitonly-32-s1234` |

## Projected Full-Campaign Cost

| Scale | S min/run | M min/run | L min/run | 50-run campaign, 2-GPU DDP | 50-run campaign, optimistic 2x single-GPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| `64` | `34.5` | `51.8` | `69.0` | `40.8h` | `20.4h` |
| `48` | `27.0` | `40.5` | `54.0` | `31.9h` | `16.0h` |
| `40` | `23.1` | `34.7` | `46.2` | `27.3h` | `13.7h` |
| `32` | `18.3` | `27.5` | `36.6` | `21.7h` | `10.9h` |

## Interpretation

64 and 48 were immediately ruled out by runtime. 40 was still too expensive to justify a 50-run campaign. 32 was the only scale that remained technically runnable, but the harder v20 task settings made it much slower than the older v18 home regime, and the projected campaign still landed in the ~22h range even before extra eval sweeps, aggregation, and reruns.

## Conclusion

v20 did not produce a selector-family winner. The useful result is the budget gate itself: under the requested harder regimes and verification requirements, no scale from `64/48/40/32` yields a realistic full v20 campaign on this 2-GPU machine.

## Recommended Next Step

Keep v18 `visitonly` as the working default. If selector-family scale work continues here, the next round should reduce run count and/or schedule length first, instead of expanding the selector family again under a budget that does not fit the hardware.
