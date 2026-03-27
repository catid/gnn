# Final Report v68 Cache Output Readout

## What Changed

v68 tests one narrow architecture improvement derived from v66/v67: add a direct output-time cache-summary readout head for query packets leaving home.

## Pilot Choices

- Baseline LR multiplier: `0.6`
- Readout LR multiplier: `0.6`

## Main Results

- `c1` baseline dense `0.0286`, readout dense `0.0599`, gap `+0.0312`
- `c2` baseline dense `0.0286`, readout dense `0.0495`, gap `+0.0208`

## Fresh Rerun

- Regime: `c1`
- Baseline dense: `0.0586`
- Readout dense: `0.0625`

## Probe Audit on C2 Best Checkpoints

- Baseline sink/cache/home probe test acc: `0.000` / `0.300` / `0.100`
- Readout sink/cache/home probe test acc: `0.000` / `0.350` / `0.050`

## Conclusion

- Positive: `True`
- Next move: `promote this rescue to a broader collision follow-up`
