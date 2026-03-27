# Final Report v67 Cache Summary Fusion

## What Changed

v67 tests one narrow architecture improvement derived from v66: fuse query-at-home hidden state with the mean home-cache state when cache entries are present.

## Pilot Choices

- Baseline LR multiplier: `0.6`
- Fusion LR multiplier: `0.6`

## Main Results

- `c1` baseline dense `0.0286`, fusion dense `0.0312`, gap `+0.0026`
- `c2` baseline dense `0.0286`, fusion dense `0.0286`, gap `-0.0000`

## Fresh Rerun

- Regime: `c1`
- Baseline dense: `0.0586`
- Fusion dense: `0.0547`

## Probe Audit on C2 Best Checkpoints

- Baseline sink/cache/home probe test acc: `0.000` / `0.300` / `0.100`
- Fusion sink/cache/home probe test acc: `0.100` / `0.250` / `0.050`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
