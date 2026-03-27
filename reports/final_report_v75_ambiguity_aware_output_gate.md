# Final Report v75 Ambiguity-Aware Output Gate

## What Changed

v75 tests one narrow architecture improvement beyond v68-v74: keep the proven mean home-cache output readout and add a zero-init ambiguity-aware correction to the output gate using retrieval top-mass and normalized entry count.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Ambiguity-aware LR multiplier: `0.6`

## Main Results

- `c1` mean dense `0.0599`, ambig dense `0.0547`, gap `-0.0052`
- `c2` mean dense `0.0495`, ambig dense `0.0573`, gap `+0.0078`

## Fresh Rerun

- Regime: `c2`
- Mean dense: `0.0625`
- Ambiguity-aware dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/cache-max/home probe test acc: `0.000` / `0.250` / `0.300` / `0.050`
- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.100` / `0.200` / `0.050`

## Conclusion

- Positive: `True`
- Next move: `promote this rescue to a broader collision follow-up`
