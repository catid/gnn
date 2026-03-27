# Final Report v84 Collision-Split Floor Gate

## What Changed

v84 tests one narrow architecture improvement beyond v75: keep the ambiguity-aware mean home-cache output readout, apply the learned ambiguity delta only when the home cache looks collision-heavy, and apply a confidence floor only in low-collision cases.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-split-floor LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, splitfloor dense `0.0625`, gap `+0.0078`
- `c2` ambig dense `0.0573`, splitfloor dense `0.0573`, gap `+0.0000`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Collision-split-floor dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.400` / `0.300` / `0.000`
- Collision-split-floor sink/cache/cache-max/home probe test acc: `0.100` / `0.350` / `0.300` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
