# Final Report v82 Confidence-Floor Gate

## What Changed

v82 tests one narrow architecture improvement beyond v75: keep the ambiguity-aware mean home-cache output readout and add a low-ambiguity gate floor driven by retrieval top-mass.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Confidence-floor LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, confloor dense `0.0599`, gap `+0.0052`
- `c2` ambig dense `0.0573`, confloor dense `0.0547`, gap `-0.0026`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Confidence-floor dense: `0.0625`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.050` / `0.350` / `0.350` / `0.000`
- Confidence-floor sink/cache/cache-max/home probe test acc: `0.000` / `0.350` / `0.300` / `0.100`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
