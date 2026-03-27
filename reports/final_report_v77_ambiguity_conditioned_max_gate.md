# Final Report v77 Ambiguity-Conditioned Max Gate

## What Changed

v77 tests one narrow architecture improvement beyond v75/v76: keep the ambiguity-aware mean home-cache output readout and add a zero-init max-summary correction whose strength is itself conditioned on the same ambiguity features.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Ambiguity+max-gate LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigmaxgate dense `0.0521`, gap `-0.0026`
- `c2` ambig dense `0.0573`, ambigmaxgate dense `0.0495`, gap `-0.0078`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Ambiguity+max-gate dense: `0.0742`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.050` / `0.400` / `0.300` / `0.100`
- Ambiguity+max-gate sink/cache/cache-max/home probe test acc: `0.100` / `0.100` / `0.200` / `0.050`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
