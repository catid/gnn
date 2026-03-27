# Final Report v78 Ambiguity-Feature Output Modulation

## What Changed

v78 tests one narrow architecture improvement beyond v75/v77: keep the ambiguity-aware mean home-cache output readout and add a zero-init ambiguity-feature FiLM on the cache-summary state before output readout.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Ambiguity-feature-modulated LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigfilm dense `0.0573`, gap `+0.0026`
- `c2` ambig dense `0.0573`, ambigfilm dense `0.0443`, gap `-0.0130`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Ambiguity-feature-modulated dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.000` / `0.300` / `0.200` / `0.000`
- Ambiguity-feature-modulated sink/cache/cache-max/home probe test acc: `0.050` / `0.300` / `0.250` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
