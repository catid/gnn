# Final Report v83 Collision-Switch Gate

## What Changed

v83 tests one narrow architecture improvement beyond v75: keep the ambiguity-aware mean home-cache output readout and only turn on the feature-conditioned ambiguity adjustment when the home cache actually looks collision-heavy.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-switch LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, collswitch dense `0.0521`, gap `-0.0026`
- `c2` ambig dense `0.0573`, collswitch dense `0.0547`, gap `-0.0026`

## Fresh Rerun

- Regime: `c2`
- Ambiguity-aware dense: `0.0664`
- Collision-switch dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.050` / `0.200` / `0.200` / `0.000`
- Collision-switch sink/cache/cache-max/home probe test acc: `0.100` / `0.250` / `0.300` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
