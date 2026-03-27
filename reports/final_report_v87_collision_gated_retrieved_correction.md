# Final Report v87 Collision-Gated Retrieved Correction

## What Changed

v87 tests one narrow architecture improvement beyond v75-v86: keep the ambiguity-aware output gate and add a zero-init retrieved-summary correction head that only contributes under real multi-entry collision.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-retrieved-head LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, retrhead dense `0.0521`, gap `-0.0026`
- `c2` ambig dense `0.0573`, retrhead dense `0.0469`, gap `-0.0104`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Collision-retrieved-head dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.150` / `0.200` / `0.000`
- Collision-retrieved-head sink/cache/cache-max/home probe test acc: `0.000` / `0.300` / `0.050` / `0.100`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
