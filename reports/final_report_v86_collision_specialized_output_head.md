# Final Report v86 Collision-Specialized Output Head

## What Changed

v86 tests one narrow architecture improvement beyond v75-v85: keep the ambiguity-aware output gate and add a zero-init collision-specialized output summary head that only contributes under real multi-entry collision.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-specialized-head LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, splithead dense `0.0703`, gap `+0.0156`
- `c2` ambig dense `0.0573`, splithead dense `0.0521`, gap `-0.0052`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Collision-specialized-head dense: `0.0742`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.000` / `0.350` / `0.150` / `0.000`
- Collision-specialized-head sink/cache/cache-max/home probe test acc: `0.100` / `0.350` / `0.200` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
