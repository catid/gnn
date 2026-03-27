# Final Report v88 Collision-Sharpened Summary Blend

## What Changed

v88 tests one narrow architecture improvement beyond v75-v87: keep the ambiguity-aware output gate and add a zero-init collision-only blend from the mean home-cache summary toward the retrieved summary before output readout.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-sharpened-blend LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, sharpblend dense `0.0573`, gap `+0.0026`
- `c2` ambig dense `0.0573`, sharpblend dense `0.0495`, gap `-0.0078`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Collision-sharpened-blend dense: `0.0625`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.050` / `0.400` / `0.250` / `0.000`
- Collision-sharpened-blend sink/cache/cache-max/home probe test acc: `0.050` / `0.400` / `0.250` / `0.150`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
