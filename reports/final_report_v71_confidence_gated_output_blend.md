# Final Report v71 Confidence-Gated Output Blend

## What Changed

v71 tests one narrow architecture improvement beyond v68-v70: keep the proven mean home-cache output readout but blend in retrieved cache content only when retrieval top-mass is high enough.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Confidence-blend LR multiplier: `0.6`

## Main Results

- `c1` mean dense `0.0599`, confblend dense `0.0521`, gap `-0.0078`
- `c2` mean dense `0.0495`, confblend dense `0.0469`, gap `-0.0026`

## Fresh Rerun

- Regime: `c2`
- Mean dense: `0.0625`
- Confidence-blend dense: `0.0625`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/home probe test acc: `0.050` / `0.300` / `0.050`
- Confidence-blend sink/cache/home probe test acc: `0.050` / `0.350` / `0.050`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
