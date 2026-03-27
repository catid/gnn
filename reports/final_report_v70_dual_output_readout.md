# Final Report v70 Dual Output Readout

## What Changed

v70 tests one narrow architecture improvement beyond v68/v69: keep the proven mean home-cache output readout and add a separate retrieved-summary correction branch at output time.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Dual-readout LR multiplier: `0.6`

## Main Results

- `c1` mean dense `0.0599`, dual dense `0.0599`, gap `+0.0000`
- `c2` mean dense `0.0495`, dual dense `0.0443`, gap `-0.0052`

## Fresh Rerun

- Regime: `c1`
- Mean dense: `0.0625`
- Dual dense: `0.0547`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/home probe test acc: `0.100` / `0.350` / `0.050`
- Dual sink/cache/home probe test acc: `0.000` / `0.350` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
