# Final Report v73 Dispersion-Augmented Output Readout

## What Changed

v73 tests one narrow architecture improvement beyond v68-v72: keep the proven mean home-cache output readout and add a zero-init dispersion-derived correction from the home-cache spread before output.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Dispersion-augmented LR multiplier: `0.8`

## Main Results

- `c1` mean dense `0.0599`, disp dense `0.0625`, gap `+0.0026`
- `c2` mean dense `0.0495`, disp dense `0.0443`, gap `-0.0052`

## Fresh Rerun

- Regime: `c1`
- Mean dense: `0.0625`
- Dispersion-augmented dense: `0.0547`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/cache-disp/home probe test acc: `0.150` / `0.400` / `0.050` / `0.050`
- Dispersion-augmented sink/cache/cache-disp/home probe test acc: `0.100` / `0.350` / `0.100` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
