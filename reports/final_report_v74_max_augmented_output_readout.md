# Final Report v74 Max-Augmented Output Readout

## What Changed

v74 tests one narrow architecture improvement beyond v68-v73: keep the proven mean home-cache output readout and add a zero-init max-summary correction from the home cache before output.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Max-augmented LR multiplier: `0.6`

## Main Results

- `c1` mean dense `0.0599`, maxaug dense `0.0651`, gap `+0.0052`
- `c2` mean dense `0.0495`, maxaug dense `0.0495`, gap `+0.0000`

## Fresh Rerun

- Regime: `c1`
- Mean dense: `0.0625`
- Max-augmented dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/cache-max/home probe test acc: `0.050` / `0.350` / `0.250` / `0.050`
- Max-augmented sink/cache/cache-max/home probe test acc: `0.050` / `0.250` / `0.150` / `0.050`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
