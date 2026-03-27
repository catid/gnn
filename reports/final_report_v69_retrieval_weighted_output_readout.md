# Final Report v69 Retrieval-Weighted Output Readout

## What Changed

v69 tests one narrow architecture improvement beyond v68: replace the coarse mean home-cache output readout with a retrieval-weighted cache summary at output time.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Retrieved-readout LR multiplier: `1.0`

## Main Results

- `c1` mean dense `0.0599`, retrieved dense `0.0417`, gap `-0.0182`
- `c2` mean dense `0.0495`, retrieved dense `0.0365`, gap `-0.0130`

## Fresh Rerun

- Regime: `c2`
- Mean dense: `0.0625`
- Retrieved dense: `0.0547`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/home probe test acc: `0.050` / `0.300` / `0.000`
- Retrieved sink/cache/home probe test acc: `0.150` / `0.200` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
