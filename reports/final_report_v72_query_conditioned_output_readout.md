# Final Report v72 Query-Conditioned Output Readout

## What Changed

v72 tests one narrow architecture improvement beyond v68-v71: keep the proven mean home-cache output readout, but condition the cache-summary features on the query residual before output.

## Pilot Choices

- Mean-readout LR multiplier: `0.6`
- Query-conditioned LR multiplier: `0.6`

## Main Results

- `c1` mean dense `0.0599`, qcond dense `0.0469`, gap `-0.0130`
- `c2` mean dense `0.0495`, qcond dense `0.0521`, gap `+0.0026`

## Fresh Rerun

- Regime: `c2`
- Mean dense: `0.0625`
- Query-conditioned dense: `0.0547`

## Probe Audit on C2 Best Checkpoints

- Mean sink/cache/home probe test acc: `0.150` / `0.350` / `0.000`
- Query-conditioned sink/cache/home probe test acc: `0.000` / `0.400` / `0.050`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
