# Final Report v85 Collision-Interaction Floor Gate

## What Changed

v85 tests one narrow architecture improvement beyond v84: keep the collision-split floor gate and add one explicit collision-ambiguity interaction feature to the learned gate delta, so the model can separate low-top-mass multi-entry collision from merely medium occupancy.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Collision-interaction-floor LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, intfloor dense `0.0703`, gap `+0.0156`
- `c2` ambig dense `0.0573`, intfloor dense `0.0547`, gap `-0.0026`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Collision-interaction-floor dense: `0.0625`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.000` / `0.400` / `0.250` / `0.000`
- Collision-interaction-floor sink/cache/cache-max/home probe test acc: `0.000` / `0.250` / `0.150` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
