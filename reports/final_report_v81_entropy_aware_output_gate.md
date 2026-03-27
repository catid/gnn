# Final Report v81 Entropy-Aware Output Gate

## What Changed

v81 tests one narrow architecture improvement beyond v75: keep the ambiguity-aware mean home-cache output readout and add normalized retrieval entropy as a third output-gate feature so the gate can distinguish diffuse collision from concentrated multi-entry retrieval.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Entropy-aware LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigent dense `0.0677`, gap `+0.0130`
- `c2` ambig dense `0.0573`, ambigent dense `0.0469`, gap `-0.0104`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Entropy-aware dense: `0.0586`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.300` / `0.300` / `0.000`
- Entropy-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.300` / `0.200` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
