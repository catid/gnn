# Final Report v79 Ambiguity-Conditioned Base Suppression

## What Changed

v79 tests one narrow architecture improvement beyond v75/v78: keep the ambiguity-aware mean home-cache output readout and add a zero-init ambiguity-conditioned suppression/boost path on the base output logits.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Ambiguity-base-suppressed LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigbase dense `0.0573`, gap `+0.0026`
- `c2` ambig dense `0.0573`, ambigbase dense `0.0469`, gap `-0.0104`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Ambiguity-base-suppressed dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.000` / `0.350` / `0.300` / `0.050`
- Ambiguity-base-suppressed sink/cache/cache-max/home probe test acc: `0.000` / `0.300` / `0.300` / `0.100`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
