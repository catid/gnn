# Final Report v80 Ambiguity-Conditioned Class-Slice Suppression

## What Changed

v80 tests one narrow architecture improvement beyond v75/v79: keep the ambiguity-aware mean home-cache output readout and add a zero-init ambiguity-conditioned scaling path on only the reserved class-slice contribution to the base logits.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Ambiguity-class-suppressed LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigclass dense `0.0573`, gap `+0.0026`
- `c2` ambig dense `0.0573`, ambigclass dense `0.0469`, gap `-0.0104`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Ambiguity-class-suppressed dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.000` / `0.250` / `0.300` / `0.100`
- Ambiguity-class-suppressed sink/cache/cache-max/home probe test acc: `0.100` / `0.250` / `0.200` / `0.100`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
