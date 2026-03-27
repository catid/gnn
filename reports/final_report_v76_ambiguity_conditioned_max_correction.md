# Final Report v76 Ambiguity-Conditioned Max Correction

## What Changed

v76 tests one narrow architecture improvement beyond v75: keep the ambiguity-aware mean home-cache output readout and add a zero-init max-summary correction under the same output-time ambiguity gate.

## Pilot Choices

- Ambiguity-aware LR multiplier: `0.6`
- Ambiguity+max LR multiplier: `0.6`

## Main Results

- `c1` ambig dense `0.0547`, ambigmax dense `0.0625`, gap `+0.0078`
- `c2` ambig dense `0.0573`, ambigmax dense `0.0443`, gap `-0.0130`

## Fresh Rerun

- Regime: `c1`
- Ambiguity-aware dense: `0.0664`
- Ambiguity+max dense: `0.0664`

## Probe Audit on C2 Best Checkpoints

- Ambiguity-aware sink/cache/cache-max/home probe test acc: `0.100` / `0.350` / `0.150` / `0.000`
- Ambiguity+max sink/cache/cache-max/home probe test acc: `0.000` / `0.250` / `0.100` / `0.000`

## Conclusion

- Positive: `False`
- Next move: `do not promote this rescue as the main path`
