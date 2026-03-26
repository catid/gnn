# APSGNN v65: Architectural Headroom

## What Changed From v64

v65 stops spending the main budget on selector micro-tiebreaks and instead uses benchmark-pack discipline to test unresolved architectural debt: heavy-collision retrieval, reserved class-slice removal, and a benchmark where nonzero delay is causally necessary.

## Chosen Contract Family

- chosen contract family: `d`
- incumbent pair: `visit_taskgrad_half_d`
- runner-up pair: `visitonly_d`

## Collision Pack Summary

- `c0`: cache-on dense `0.0339`, cache-off dense `0.0365`, gap `-0.0026`
- `c1`: cache-on dense `0.0260`, cache-off dense `0.0365`, gap `-0.0104`
- `c2`: cache-on dense `0.0208`, cache-off dense `0.0243`, gap `-0.0035`

## Class-Slice Pack Summary

- skipped: collision gate did not pass, so class-slice-off confirmation was not run.
- `c1`: class-on reference dense `0.0260`; class-off not run
- `c2`: class-on reference dense `0.0208`; class-off not run

## Delay Pack Summary

- validation controls:
  - `learned` dense `0.0312`, first-hop nonzero `1.0000`
  - `zero` dense `0.0312`, first-hop nonzero `0.0000`
  - `random` dense `0.0541`, first-hop nonzero `0.8906`
  - `fixed` dense `0.0312`, first-hop nonzero `1.0000`
- `d0`: learned dense `0.0365`, forced-zero dense `0.0312`, gap `0.0052`
- `d1`: learned dense `0.0365`, forced-zero dense `0.0419`, gap `-0.0054`
- `d2`: learned dense `0.0208`, forced-zero dense `0.0469`, gap `-0.0260`

## Decodability Audit

- audited checkpoints: `4`

## Final Diagnosis

- collision gate passed: `False`
- delay gate passed: `False`
- optional follow-up triggered: `False`
- architectural headroom conclusion: `architectural_headroom_weakened`
