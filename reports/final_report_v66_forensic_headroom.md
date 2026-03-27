# APSGNN v66: Forensic Architectural Headroom

## What Changed From v65

v66 treats `gnn` as a forensic diagnostic repo instead of the main frontier for broad mechanism search. The budget stays on the incumbent `VT-0.5 / D`, with `V / D` used only as a narrow control, and the main work goes into collision failure diagnosis plus a delay-benchmark audit/redesign.

## Chosen Base

- contract family: `d`
- incumbent: `visit_taskgrad_half_d`
- control: `visitonly_d`
- visible GPUs used: `2`
- schedules: `P=300`, `M=1350`, `L=2160`

## Collision Pack Definitions

- `c1`: writers `6`, home-pool `8`, start-pool `2`, ttl `2-3`
- `c2`: writers `6`, home-pool `2`, start-pool `2`, ttl `2-3`

## Delay Pack Definitions

- `d1`: mode `required_wait`, required-delay `1-2`, hash-bits `2`
- `d2`: mode `required_wait`, required-delay `3-4`, hash-bits `2`
- `rd1`: mode `key_hash_exact_wait`, required-delay `1-2`, hash-bits `2`
- `rd2`: mode `key_hash_exact_wait`, required-delay `2-4`, hash-bits `3`

## Completed Runs

| Pack | Regime | Condition | Pair | Sched | Seed | Dense | Last |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| collision | c1 | cacheon | VT-0.5/D | m | 2234 | 0.0208 | 0.0000 |
| collision | c1 | nocache | VT-0.5/D | m | 2234 | 0.0208 | 0.0000 |
| collision | c1 | recent1 | VT-0.5/D | m | 2234 | 0.0208 | 0.0000 |
| collision | c1 | recent1 | VT-0.5/D | m | 3234 | 0.0365 | 0.0417 |
| collision | c1 | topk1 | VT-0.5/D | m | 2234 | 0.0208 | 0.0000 |
| collision | c2 | cacheon | VT-0.5/D | m | 2234 | 0.0260 | 0.0000 |
| collision | c2 | cacheon | V/D | m | 4234 | 0.0417 | 0.0417 |
| collision | c2 | nocache | VT-0.5/D | m | 2234 | 0.0208 | 0.0000 |
| collision | c2 | nocache | V/D | m | 4234 | 0.0417 | 0.0417 |
| collision | c2 | recent1 | VT-0.5/D | m | 2234 | 0.0260 | 0.0000 |
| collision | c2 | recent1 | VT-0.5/D | m | 3234 | 0.0312 | 0.0000 |
| collision | c2 | topk1 | VT-0.5/D | m | 2234 | 0.0312 | 0.0000 |
| delay | d1 | learned | VT-0.5/D | m | 2234 | 0.0417 | 0.0417 |
| delay | d1 | zero | VT-0.5/D | m | 2234 | 0.0421 | 0.0833 |
| delay | d2 | learned | VT-0.5/D | m | 2234 | 0.0312 | 0.0833 |
| delay | d2 | zero | VT-0.5/D | m | 2234 | 0.0521 | 0.0833 |
| delay | rd1 | learned | VT-0.5/D | m | 2234 | 0.0417 | 0.0417 |
| delay | rd1 | learned | VT-0.5/D | m | 3234 | 0.0312 | 0.0000 |
| delay | rd1 | zero | VT-0.5/D | m | 2234 | 0.0421 | 0.0833 |
| delay | rd1 | zero | VT-0.5/D | m | 3234 | 0.0417 | 0.0000 |
| delay | rd2 | learned | VT-0.5/D | m | 2234 | 0.0426 | 0.0417 |
| delay | rd2 | learned | VT-0.5/D | m | 3234 | 0.0312 | 0.0000 |
| delay | rd2 | zero | VT-0.5/D | m | 2234 | 0.0417 | 0.0833 |
| delay | rd2 | zero | VT-0.5/D | m | 3234 | 0.0417 | 0.0000 |

## Collision Pack Summary

- `c1` baseline: cache-on `0.0208`, cache-off `0.0208`, gap `0.0000`
- `c2` baseline: cache-on `0.0260`, cache-off `0.0208`, gap `0.0052`
- `recent1:c1`: dense `0.0286`, recovery vs cache-on `0.0078`, recovery fraction `0.0000`
- `topk1:c1`: dense `0.0208`, recovery vs cache-on `0.0000`, recovery fraction `0.0000`
- `recent1:c2`: dense `0.0286`, recovery vs cache-on `0.0026`, recovery fraction `0.0000`
- `topk1:c2`: dense `0.0312`, recovery vs cache-on `0.0052`, recovery fraction `0.0000`
- bypass `cacheon:c1`: normal `0.0312`, bypass `0.0312`, delta `0.0000`
- bypass `recent1:c1`: normal `0.0521`, bypass `0.0729`, delta `-0.0208`
- bypass `cacheon:c2`: normal `0.0417`, bypass `0.0312`, delta `0.0104`
- bypass `recent1:c2`: normal `0.0312`, bypass `0.0729`, delta `-0.0417`
- best intervention: `recent1`
- collision bundle positive: `False`

## Delay Pack Audit Summary

- current `d1`: learned `0.0417`, zero `0.0421`, gap `-0.0004`
- current `d2`: learned `0.0312`, zero `0.0521`, gap `-0.0208`
- redesigned `rd1`: learned `0.0365`, zero `0.0419`, gap `-0.0054`
- redesigned `rd2`: learned `0.0369`, zero `0.0417`, gap `-0.0048`
- timing audit `d1`: acceptable-delay-count `6.49`, zero `0.0000`, fixed `1.0000`, random `0.8116`, oracle `1.0000`
- timing audit `d2`: acceptable-delay-count `4.49`, zero `0.0000`, fixed `0.4929`, random `0.5616`, oracle `1.0000`
- timing audit `rd1`: acceptable-delay-count `1.00`, zero `0.0000`, fixed `0.5020`, random `0.1250`, oracle `1.0000`
- timing audit `rd2`: acceptable-delay-count `1.00`, zero `0.0000`, fixed `0.3716`, random `0.1250`, oracle `1.0000`
- current benchmark needs redesign: `True`
- redesigned benchmark positive: `False`

## Decodability / Source-Quality Audit

- `collision` audited checkpoints: `3`
  - `cacheon:c2` sink `0.100`, cache `0.150`, hidden `0.050`
  - `nocache:c2` sink `0.050`, cache `0.000`, hidden `0.000`
  - `recent1:c2` sink `0.000`, cache `0.150`, hidden `0.000`
- `delay` audited checkpoints: `4`
  - `d2:learned` sink `0.000`, cache `0.615`, hidden `0.000`
  - `d2:zero` sink `0.000`, cache `0.538`, hidden `0.000`
  - `rd2:learned` sink `0.000`, cache `0.500`, hidden `0.000`
  - `rd2:zero` sink `0.000`, cache `0.615`, hidden `0.000`

## Optional Follow-up

- not triggered

## Final Diagnosis

- headroom conclusion: `low_priority_maintenance`
- next move: `shift_main_research_effort_elsewhere`
