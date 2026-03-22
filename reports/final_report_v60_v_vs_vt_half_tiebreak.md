# APSGNN v60: V vs VT-0.5 Tie-Break

## What Changed

v60 adds a third fresh same-seed six-regime matched pair after v58 and v59 disagreed. It reports both fresh-v60 results and the pooled v58+v59+v60 same-structure aggregate.

## Recommendation

Fresh rule winner: `regime_keyed_vt_home_ingress_v_transfer_non_ingress` (VT-0.5 on Core/T2a, V on T1/T1r/T2b/T2c).
Pooled v58+v59+v60 rule winner: `regime_keyed_vt_home_ingress_v_transfer_non_ingress` (VT-0.5 on Core/T2a, V on T1/T1r/T2b/T2c).

The three-pair pooled split is consistent by regime family:
- `VT-0.5` wins `Core` and `T2a`.
- `V` wins `T1`, `T1r`, `T2b`, and `T2c`.
- That keyed rule now beats both single-selector baselines on both fresh-v60 and pooled v58+v59+v60 scoring.

- Summary JSON: [summary_metrics_v60.json](/home/catid/gnn/reports/summary_metrics_v60.json)
- Report: [final_report_v60_v_vs_vt_half_tiebreak.md](/home/catid/gnn/reports/final_report_v60_v_vs_vt_half_tiebreak.md)
