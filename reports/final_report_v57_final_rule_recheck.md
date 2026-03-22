# APSGNN v57: Final Rule Recheck

## What Changed

v57 reruns the six-regime matched selector comparison on fresh same-seed pairs after the v56 StageLate package retirement. It compares VT-0.5 directly against mutation-free StageLate-0.5 and scores the two practical decision rules.

## Recommendation

Fresh rule winner: `single_selector_vt_half` (VT-0.5 everywhere).
Pooled v51+v57 rule winner: `single_selector_vt_half` (VT-0.5 everywhere).

- Summary JSON: [summary_metrics_v57.json](/home/catid/gnn/reports/summary_metrics_v57.json)
- Report: [final_report_v57_final_rule_recheck.md](/home/catid/gnn/reports/final_report_v57_final_rule_recheck.md)
