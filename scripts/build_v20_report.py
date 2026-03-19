#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "summary_metrics_v20.json"
REPORT_PATH = ROOT / "reports" / "final_report_v20_budgeted_selector_scale.md"


def build_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# APSGNN v20 Budgeted Selector Scale")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append(
        "v20 stopped at the budget gate. The campaign was designed specifically to avoid another fantasy-scale matrix, "
        "and the measured 64/48/40/32 profiles showed that the full 50-run plan was still too expensive on the 2-GPU machine under the requested harder regimes and S/M/L schedules."
    )
    lines.append("")
    lines.append("## What Changed From v19")
    lines.append("")
    lines.append("- Added a v20 config family for 32- and 40-leaf selector-family screening/confirmation/transfer schedules.")
    lines.append("- Added 64/48/40/32 gate configs for a clean scale/budget decision.")
    lines.append("- Added focused v20 tests for selector scoring, bootstrap exclusion, schedule matching, larger-scale target mapping, and reproducibility.")
    lines.append("")
    lines.append("## Budget Gate")
    lines.append("")
    lines.append("- Candidate scales profiled: `64`, `48`, `40`, `32`")
    lines.append(f"- Visible GPU count used: `{summary['visible_gpu_count']}`")
    lines.append(f"- Largest technically runnable scale: `{summary['largest_technically_runnable_scale']}`")
    lines.append("- Feasible campaign scale selected: `none`")
    lines.append("")
    lines.append("| Scale | Runtime @100 | Stage @100 | Active @100 | Task visit cov @100 | Max GB @100 | Config | Run |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for gate in summary["gate_profiles"]:
        lines.append(
            f"| `{gate['scale']}` | `~{gate['runtime_seconds_approx']}s` | `{gate['stage_index_at_step_100']}` | "
            f"`{gate['active_compute_nodes_at_step_100']}` | `{gate['task_visit_coverage_at_step_100']:.3f}` | "
            f"`{gate['max_memory_gb_at_step_100']:.3f}` | `{gate['config']}` | `{gate['run_dir']}` |"
        )
    lines.append("")
    lines.append("## Projected Full-Campaign Cost")
    lines.append("")
    lines.append("| Scale | S min/run | M min/run | L min/run | 50-run campaign, 2-GPU DDP | 50-run campaign, optimistic 2x single-GPU |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for scale in ["64", "48", "40", "32"]:
        per = summary["schedule_minutes_per_run"][scale]
        total = summary["campaign_projection_hours"][scale]
        lines.append(
            f"| `{scale}` | `{per['S']:.1f}` | `{per['M']:.1f}` | `{per['L']:.1f}` | "
            f"`{total['ddp_two_gpu_single_run']:.1f}h` | `{total['optimistic_two_single_gpu_parallel']:.1f}h` |"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "64 and 48 were immediately ruled out by runtime. 40 was still too expensive to justify a 50-run campaign. "
        "32 was the only scale that remained technically runnable, but the harder v20 task settings made it much slower than the older v18 home regime, "
        "and the projected campaign still landed in the ~22h range even before extra eval sweeps, aggregation, and reruns."
    )
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "v20 did not produce a selector-family winner. The useful result is the budget gate itself: under the requested harder regimes and verification requirements, "
        "no scale from `64/48/40/32` yields a realistic full v20 campaign on this 2-GPU machine."
    )
    lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "Keep v18 `visitonly` as the working default. If selector-family scale work continues here, the next round should reduce run count and/or schedule length first, "
        "instead of expanding the selector family again under a budget that does not fit the hardware."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    REPORT_PATH.write_text(build_report(summary), encoding="utf-8")


if __name__ == "__main__":
    main()
