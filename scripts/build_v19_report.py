#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "reports" / "summary_metrics_v19.json"
REPORT_PATH = ROOT / "reports" / "final_report_v19_selector_family_scale.md"


def fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def build_report(summary: dict) -> str:
    lines: list[str] = []
    lines.append("# APSGNN v19 Selector Family Scale")
    lines.append("")
    lines.append("## Status")
    lines.append("")
    lines.append("v19 was stopped at the scale-feasibility gate rather than after a full selector-family matrix.")
    lines.append(
        f"The 64-leaf benchmark was attempted first, then the allowed 48-leaf fallback was profiled after a narrow growth-selection optimization. "
        f"On the 2 visible GPUs, both remained too slow for the mandated 40-run initial matrix plus 10+ run follow-up campaign."
    )
    lines.append("")
    lines.append("## What Changed From v18")
    lines.append("")
    lines.append("- Added a v19 selector-family config/script scaffold for `visitonly`, `visit+task_grad`, `visit+query_grad`, `full querygrad`, and `querygrad-only`.")
    lines.append("- Added 64-leaf target/schedule support and the allowed 48-leaf fallback configs.")
    lines.append("- Added a grouped size-allocation path in `apsgnn/growth.py` so large selective transitions like `32 -> 48` and `48 -> 64` no longer rely on exhaustive parent-subset enumeration.")
    lines.append("- Added v19 tests covering selector score computation, bootstrap exclusion, larger target mapping, and schedule matching.")
    lines.append("")
    lines.append("## Larger Benchmark Attempt")
    lines.append("")
    lines.append(f"- Attempted final compute leaves: `{summary['attempted_final_compute_leaves']}`")
    lines.append(f"- Fallback final compute leaves: `{summary['fallback_final_compute_leaves']}`")
    lines.append(f"- Visible GPU count used: `{summary['visible_gpu_count']}`")
    lines.append("- 64-leaf schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32 -> 48 -> 64`")
    lines.append("- 48-leaf fallback schedule: `4 -> 6 -> 8 -> 12 -> 16 -> 24 -> 32 -> 48`")
    lines.append("")
    lines.append("## Completed Profiling Runs")
    lines.append("")
    lines.append("| Leaves | Steps run | Runtime | Stage @50 | Active nodes @50 | PPS @50 | Max GB @50 | Task visit cov @50 | Config | Run |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for run in summary["completed_profile_runs"]:
        lines.append(
            f"| `{run['final_compute_leaves']}` | `{run['train_steps_run']}` | `~{run['observed_runtime_seconds_approx']}s` | "
            f"`{run['stage_at_step_50']}` | `{run['active_compute_nodes_at_step_50']}` | "
            f"`{run['train_packets_per_second_at_step_50']:.1f}` | `{run['train_max_memory_gb_at_step_50']:.3f}` | "
            f"`{run['task_visit_coverage_at_step_50']:.3f}` | `{run['config']}` | "
            f"`{run['run_dir']}` |"
        )
    lines.append("")
    lines.append("## Why The Full Matrix Was Not Completed")
    lines.append("")
    lines.append("- The 64-leaf 50-step profile already reached stage index `8` with `64` active compute nodes and took about `61s` for only `50` steps.")
    lines.append("- The allowed 48-leaf fallback still took about `49s` for `50` steps.")
    lines.append("- Those profiles imply lower-bound runtimes of roughly `3.05h` per Core-L run and `3.39h` per T1-L run at 64 leaves, or `2.45h` and `2.72h` respectively even at the 48-leaf fallback.")
    lines.append("- Lower-bound wall-clock for the required initial matrix alone:")
    lines.append(f"  - 64-leaf attempt: `~{summary['lower_bound_wallclock_estimates_hours']['initial_matrix_64']:.1f}h`")
    lines.append(f"  - 48-leaf fallback: `~{summary['lower_bound_wallclock_estimates_hours']['initial_matrix_48']:.1f}h`")
    lines.append("- Lower-bound wall-clock for the full requested campaign:")
    lines.append(f"  - 64-leaf attempt: `~{summary['lower_bound_wallclock_estimates_hours']['minimum_full_campaign_64']:.1f}h`")
    lines.append(f"  - 48-leaf fallback: `~{summary['lower_bound_wallclock_estimates_hours']['minimum_full_campaign_48']:.1f}h`")
    lines.append("- These are optimistic lower bounds and do not include slower later-stage behavior, extra eval sweeps, report generation time, or the required confirmatory reruns.")
    lines.append("")
    lines.append("## Partial Runs Not Counted")
    lines.append("")
    for run in summary["partial_runs_not_counted"]:
        lines.append(f"- `{run}`")
    lines.append("")
    lines.append("## Tests And Verification")
    lines.append("")
    lines.append("- `python -m compileall apsgnn tests scripts/run_v19_eval_sweep.py`")
    lines.append("- `pytest -q tests/test_v19_selector_family.py`")
    lines.append("- `pytest -q`")
    lines.append("- Final test status before stopping: `83 passed`")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        "v19 did not produce a valid selector-family comparison because the requested scale-up campaign was not feasible on the available hardware at the mandated budgets. "
        "The correct stopping point was to preserve the 64-leaf attempt, the 48-leaf fallback, the growth-selection optimization that made the attempt runnable at all, and the profiling evidence showing why the full matrix should not be faked."
    )
    lines.append("")
    lines.append("## Recommended Next Step")
    lines.append("")
    lines.append(
        "Keep the v18 `visitonly` selector as the working default and revisit selector-family scale only with either more compute or materially shorter schedules. "
        "If selector-family exploration resumes under current hardware, the only defensible version is a reduced-budget campaign rather than the original v19 matrix."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    summary = json.loads(SUMMARY_PATH.read_text())
    REPORT_PATH.write_text(build_report(summary))


if __name__ == "__main__":
    main()
