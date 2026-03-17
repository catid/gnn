# Agent Notes

## Issue Tracking

This repo uses `bd` for task tracking.

- Start with `bd ready` or `bd show <id>`
- Claim with `bd update <id> --claim`
- Close with `bd close <id>`
- Keep Beads operations sequential, not parallel, to avoid Dolt server races

## Project Commands

- Environment: `bash scripts/setup_env.sh`
- Smoke: `bash scripts/smoke.sh`
- Main training: `bash scripts/train_4gpu.sh`
- No-cache ablation: `bash scripts/ablate_no_cache.sh`
- Throughput benchmark: `bash scripts/benchmark_throughput.sh`

## Constraints

- Keep the implementation in pure PyTorch
- Preserve per-sample cache isolation
- Prefer correctness and inspectability over aggressive optimization
