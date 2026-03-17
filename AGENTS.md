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
- V2 smoke: `bash scripts/smoke_v2.sh`
- V2 main training: `bash scripts/train_v2.sh`
- V2 no-cache ablation: `bash scripts/ablate_v2_no_cache.sh`
- V3 CE smoke: `bash scripts/smoke_v3_router_ce.sh`
- V3 aux smoke: `bash scripts/smoke_v3_router_aux.sh`
- V3 main training: `bash scripts/train_v3_router.sh`
- V3 no-cache ablation: `bash scripts/ablate_v3_router_no_cache.sh`

## Constraints

- Keep the implementation in pure PyTorch
- Preserve per-sample cache isolation
- Prefer correctness and inspectability over aggressive optimization
