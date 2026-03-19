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
- V4 implicit retrieval smoke: `bash scripts/smoke_v4_retrieval_implicit.sh`
- V4 key-conditioned retrieval smoke: `bash scripts/smoke_v4_retrieval_keycond.sh`
- V4 main training: `bash scripts/train_v4_retrieval.sh`
- V4 no-cache ablation: `bash scripts/ablate_v4_retrieval_no_cache.sh`
- V5 static sparse smoke: `bash scripts/smoke_v5_static_sparse.sh`
- V5 static bootstrap smoke: `bash scripts/smoke_v5_static_bootstrap.sh`
- V5 growth clone smoke: `bash scripts/smoke_v5_growth_clone.sh`
- V5 growth mutate smoke: `bash scripts/smoke_v5_growth_mutate.sh`
- V5 static sparse training: `bash scripts/train_v5_static_sparse.sh`
- V5 static bootstrap training: `bash scripts/train_v5_static_bootstrap.sh`
- V5 growth clone training: `bash scripts/train_v5_growth_clone.sh`
- V5 growth mutate training: `bash scripts/train_v5_growth_mutate.sh`
- V6 static moderate smoke: `bash scripts/smoke_v6_static_moderate.sh`
- V6 static bootstrap moderate smoke: `bash scripts/smoke_v6_static_bootstrap_moderate.sh`
- V6 growth clone moderate smoke: `bash scripts/smoke_v6_growth_clone_moderate.sh`
- V6 static hard smoke: `bash scripts/smoke_v6_static_hard.sh`
- V6 static bootstrap hard smoke: `bash scripts/smoke_v6_static_bootstrap_hard.sh`
- V6 growth clone hard smoke: `bash scripts/smoke_v6_growth_clone_hard.sh`
- V6 growth mutate hard smoke: `bash scripts/smoke_v6_growth_mutate_followup.sh`
- V6 static moderate training: `bash scripts/train_v6_static_moderate.sh`
- V6 static bootstrap moderate training: `bash scripts/train_v6_static_bootstrap_moderate.sh`
- V6 growth clone moderate training: `bash scripts/train_v6_growth_clone_moderate.sh`
- V6 static hard training: `bash scripts/train_v6_static_hard.sh`
- V6 static bootstrap hard training: `bash scripts/train_v6_static_bootstrap_hard.sh`
- V6 growth clone hard training: `bash scripts/train_v6_growth_clone_hard.sh`
- V6 growth mutate hard training: `bash scripts/train_v6_growth_mutate_followup.sh`
- V7 static bootstrap hard smoke: `bash scripts/smoke_v7_static_bootstrap_hard.sh`
- V7 staged static hard smoke: `bash scripts/smoke_v7_staged_static_hard.sh`
- V7 growth clone hard smoke: `bash scripts/smoke_v7_growth_clone_hard.sh`
- V7 growth mutate hard smoke: `bash scripts/smoke_v7_growth_mutate_hard.sh`
- V7 static bootstrap hard training: `bash scripts/train_v7_static_bootstrap_hard.sh`
- V7 staged static hard training: `bash scripts/train_v7_staged_static_hard.sh`
- V7 growth clone hard training: `bash scripts/train_v7_growth_clone_hard.sh`
- V7 growth mutate hard training: `bash scripts/train_v7_growth_mutate_hard.sh`
- V7 growth mutate hard long training: `bash scripts/train_v7_growth_mutate_hard_long.sh`
- V8 static bootstrap hard smoke: `bash scripts/smoke_v8_static_bootstrap_hard.sh`
- V8 staged static selective hard smoke: `bash scripts/smoke_v8_staged_static_selective_hard.sh`
- V8 clone selective hard smoke: `bash scripts/smoke_v8_clone_selective_hard.sh`
- V8 random selective hard smoke: `bash scripts/smoke_v8_random_selective_hard.sh`
- V8 utility selective hard smoke: `bash scripts/smoke_v8_utility_selective_hard.sh`
- V8 utility mutate hard smoke: `bash scripts/smoke_v8_utility_mutate_hard.sh`
- V8 static bootstrap hard training: `bash scripts/train_v8_static_bootstrap_hard.sh`
- V8 staged static selective hard training: `bash scripts/train_v8_staged_static_selective_hard.sh`
- V8 clone selective hard training: `bash scripts/train_v8_clone_selective_hard.sh`
- V8 random selective hard training: `bash scripts/train_v8_random_selective_hard.sh`
- V8 utility selective hard training: `bash scripts/train_v8_utility_selective_hard.sh`
- V8 utility mutate hard training: `bash scripts/train_v8_utility_mutate_hard.sh`
- V9 staged static selective long smoke: `bash scripts/smoke_v9_staged_static_selective_long.sh`
- V9 clone selective long smoke: `bash scripts/smoke_v9_clone_selective_long.sh`
- V9 utility selective long smoke: `bash scripts/smoke_v9_utility_selective_long.sh`
- V9 utility mutate long smoke: `bash scripts/smoke_v9_utility_mutate_long.sh`
- V9 utility no-success long smoke: `bash scripts/smoke_v9_utility_nosuccess_long.sh`
- V9 utility no-grad long smoke: `bash scripts/smoke_v9_utility_nograd_long.sh`
- V9 staged static selective long training: `bash scripts/train_v9_staged_static_selective_long.sh`
- V9 clone selective long training: `bash scripts/train_v9_clone_selective_long.sh`
- V9 utility selective long training: `bash scripts/train_v9_utility_selective_long.sh`
- V9 utility mutate long training: `bash scripts/train_v9_utility_mutate_long.sh`
- V9 utility no-success long training: `bash scripts/train_v9_utility_nosuccess_long.sh`
- V9 utility no-grad long training: `bash scripts/train_v9_utility_nograd_long.sh`
- V9 random mutate long training: `bash scripts/train_v9_random_mutate_long.sh`
- V9 transfer H1 utility long training: `bash scripts/train_v9_transfer_h1_utility_long.sh`
- V9 transfer H1 utility-mutate long training: `bash scripts/train_v9_transfer_h1_utility_mutate_long.sh`

## Constraints

- Keep the implementation in pure PyTorch
- Preserve per-sample cache isolation
- Prefer correctness and inspectability over aggressive optimization
- Keep first-hop routing and retrieval changes isolated across experiment rounds
- Prefer single-GPU runs when that is the fairest way to execute matched experiment sweeps on this machine
- For git pushes on this machine, prefer the current forwarded SSH agent over local keys:
  if `SSH_AUTH_SOCK` is stale, locate the newest `/tmp/ssh-*/agent*` socket, confirm it with
  `ssh-add -L` or `ssh -T git@github.com`, and use that socket for `git pull --rebase`, `bd sync`,
  and `git push`; do not fall back to stale local GitHub keys unless explicitly requested
