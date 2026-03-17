# APSGNN

Constrained asynchronous packet-switched graph network experiments in pure PyTorch. The main task is a write-then-query memory-routing problem where writers expire into per-sample node-local cache and a later query must route to the same latent node, read the cached residual, and route to output.

## Repo Layout

- `apsgnn/`: model, routing, buffers, tasks, training, evaluation
- `configs/`: v1 and v2 smoke, main, no-cache, throughput configs
- `scripts/`: setup and run entrypoints
- `tests/`: unit tests for routing, buffers, TTL, cache isolation
- `runs/`: logs, checkpoints, metrics
- `reports/`: final report and plots

## Setup

```bash
bash scripts/setup_env.sh
source .venv/bin/activate
```

## Commands

Smoke:

```bash
bash scripts/smoke.sh
```

Main training:

```bash
bash scripts/train_4gpu.sh
```

No-cache ablation:

```bash
bash scripts/ablate_no_cache.sh
```

Throughput benchmark:

```bash
bash scripts/benchmark_throughput.sh
```

V2 smoke:

```bash
bash scripts/smoke_v2.sh
```

V2 learned-router training:

```bash
bash scripts/train_v2.sh
```

V2 learned-router no-cache ablation:

```bash
bash scripts/ablate_v2_no_cache.sh
```

V3 router CE smoke:

```bash
bash scripts/smoke_v3_router_ce.sh
```

V3 router auxiliary smoke:

```bash
bash scripts/smoke_v3_router_aux.sh
```

V3 selected-router training:

```bash
bash scripts/train_v3_router.sh
```

V3 selected-router no-cache ablation:

```bash
bash scripts/ablate_v3_router_no_cache.sh
```

V4 implicit-retrieval smoke:

```bash
bash scripts/smoke_v4_retrieval_implicit.sh
```

V4 key-conditioned-retrieval smoke:

```bash
bash scripts/smoke_v4_retrieval_keycond.sh
```

V4 selected-retrieval training:

```bash
bash scripts/train_v4_retrieval.sh
```

V4 selected-retrieval no-cache ablation:

```bash
bash scripts/ablate_v4_retrieval_no_cache.sh
```

V5 static sparse smoke:

```bash
bash scripts/smoke_v5_static_sparse.sh
```

V5 static bootstrap smoke:

```bash
bash scripts/smoke_v5_static_bootstrap.sh
```

V5 growth clone smoke:

```bash
bash scripts/smoke_v5_growth_clone.sh
```

V5 growth mutate smoke:

```bash
bash scripts/smoke_v5_growth_mutate.sh
```

V5 static sparse training:

```bash
bash scripts/train_v5_static_sparse.sh
```

V5 static bootstrap training:

```bash
bash scripts/train_v5_static_bootstrap.sh
```

V5 growth clone training:

```bash
bash scripts/train_v5_growth_clone.sh
```

V5 growth mutate training:

```bash
bash scripts/train_v5_growth_mutate.sh
```

V6 static moderate smoke:

```bash
bash scripts/smoke_v6_static_moderate.sh
```

V6 static bootstrap moderate smoke:

```bash
bash scripts/smoke_v6_static_bootstrap_moderate.sh
```

V6 growth clone moderate smoke:

```bash
bash scripts/smoke_v6_growth_clone_moderate.sh
```

V6 static hard smoke:

```bash
bash scripts/smoke_v6_static_hard.sh
```

V6 static bootstrap hard smoke:

```bash
bash scripts/smoke_v6_static_bootstrap_hard.sh
```

V6 growth clone hard smoke:

```bash
bash scripts/smoke_v6_growth_clone_hard.sh
```

V6 growth mutate hard smoke:

```bash
bash scripts/smoke_v6_growth_mutate_followup.sh
```

V6 static moderate training:

```bash
bash scripts/train_v6_static_moderate.sh
```

V6 static bootstrap moderate training:

```bash
bash scripts/train_v6_static_bootstrap_moderate.sh
```

V6 growth clone moderate training:

```bash
bash scripts/train_v6_growth_clone_moderate.sh
```

V6 static hard training:

```bash
bash scripts/train_v6_static_hard.sh
```

V6 static bootstrap hard training:

```bash
bash scripts/train_v6_static_bootstrap_hard.sh
```

V6 growth clone hard training:

```bash
bash scripts/train_v6_growth_clone_hard.sh
```

V6 growth mutate hard follow-up:

```bash
bash scripts/train_v6_growth_mutate_followup.sh
```

V7 static bootstrap hard smoke:

```bash
bash scripts/smoke_v7_static_bootstrap_hard.sh
```

V7 staged static hard smoke:

```bash
bash scripts/smoke_v7_staged_static_hard.sh
```

V7 growth clone hard smoke:

```bash
bash scripts/smoke_v7_growth_clone_hard.sh
```

V7 growth mutate hard smoke:

```bash
bash scripts/smoke_v7_growth_mutate_hard.sh
```

V7 static bootstrap hard training:

```bash
bash scripts/train_v7_static_bootstrap_hard.sh
```

V7 staged static hard training:

```bash
bash scripts/train_v7_staged_static_hard.sh
```

V7 growth clone hard training:

```bash
bash scripts/train_v7_growth_clone_hard.sh
```

V7 growth mutate hard training:

```bash
bash scripts/train_v7_growth_mutate_hard.sh
```

V7 growth mutate hard long training:

```bash
bash scripts/train_v7_growth_mutate_hard_long.sh
```

Manual evaluation of a checkpoint:

```bash
source .venv/bin/activate
torchrun --standalone --nproc_per_node=2 -m apsgnn.eval \
  --config configs/main.yaml \
  --checkpoint runs/<run>/best.pt \
  --tag best_k6_ddp \
  --batches 40 \
  --output runs/<run>/best_k6_ddp.json
python -m apsgnn.eval \
  --config configs/main.yaml \
  --checkpoint runs/<run>/best.pt \
  --writers-per-episode 10 \
  --tag k10 \
  --batches 40 \
  --output runs/<run>/k10.json

torchrun --standalone --nproc_per_node=2 -m apsgnn.eval \
  --config configs/v2_learned_router.yaml \
  --checkpoint runs/<run>/best.pt \
  --tag best_k6_ddp \
  --batches 40 \
  --output runs/<run>/best_k6_ddp.json
```

## Notes

- Address routing uses negative squared L2 to a frozen orthogonal address table with node `0` fixed at the origin.
- APSGNN v2 replaces the frozen first-hop key hint with a learned strongly supervised first-hop router and optional teacher forcing on the first hop only.
- APSGNN v3 keeps the v2 task and memory path stable, but upgrades first-hop routing with a stronger key-centric router and a CE-vs-aux selection path.
- APSGNN v4 keeps the v3 router fixed, warm-starts from the v3 cached checkpoint, freezes the first-hop router, and weakens cache retrieval with learned implicit or learned key-conditioned attention over cached residuals.
- APSGNN v5 keeps the v4 memory path and v3-style router family, but tests a reduced 16-leaf benchmark with a clockwise transport prior, stage bootstraps, and 4->8->16 growth via clone or mutate splitting.
- APSGNN v6 scales the growth study to a harder 32-leaf benchmark, uses task-packet-only coverage metrics, hardens ingress coverage with a restricted start-node pool, and compares static, static+bootstrap, growth clone, and mutate follow-up runs across moderate and hard regimes.
- APSGNN v7 keeps the v6 hard 32-leaf benchmark and adds the key staged-static curriculum control so the main comparison is now static+bootstrap vs staged-static vs growth-clone across multiple seeds.
- Scripts requesting 4 GPUs automatically fall back to the available GPU count.
- Metrics and checkpoints are written to `runs/<timestamp>-<name>/`.
- Final report and summary plots are written to `reports/`.
