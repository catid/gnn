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
- Scripts requesting 4 GPUs automatically fall back to the available GPU count.
- Metrics and checkpoints are written to `runs/<timestamp>-<name>/`.
- Final report and summary plots are written to `reports/`.
