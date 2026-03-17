# APSGNN v1

Constrained asynchronous packet-switched graph network experiment in pure PyTorch. The main task is a write-then-query memory-routing problem where writers expire into per-sample node-local cache and a later query must route to the same latent node, read the cached residual, and route to output.

## Repo Layout

- `apsgnn/`: model, routing, buffers, tasks, training, evaluation
- `configs/`: smoke, main, no-cache, throughput configs
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
```

## Notes

- Address routing uses negative squared L2 to a frozen orthogonal address table with node `0` fixed at the origin.
- Scripts requesting 4 GPUs automatically fall back to the available GPU count.
- Metrics and checkpoints are written to `runs/<timestamp>-<name>/`.
- Final report and summary plots are written to `reports/`.
