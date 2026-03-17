from __future__ import annotations

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.model import APSGNNModel
from apsgnn.node import ComputeNodeCell
from apsgnn.tasks import MemoryRoutingTask


def test_no_cache_ablation_disables_cache_path(monkeypatch) -> None:
    config = ExperimentConfig()
    config.model.enable_cache = False
    config.model.nodes_total = 8
    config.model.cache_capacity = 8
    config.task.writers_per_episode = 2
    config.task.max_rollout_steps = 4
    config.train.batch_size_per_gpu = 2

    seen_cache_masks: list[bool] = []
    original_forward = ComputeNodeCell.forward

    def wrapped_forward(self, packets, packet_mask, cache, cache_mask):
        seen_cache_masks.append(bool(cache_mask.any().item()))
        return original_forward(self, packets, packet_mask, cache, cache_mask)

    monkeypatch.setattr(ComputeNodeCell, "forward", wrapped_forward)
    model = APSGNNModel(config)
    model.eval()
    task = MemoryRoutingTask(config)
    batch = task.generate(batch_size=2, seed=0).to(torch.device("cpu"))
    model(batch)

    assert seen_cache_masks
    assert not any(seen_cache_masks)
