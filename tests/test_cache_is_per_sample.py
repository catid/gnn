from __future__ import annotations

import torch

from apsgnn.buffer import NodeCache


def test_cache_isolated_per_sample() -> None:
    cache = NodeCache(
        batch_size=2,
        nodes_total=4,
        capacity=4,
        d_model=3,
        device=torch.device("cpu"),
        dtype=torch.float32,
        enabled=True,
    )
    residuals = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [9.0, 8.0, 7.0],
        ]
    )
    batch_index = torch.tensor([0, 1])
    node_index = torch.tensor([2, 2])
    cache.write(residuals, batch_index, node_index)

    gathered, mask = cache.gather(batch_index, node_index)
    assert mask[0, 0].item() is True
    assert mask[1, 0].item() is True
    assert torch.equal(gathered[0, 0], residuals[0])
    assert torch.equal(gathered[1, 0], residuals[1])
    assert not torch.equal(gathered[0, 0], gathered[1, 0])
