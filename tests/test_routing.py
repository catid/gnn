from __future__ import annotations

import torch

from apsgnn.routing import build_address_table, negative_squared_l2


def test_routing_uses_negative_squared_l2() -> None:
    node_addresses = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    packet_address = torch.tensor(
        [
            [0.8, 0.1],
            [0.1, 0.9],
            [0.02, 0.01],
        ],
        dtype=torch.float32,
    )
    logits = negative_squared_l2(packet_address, node_addresses)
    assert logits.argmax(dim=-1).tolist() == [1, 2, 0]


def test_output_node_address_is_exact_zero() -> None:
    table = build_address_table(nodes_total=16, address_dim=128)
    assert torch.equal(table[0], torch.zeros(128))
