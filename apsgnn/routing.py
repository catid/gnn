from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def build_address_table(nodes_total: int, address_dim: int, device: torch.device | None = None) -> Tensor:
    if nodes_total - 1 > address_dim:
        raise ValueError("Need address_dim >= nodes_total - 1 for orthogonal compute addresses.")
    basis = torch.empty(address_dim, address_dim, device=device)
    nn.init.orthogonal_(basis)
    table = torch.zeros(nodes_total, address_dim, device=device)
    table[1:] = basis[: nodes_total - 1]
    return table


def negative_squared_l2(address: Tensor, node_addresses: Tensor) -> Tensor:
    diff = address[:, None, :] - node_addresses[None, :, :]
    return -(diff.square().sum(dim=-1))


def straight_through_sample(logits: Tensor, temperature: float, training: bool) -> Tensor:
    if training:
        return F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)
    indices = logits.argmax(dim=-1)
    hard = F.one_hot(indices, num_classes=logits.size(-1)).to(logits.dtype)
    soft = logits.softmax(dim=-1)
    return hard - soft.detach() + soft


def route_from_address(
    address: Tensor,
    node_addresses: Tensor,
    temperature: float,
    training: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    logits = negative_squared_l2(address, node_addresses)
    one_hot = straight_through_sample(logits, temperature=temperature, training=training)
    indices = one_hot.argmax(dim=-1)
    return logits, one_hot, indices


def sample_delay(
    delay_logits: Tensor,
    temperature: float,
    training: bool,
) -> tuple[Tensor, Tensor]:
    one_hot = straight_through_sample(delay_logits, temperature=temperature, training=training)
    indices = one_hot.argmax(dim=-1)
    return one_hot, indices
