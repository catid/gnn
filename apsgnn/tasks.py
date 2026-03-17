from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from apsgnn.config import ExperimentConfig


@dataclass
class MemoryBatch:
    writer_keys: Tensor
    writer_labels: Tensor
    writer_start_nodes: Tensor
    writer_home_nodes: Tensor
    query_keys: Tensor
    query_start_nodes: Tensor
    query_home_nodes: Tensor
    query_labels: Tensor
    query_ttl: Tensor
    query_writer_index: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.query_labels.size(0))

    def to(self, device: torch.device) -> "MemoryBatch":
        return MemoryBatch(
            writer_keys=self.writer_keys.to(device),
            writer_labels=self.writer_labels.to(device),
            writer_start_nodes=self.writer_start_nodes.to(device),
            writer_home_nodes=self.writer_home_nodes.to(device),
            query_keys=self.query_keys.to(device),
            query_start_nodes=self.query_start_nodes.to(device),
            query_home_nodes=self.query_home_nodes.to(device),
            query_labels=self.query_labels.to(device),
            query_ttl=self.query_ttl.to(device),
            query_writer_index=self.query_writer_index.to(device),
        )


@dataclass
class SanityBatch:
    inputs: Tensor
    start_nodes: Tensor
    ttl: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.ttl.size(0))

    def to(self, device: torch.device) -> "SanityBatch":
        return SanityBatch(
            inputs=self.inputs.to(device),
            start_nodes=self.start_nodes.to(device),
            ttl=self.ttl.to(device),
        )


class MemoryRoutingTask:
    def __init__(self, config: ExperimentConfig) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.task.hash_seed)
        self.config = config
        self.home_hash = torch.randn(
            config.model.key_dim,
            config.model.num_compute_nodes,
            generator=generator,
        )

    def generate(
        self,
        batch_size: int,
        seed: int,
        writers_per_episode: int | None = None,
    ) -> MemoryBatch:
        cfg = self.config
        writers = writers_per_episode or cfg.task.writers_per_episode
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        writer_keys = torch.randn(batch_size, writers, cfg.model.key_dim, generator=generator)
        writer_labels = torch.randint(0, cfg.model.num_classes, (batch_size, writers), generator=generator)
        writer_scores = writer_keys @ self.home_hash
        writer_home_nodes = 1 + writer_scores.argmax(dim=-1)
        writer_start_nodes = torch.randint(1, cfg.model.nodes_total, (batch_size, writers), generator=generator)

        batch_indices = torch.arange(batch_size)
        query_writer_index = torch.randint(0, writers, (batch_size,), generator=generator)
        query_keys = writer_keys[batch_indices, query_writer_index].clone()
        query_home_nodes = writer_home_nodes[batch_indices, query_writer_index].clone()
        query_labels = writer_labels[batch_indices, query_writer_index].clone()
        query_start_nodes = torch.randint(1, cfg.model.nodes_total, (batch_size,), generator=generator)
        query_ttl = torch.randint(3, 7, (batch_size,), generator=generator)

        return MemoryBatch(
            writer_keys=writer_keys,
            writer_labels=writer_labels,
            writer_start_nodes=writer_start_nodes,
            writer_home_nodes=writer_home_nodes,
            query_keys=query_keys,
            query_start_nodes=query_start_nodes,
            query_home_nodes=query_home_nodes,
            query_labels=query_labels,
            query_ttl=query_ttl,
            query_writer_index=query_writer_index,
        )


class SanityRoutingTask:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def generate(self, batch_size: int, seed: int) -> SanityBatch:
        cfg = self.config
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        inputs = torch.randn(batch_size, cfg.model.d_model, generator=generator)
        start_nodes = torch.randint(1, cfg.model.nodes_total, (batch_size,), generator=generator)
        ttl = torch.randint(
            cfg.task.sanity_min_ttl,
            cfg.task.sanity_max_ttl + 1,
            (batch_size,),
            generator=generator,
        )
        return SanityBatch(inputs=inputs, start_nodes=start_nodes, ttl=ttl)
