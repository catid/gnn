from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from apsgnn.config import ExperimentConfig
from apsgnn.growth import (
    GrowthTopology,
    active_node_ids,
    active_node_ids_from_topology,
    project_home_leaves,
    project_home_leaves_topology,
)


def sample_start_nodes(
    active_nodes: int | None,
    pool_size: int,
    shape: tuple[int, ...],
    generator: torch.Generator,
    *,
    active_node_ids_tensor: Tensor | None = None,
) -> Tensor:
    if active_node_ids_tensor is None:
        if active_nodes is None:
            raise ValueError("active_nodes is required when active_node_ids_tensor is not provided.")
        active_ids = active_node_ids(active_nodes)
    else:
        active_ids = active_node_ids_tensor.to(dtype=torch.long).reshape(-1).cpu()
        active_nodes = int(active_ids.numel())

    if pool_size <= 0 or pool_size >= active_nodes:
        sampled = torch.randint(0, active_nodes, shape, generator=generator)
        return active_ids[sampled]

    stride = active_nodes / float(pool_size)
    ingress_indices = torch.floor(torch.arange(pool_size, dtype=torch.float32) * stride).long()
    ingress_nodes = active_ids[ingress_indices.clamp_max(active_nodes - 1)]
    sampled = torch.randint(0, ingress_nodes.numel(), shape, generator=generator)
    return ingress_nodes[sampled]


def delay_targets_from_key(
    keys: Tensor,
    *,
    min_delay: int,
    max_delay: int,
    hash_bits: int,
) -> Tensor:
    span = max(int(max_delay - min_delay + 1), 1)
    bits = max(int(hash_bits), 1)
    take = min(bits, int(keys.size(-1)))
    signs = (keys[..., :take] > 0).to(torch.long)
    weights = (2 ** torch.arange(take, dtype=torch.long)).view(*([1] * (signs.dim() - 1)), take)
    hashed = (signs * weights).sum(dim=-1)
    return int(min_delay) + (hashed % span)


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
    query_required_delay: Tensor | None = None
    bootstrap_start_nodes: Tensor | None = None
    bootstrap_ttl: Tensor | None = None

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
            query_required_delay=None if self.query_required_delay is None else self.query_required_delay.to(device),
            bootstrap_start_nodes=None if self.bootstrap_start_nodes is None else self.bootstrap_start_nodes.to(device),
            bootstrap_ttl=None if self.bootstrap_ttl is None else self.bootstrap_ttl.to(device),
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
        self.home_candidates = self._home_candidates(
            num_compute_nodes=config.model.num_compute_nodes,
            pool_size=config.task.home_node_pool_size,
        )

    @staticmethod
    def _home_candidates(*, num_compute_nodes: int, pool_size: int) -> Tensor:
        if pool_size <= 0 or pool_size >= num_compute_nodes:
            return torch.arange(1, num_compute_nodes + 1, dtype=torch.long)
        stride = num_compute_nodes / float(pool_size)
        candidate_indices = torch.floor(torch.arange(pool_size, dtype=torch.float32) * stride).long()
        candidate_indices = candidate_indices.clamp_max(num_compute_nodes - 1)
        return 1 + torch.unique_consecutive(candidate_indices)

    def _sample_home_nodes(self, writer_keys: Tensor) -> Tensor:
        scores = writer_keys @ self.home_hash
        if int(self.home_candidates.numel()) == self.config.model.num_compute_nodes:
            return 1 + scores.argmax(dim=-1)
        allowed = (self.home_candidates - 1).to(dtype=torch.long)
        allowed_scores = scores.index_select(-1, allowed)
        return self.home_candidates[allowed_scores.argmax(dim=-1)]

    def generate(
        self,
        batch_size: int,
        seed: int,
        writers_per_episode: int | None = None,
        active_compute_nodes: int | None = None,
        bootstrap_mode: bool = False,
        topology: GrowthTopology | None = None,
    ) -> MemoryBatch:
        cfg = self.config
        writers = writers_per_episode or cfg.task.writers_per_episode
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        writer_keys = torch.randn(batch_size, writers, cfg.model.key_dim, generator=generator)
        writer_labels = torch.randint(0, cfg.model.num_classes, (batch_size, writers), generator=generator)
        writer_home_nodes = self._sample_home_nodes(writer_keys)
        writer_start_nodes = sample_start_nodes(
            active_nodes=cfg.model.num_compute_nodes,
            pool_size=cfg.task.start_node_pool_size,
            shape=(batch_size, writers),
            generator=generator,
        )

        batch_indices = torch.arange(batch_size)
        query_writer_index = torch.randint(0, writers, (batch_size,), generator=generator)
        query_keys = writer_keys[batch_indices, query_writer_index].clone()
        query_home_nodes = writer_home_nodes[batch_indices, query_writer_index].clone()
        query_labels = writer_labels[batch_indices, query_writer_index].clone()
        query_start_nodes = sample_start_nodes(
            active_nodes=cfg.model.num_compute_nodes,
            pool_size=cfg.task.start_node_pool_size,
            shape=(batch_size,),
            generator=generator,
        )
        query_ttl = torch.randint(
            cfg.task.query_ttl_min,
            cfg.task.query_ttl_max + 1,
            (batch_size,),
            generator=generator,
        )
        query_required_delay = None
        delay_mode = str(cfg.task.delay_mode or "none")
        if delay_mode == "required_wait":
            query_required_delay = torch.randint(
                cfg.task.required_delay_min,
                cfg.task.required_delay_max + 1,
                (batch_size,),
                generator=generator,
            )
        elif delay_mode == "key_hash_exact_wait":
            query_required_delay = delay_targets_from_key(
                query_keys,
                min_delay=cfg.task.required_delay_min,
                max_delay=cfg.task.required_delay_max,
                hash_bits=cfg.task.required_delay_hash_bits,
            )

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
            query_required_delay=query_required_delay,
        )


class GrowthMemoryRoutingTask:
    def __init__(self, config: ExperimentConfig) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.task.hash_seed)
        self.config = config
        self.final_compute_nodes = config.model.num_compute_nodes
        self.home_hash = torch.randn(
            config.model.key_dim,
            self.final_compute_nodes,
            generator=generator,
        )
        self.home_candidates = MemoryRoutingTask._home_candidates(
            num_compute_nodes=self.final_compute_nodes,
            pool_size=config.task.home_node_pool_size,
        )

    def _sample_home_leaf(self, writer_keys: Tensor) -> Tensor:
        scores = writer_keys @ self.home_hash
        if int(self.home_candidates.numel()) == self.final_compute_nodes:
            return 1 + scores.argmax(dim=-1)
        allowed = (self.home_candidates - 1).to(dtype=torch.long)
        allowed_scores = scores.index_select(-1, allowed)
        return self.home_candidates[allowed_scores.argmax(dim=-1)]

    def generate(
        self,
        batch_size: int,
        seed: int,
        writers_per_episode: int | None = None,
        active_compute_nodes: int | None = None,
        bootstrap_mode: bool = False,
        topology: GrowthTopology | None = None,
    ) -> MemoryBatch:
        cfg = self.config
        writers = writers_per_episode or cfg.task.writers_per_episode
        active_nodes = active_compute_nodes or self.final_compute_nodes
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        active_node_tensor = active_node_ids_from_topology(topology) if topology is not None else active_node_ids(active_nodes)

        writer_keys = torch.randn(batch_size, writers, cfg.model.key_dim, generator=generator)
        writer_labels = torch.randint(0, cfg.model.num_classes, (batch_size, writers), generator=generator)
        writer_home_leaf = self._sample_home_leaf(writer_keys)
        if topology is None:
            writer_home_nodes = project_home_leaves(writer_home_leaf, active_nodes, self.final_compute_nodes)
        else:
            writer_home_nodes = project_home_leaves_topology(writer_home_leaf, topology)
        writer_start_nodes = sample_start_nodes(
            active_nodes=active_nodes,
            pool_size=cfg.task.start_node_pool_size,
            shape=(batch_size, writers),
            generator=generator,
            active_node_ids_tensor=active_node_tensor,
        )

        batch_indices = torch.arange(batch_size)
        query_writer_index = torch.randint(0, writers, (batch_size,), generator=generator)
        query_keys = writer_keys[batch_indices, query_writer_index].clone()
        query_home_leaf = writer_home_leaf[batch_indices, query_writer_index].clone()
        if topology is None:
            query_home_nodes = project_home_leaves(query_home_leaf, active_nodes, self.final_compute_nodes)
        else:
            query_home_nodes = project_home_leaves_topology(query_home_leaf, topology)
        query_labels = writer_labels[batch_indices, query_writer_index].clone()
        query_start_nodes = sample_start_nodes(
            active_nodes=active_nodes,
            pool_size=cfg.task.start_node_pool_size,
            shape=(batch_size,),
            generator=generator,
            active_node_ids_tensor=active_node_tensor,
        )
        query_ttl = torch.randint(
            cfg.task.query_ttl_min,
            cfg.task.query_ttl_max + 1,
            (batch_size,),
            generator=generator,
        )
        query_required_delay = None
        delay_mode = str(cfg.task.delay_mode or "none")
        if delay_mode == "required_wait":
            query_required_delay = torch.randint(
                cfg.task.required_delay_min,
                cfg.task.required_delay_max + 1,
                (batch_size,),
                generator=generator,
            )
        elif delay_mode == "key_hash_exact_wait":
            query_required_delay = delay_targets_from_key(
                query_keys,
                min_delay=cfg.task.required_delay_min,
                max_delay=cfg.task.required_delay_max,
                hash_bits=cfg.task.required_delay_hash_bits,
            )

        bootstrap_start_nodes = None
        bootstrap_ttl = None
        if bootstrap_mode:
            bootstrap_start_nodes = active_node_tensor.unsqueeze(0).repeat(batch_size, 1)
            bootstrap_ttl = torch.full_like(bootstrap_start_nodes, fill_value=active_nodes)

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
            query_required_delay=query_required_delay,
            bootstrap_start_nodes=bootstrap_start_nodes,
            bootstrap_ttl=bootstrap_ttl,
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
