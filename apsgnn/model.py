from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from apsgnn.buffer import NodeCache, TemporalRingBuffer
from apsgnn.config import ExperimentConfig
from apsgnn.growth import clockwise_successor
from apsgnn.node import ComputeNodeCell
from apsgnn.routing import build_address_table, route_from_address, sample_delay, straight_through_sample
from apsgnn.tasks import MemoryBatch, SanityBatch


ROLE_WRITER = 0
ROLE_QUERY = 1
ROLE_SANITY = 2


@dataclass
class PacketBatch:
    residual: Tensor
    routing_key: Tensor
    ttl: Tensor
    batch_index: Tensor
    current_node: Tensor
    role: Tensor
    target_label: Tensor
    target_home: Tensor
    hop_index: Tensor
    has_visited_home: Tensor
    packet_id: Tensor

    def __len__(self) -> int:
        return int(self.ttl.numel())

    def select(self, mask: Tensor) -> "PacketBatch":
        return PacketBatch(
            residual=self.residual[mask],
            routing_key=self.routing_key[mask],
            ttl=self.ttl[mask],
            batch_index=self.batch_index[mask],
            current_node=self.current_node[mask],
            role=self.role[mask],
            target_label=self.target_label[mask],
            target_home=self.target_home[mask],
            hop_index=self.hop_index[mask],
            has_visited_home=self.has_visited_home[mask],
            packet_id=self.packet_id[mask],
        )


@dataclass
class CacheWriteEvent:
    residual: Tensor
    batch_index: Tensor
    node_index: Tensor

    def select(self, mask: Tensor) -> "CacheWriteEvent":
        return CacheWriteEvent(
            residual=self.residual[mask],
            batch_index=self.batch_index[mask],
            node_index=self.node_index[mask],
        )


@dataclass
class OutputEvent:
    residual: Tensor
    batch_index: Tensor
    role: Tensor
    target_label: Tensor
    hop_index: Tensor
    packet_id: Tensor

    def select(self, mask: Tensor) -> "OutputEvent":
        return OutputEvent(
            residual=self.residual[mask],
            batch_index=self.batch_index[mask],
            role=self.role[mask],
            target_label=self.target_label[mask],
            hop_index=self.hop_index[mask],
            packet_id=self.packet_id[mask],
        )


def _concat_packets(batches: list[PacketBatch]) -> PacketBatch | None:
    if not batches:
        return None
    return PacketBatch(
        residual=torch.cat([batch.residual for batch in batches], dim=0),
        routing_key=torch.cat([batch.routing_key for batch in batches], dim=0),
        ttl=torch.cat([batch.ttl for batch in batches], dim=0),
        batch_index=torch.cat([batch.batch_index for batch in batches], dim=0),
        current_node=torch.cat([batch.current_node for batch in batches], dim=0),
        role=torch.cat([batch.role for batch in batches], dim=0),
        target_label=torch.cat([batch.target_label for batch in batches], dim=0),
        target_home=torch.cat([batch.target_home for batch in batches], dim=0),
        hop_index=torch.cat([batch.hop_index for batch in batches], dim=0),
        has_visited_home=torch.cat([batch.has_visited_home for batch in batches], dim=0),
        packet_id=torch.cat([batch.packet_id for batch in batches], dim=0),
    )


def _concat_cache_events(events: list[CacheWriteEvent]) -> CacheWriteEvent | None:
    if not events:
        return None
    return CacheWriteEvent(
        residual=torch.cat([event.residual for event in events], dim=0),
        batch_index=torch.cat([event.batch_index for event in events], dim=0),
        node_index=torch.cat([event.node_index for event in events], dim=0),
    )


def _concat_output_events(events: list[OutputEvent]) -> OutputEvent | None:
    if not events:
        return None
    return OutputEvent(
        residual=torch.cat([event.residual for event in events], dim=0),
        batch_index=torch.cat([event.batch_index for event in events], dim=0),
        role=torch.cat([event.role for event in events], dim=0),
        target_label=torch.cat([event.target_label for event in events], dim=0),
        hop_index=torch.cat([event.hop_index for event in events], dim=0),
        packet_id=torch.cat([event.packet_id for event in events], dim=0),
    )


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, layers: int) -> nn.Sequential:
    modules: list[nn.Module] = []
    current_dim = input_dim
    total_layers = max(layers, 1)
    for layer_idx in range(total_layers):
        next_dim = output_dim if layer_idx == total_layers - 1 else hidden_dim
        modules.append(nn.Linear(current_dim, next_dim))
        if layer_idx != total_layers - 1:
            modules.append(nn.GELU())
            modules.append(nn.LayerNorm(next_dim))
        current_dim = next_dim
    return nn.Sequential(*modules)


class KeyCentricFirstHopRouter(nn.Module):
    def __init__(
        self,
        *,
        key_dim: int,
        d_model: int,
        hidden_dim: int,
        layers: int,
        num_compute_nodes: int,
        address_dim: int,
        separate_heads: bool,
        use_residual: bool,
        residual_scale: float,
        aux_type: str,
    ) -> None:
        super().__init__()
        self.separate_heads = separate_heads
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        self.aux_type = aux_type

        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.aux_proj = nn.Linear(d_model, hidden_dim)
        self.residual_proj = nn.Linear(d_model, hidden_dim) if use_residual else None
        self.input_ln = nn.LayerNorm(hidden_dim)

        backbone_layers: list[nn.Module] = []
        for _ in range(max(layers, 1)):
            backbone_layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim),
                ]
            )
        self.backbone = nn.Sequential(*backbone_layers)

        if separate_heads:
            self.writer_head = nn.Linear(hidden_dim, num_compute_nodes)
            self.query_head = nn.Linear(hidden_dim, num_compute_nodes)
        else:
            self.shared_head = nn.Linear(hidden_dim, num_compute_nodes)

        self.address_head = nn.Linear(hidden_dim, address_dim) if aux_type == "address_l2" else None

    def forward(
        self,
        *,
        routing_key: Tensor,
        aux_features: Tensor,
        residual: Tensor,
        role: Tensor,
    ) -> dict[str, Tensor]:
        hidden = self.key_proj(routing_key) + self.aux_proj(aux_features)
        if self.residual_proj is not None:
            hidden = hidden + self.residual_scale * self.residual_proj(residual)
        hidden = self.backbone(self.input_ln(hidden))

        if self.separate_heads:
            writer_mask = role == ROLE_WRITER
            query_mask = role == ROLE_QUERY
            writer_logits = self.writer_head(hidden[writer_mask]) if writer_mask.any() else None
            query_logits = self.query_head(hidden[query_mask]) if query_mask.any() else None
            fallback_dtype = (
                writer_logits.dtype
                if writer_logits is not None
                else query_logits.dtype
                if query_logits is not None
                else hidden.dtype
            )
            logits = torch.empty(
                hidden.size(0),
                self.writer_head.out_features,
                device=hidden.device,
                dtype=fallback_dtype,
            )
            if writer_mask.any():
                logits[writer_mask] = writer_logits
            if query_mask.any():
                logits[query_mask] = query_logits
            other_mask = ~(writer_mask | query_mask)
            if other_mask.any():
                logits[other_mask] = self.query_head(hidden[other_mask])
        else:
            logits = self.shared_head(hidden)

        outputs = {
            "logits": logits,
            "hidden": hidden,
        }
        if self.address_head is not None:
            outputs["aux_address"] = self.address_head(hidden)
        return outputs


class LearnedCacheRetriever(nn.Module):
    def __init__(
        self,
        *,
        variant: str,
        d_model: int,
        key_dim: int,
        hidden_dim: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.variant = variant
        self.d_model = d_model
        self.uses_key_conditioning = variant == "learned_keycond"

        if self.uses_key_conditioning:
            self.key_condition_proj = nn.Linear(key_dim, d_model)
            query_input_dim = d_model * 2
        else:
            self.key_condition_proj = None
            query_input_dim = d_model
        self.query_mlp = _build_mlp(query_input_dim, hidden_dim, hidden_dim, layers)
        self.cache_key_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden_dim),
        )
        self.cache_value_proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )
        self.merge_proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.merge_gate = nn.Linear(d_model, d_model)

    def build_query_inputs(self, hidden: Tensor, routing_key: Tensor) -> Tensor:
        if not self.uses_key_conditioning:
            return hidden
        assert self.key_condition_proj is not None
        key_features = F.gelu(self.key_condition_proj(routing_key))
        return torch.cat([hidden, key_features], dim=-1)

    def forward(
        self,
        *,
        hidden: Tensor,
        routing_key: Tensor,
        cache_entries: Tensor,
        cache_mask: Tensor,
    ) -> dict[str, Tensor]:
        if hidden.numel() == 0:
            zeros = hidden.new_zeros(hidden.size(0), self.d_model)
            empty_weights = hidden.new_zeros(hidden.size(0), cache_entries.size(1))
            return {
                "update": zeros,
                "attention_weights": empty_weights,
                "entropy": hidden.new_zeros(hidden.size(0)),
                "top_mass": hidden.new_zeros(hidden.size(0)),
                "entry_count": hidden.new_zeros(hidden.size(0)),
            }

        query = self.query_mlp(self.build_query_inputs(hidden, routing_key))
        cache_keys = self.cache_key_proj(cache_entries)
        cache_values = self.cache_value_proj(cache_entries)

        scores = torch.einsum("gd,gcd->gc", query, cache_keys) / math.sqrt(cache_keys.size(-1))
        scores = scores.masked_fill(~cache_mask, -1.0e9)
        attention_weights = scores.softmax(dim=-1)
        attention_weights = attention_weights * cache_mask.to(attention_weights.dtype)
        attention_norm = attention_weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        attention_weights = attention_weights / attention_norm

        retrieved = torch.einsum("gc,gcd->gd", attention_weights, cache_values)
        merged = self.merge_proj(torch.cat([hidden, retrieved], dim=-1))
        gated_update = torch.sigmoid(self.merge_gate(hidden)) * merged

        safe_weights = attention_weights.clamp_min(1.0e-8)
        entropy = -(attention_weights * safe_weights.log()).sum(dim=-1)
        top_mass = attention_weights.max(dim=-1).values
        entry_count = cache_mask.sum(dim=-1).to(hidden.dtype)
        return {
            "update": gated_update,
            "attention_weights": attention_weights,
            "entropy": entropy,
            "top_mass": top_mass,
            "entry_count": entry_count,
        }


class APSGNNModel(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        model_cfg = config.model

        address_table = build_address_table(model_cfg.nodes_total, model_cfg.address_dim)
        self.register_buffer("address_table", address_table, persistent=True)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(config.task.hash_seed)
        home_hash = torch.randn(config.model.key_dim, config.model.num_compute_nodes, generator=generator)
        self.register_buffer("home_hash", home_hash, persistent=True)

        self.key_proj = nn.Linear(model_cfg.key_dim, model_cfg.d_model)
        self.class_embed = nn.Embedding(model_cfg.num_classes, model_cfg.d_model)
        self.role_embed = nn.Embedding(model_cfg.packet_roles, model_cfg.d_model)
        self.start_node_embed = nn.Embedding(model_cfg.nodes_total, model_cfg.d_model)
        self.ttl_embed = nn.Embedding(model_cfg.max_ttl + 1, model_cfg.d_model)
        self.sanity_proj = nn.Linear(model_cfg.d_model, model_cfg.d_model)
        self.input_ln = nn.LayerNorm(model_cfg.d_model)
        self.first_hop_router_ln = nn.LayerNorm(model_cfg.d_model)
        if model_cfg.first_hop_router_variant == "legacy":
            self.first_hop_router = nn.Sequential(
                nn.Linear(model_cfg.d_model, model_cfg.d_model),
                nn.GELU(),
                nn.Linear(model_cfg.d_model, model_cfg.num_compute_nodes),
            )
        else:
            self.first_hop_router = KeyCentricFirstHopRouter(
                key_dim=model_cfg.key_dim,
                d_model=model_cfg.d_model,
                hidden_dim=model_cfg.first_hop_router_hidden_dim,
                layers=model_cfg.first_hop_router_layers,
                num_compute_nodes=model_cfg.num_compute_nodes,
                address_dim=model_cfg.address_dim,
                separate_heads=model_cfg.first_hop_router_separate_heads,
                use_residual=model_cfg.first_hop_router_use_residual,
                residual_scale=model_cfg.first_hop_router_residual_scale,
                aux_type=model_cfg.first_hop_router_aux_type,
            )
        if model_cfg.cache_read_variant == "explicit":
            self.cache_retriever: LearnedCacheRetriever | None = None
        else:
            self.cache_retriever = LearnedCacheRetriever(
                variant=model_cfg.cache_read_variant,
                d_model=model_cfg.d_model,
                key_dim=model_cfg.key_dim,
                hidden_dim=model_cfg.cache_read_hidden_dim,
                layers=model_cfg.cache_read_layers,
            )
        self.first_hop_teacher_force_ratio = 0.0
        self.active_compute_nodes = model_cfg.num_compute_nodes
        self.bootstrap_active = False
        self.active_node_ids_tensor: Tensor | None = None
        self.clockwise_successor_lookup_tensor: Tensor | None = None

        self.node_cells = nn.ModuleList(
            [
                ComputeNodeCell(
                    d_model=model_cfg.d_model,
                    nhead=model_cfg.nhead,
                    delay_bins=model_cfg.delay_bins,
                    mlp_ratio=model_cfg.mlp_ratio,
                    dropout=model_cfg.dropout,
                )
                for _ in range(model_cfg.num_compute_nodes)
            ]
        )
        self.delta_head = nn.Linear(model_cfg.d_model, model_cfg.d_model)
        self.direction_head = nn.Linear(model_cfg.d_model, model_cfg.d_model)
        self.magnitude_head = nn.Linear(model_cfg.d_model, 1)
        self.delay_head = nn.Linear(model_cfg.d_model, model_cfg.delay_bins)
        self.output_head = nn.Linear(model_cfg.d_model, model_cfg.num_classes)

        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        nn.init.constant_(self.magnitude_head.bias, 2.5)

    def set_first_hop_teacher_force_ratio(self, ratio: float) -> None:
        self.first_hop_teacher_force_ratio = float(min(max(ratio, 0.0), 1.0))

    def set_growth_context(
        self,
        *,
        active_compute_nodes: int | None = None,
        bootstrap_active: bool = False,
        active_node_ids: Tensor | None = None,
        clockwise_successor_lookup: Tensor | None = None,
    ) -> None:
        if active_compute_nodes is None:
            self.active_compute_nodes = self.config.model.num_compute_nodes
        else:
            self.active_compute_nodes = int(active_compute_nodes)
        self.bootstrap_active = bool(bootstrap_active)
        self.active_node_ids_tensor = None if active_node_ids is None else active_node_ids.detach().to(dtype=torch.long)
        self.clockwise_successor_lookup_tensor = (
            None if clockwise_successor_lookup is None else clockwise_successor_lookup.detach().to(dtype=torch.long)
        )

    def uses_legacy_first_hop_router(self) -> bool:
        return self.config.model.first_hop_router_variant == "legacy"

    def uses_first_hop_router_aux(self) -> bool:
        return (
            self.config.model.use_learned_first_hop_router
            and self.config.model.first_hop_router_variant != "legacy"
            and self.config.model.first_hop_router_aux_type == "address_l2"
            and self.config.train.first_hop_router_aux_weight > 0.0
        )

    def uses_explicit_cache_read(self) -> bool:
        return self.config.model.cache_read_variant == "explicit"

    def uses_learned_cache_read(self) -> bool:
        return self.config.model.cache_read_variant in {"learned_implicit", "learned_keycond"}

    def uses_growth_curriculum(self) -> bool:
        return self.config.growth.enabled

    def _active_node_mask(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        mask = torch.zeros(self.config.model.nodes_total, device=device, dtype=dtype)
        mask[0] = 1.0
        if self.active_node_ids_tensor is None:
            mask[1 : self.active_compute_nodes + 1] = 1.0
        else:
            mask[self.active_node_ids_tensor.to(device=device)] = 1.0
        return mask

    def _clockwise_target(self, current_node: Tensor) -> Tensor:
        if self.clockwise_successor_lookup_tensor is not None:
            lookup = self.clockwise_successor_lookup_tensor.to(device=current_node.device)
            successor = lookup[current_node]
            fallback_mask = successor == 0
            if fallback_mask.any():
                fallback = clockwise_successor(current_node[fallback_mask], self.active_compute_nodes)
                successor = successor.clone()
                successor[fallback_mask] = fallback
            return successor
        return clockwise_successor(current_node, self.active_compute_nodes)

    def _coverage_packet_residual(
        self,
        *,
        start_nodes: Tensor,
        ttl: Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        residual = (
            self.role_embed.weight[ROLE_SANITY].to(dtype)
            + self.start_node_embed(start_nodes)
            + self.ttl_embed(ttl.clamp(min=0, max=self.config.model.max_ttl))
        )
        return self.input_ln(residual)

    def _register_gradient_probe(
        self,
        tensor: Tensor,
        *,
        packet_mask: Tensor,
        role_tensor: Tensor,
        node_index: int,
        gradient_buffers: dict[str, Tensor] | None,
    ) -> None:
        if gradient_buffers is None or not tensor.requires_grad:
            return

        node_slot = node_index - 1
        packet_mask_detached = packet_mask.detach()
        role_tensor_detached = role_tensor.detach()

        def _hook(grad: Tensor) -> None:
            active_grad = grad.detach()[packet_mask_detached]
            if active_grad.numel() == 0:
                return
            active_roles = role_tensor_detached[packet_mask_detached]
            per_packet_grad = active_grad.float().norm(dim=-1)
            with torch.no_grad():
                gradient_buffers["all"][node_slot] += per_packet_grad.sum()
                task_mask = active_roles != ROLE_SANITY
                if task_mask.any():
                    gradient_buffers["task"][node_slot] += per_packet_grad[task_mask].sum()
                query_mask = active_roles == ROLE_QUERY
                if query_mask.any():
                    gradient_buffers["query"][node_slot] += per_packet_grad[query_mask].sum()
                bootstrap_mask = active_roles == ROLE_SANITY
                if bootstrap_mask.any():
                    gradient_buffers["bootstrap"][node_slot] += per_packet_grad[bootstrap_mask].sum()

        tensor.register_hook(_hook)

    def encode_writers(self, batch: MemoryBatch) -> Tensor:
        writers = batch.writer_keys.size(1)
        role = self.role_embed.weight[ROLE_WRITER]
        learned = (
            self.key_proj(batch.writer_keys.reshape(-1, self.config.model.key_dim))
            + self.class_embed(batch.writer_labels.reshape(-1))
            + self.start_node_embed(batch.writer_start_nodes.reshape(-1))
            + role
        )
        structured = torch.zeros_like(learned)
        flat_keys = batch.writer_keys.reshape(-1, self.config.model.key_dim)
        flat_labels = batch.writer_labels.reshape(-1)
        structured[:, : self.config.model.key_dim] = flat_keys
        structured[:, self.config.model.key_dim : self.config.model.key_dim + self.config.model.num_classes] = F.one_hot(
            flat_labels,
            num_classes=self.config.model.num_classes,
        ).to(learned.dtype)
        encoded = self.input_ln(learned) + structured
        return encoded.view(batch.batch_size * writers, self.config.model.d_model)

    def encode_queries(self, batch: MemoryBatch) -> Tensor:
        role = self.role_embed.weight[ROLE_QUERY]
        learned = (
            self.key_proj(batch.query_keys)
            + self.start_node_embed(batch.query_start_nodes)
            + role
        )
        structured = torch.zeros_like(learned)
        structured[:, : self.config.model.key_dim] = batch.query_keys
        return self.input_ln(learned) + structured

    def encode_sanity(self, batch: SanityBatch) -> Tensor:
        role = self.role_embed.weight[ROLE_SANITY]
        encoded = self.sanity_proj(batch.inputs) + self.start_node_embed(batch.start_nodes) + role
        return self.input_ln(encoded)

    def forward(self, batch: MemoryBatch | SanityBatch) -> dict[str, object]:
        if isinstance(batch, MemoryBatch):
            return self.forward_memory(batch)
        return self.forward_sanity(batch)

    def forward_memory(self, batch: MemoryBatch) -> dict[str, object]:
        device = batch.query_keys.device
        dtype = batch.query_keys.dtype
        cfg = self.config
        zero = torch.zeros((), device=device, dtype=dtype)

        batch_size = batch.batch_size
        writers_per_episode = batch.writer_keys.size(1)
        ring_size = cfg.model.delay_bins
        cache = NodeCache(
            batch_size=batch_size,
            nodes_total=cfg.model.nodes_total,
            capacity=cfg.model.cache_capacity,
            d_model=cfg.model.d_model,
            device=device,
            dtype=dtype,
            enabled=cfg.model.enable_cache,
        )
        live_buffer: TemporalRingBuffer[PacketBatch] = TemporalRingBuffer(ring_size)
        cache_buffer: TemporalRingBuffer[CacheWriteEvent] = TemporalRingBuffer(ring_size)
        output_buffer: TemporalRingBuffer[OutputEvent] = TemporalRingBuffer(ring_size)

        writer_residual = self.encode_writers(batch)
        writer_batch_index = torch.arange(batch_size, device=device).repeat_interleave(writers_per_episode)
        writer_target_label = batch.writer_labels.reshape(-1)
        writer_target_home = batch.writer_home_nodes.reshape(-1)
        writer_start_nodes = batch.writer_start_nodes.reshape(-1)
        writer_ttl = torch.ones_like(writer_target_label)
        writer_packets = PacketBatch(
            residual=writer_residual,
            routing_key=batch.writer_keys.reshape(-1, cfg.model.key_dim),
            ttl=writer_ttl,
            batch_index=writer_batch_index,
            current_node=writer_start_nodes,
            role=torch.full_like(writer_target_label, ROLE_WRITER),
            target_label=writer_target_label,
            target_home=writer_target_home,
            hop_index=torch.zeros_like(writer_target_label),
            has_visited_home=torch.zeros_like(writer_target_label, dtype=torch.bool),
            packet_id=torch.arange(batch_size * writers_per_episode, device=device),
        )
        live_buffer.schedule(cfg.task.writer_inject_step, writer_packets)

        query_residual = self.encode_queries(batch)
        query_packets = PacketBatch(
            residual=query_residual,
            routing_key=batch.query_keys,
            ttl=batch.query_ttl,
            batch_index=torch.arange(batch_size, device=device),
            current_node=batch.query_start_nodes,
            role=torch.full((batch_size,), ROLE_QUERY, device=device, dtype=torch.long),
            target_label=batch.query_labels,
            target_home=batch.query_home_nodes,
            hop_index=torch.zeros(batch_size, device=device, dtype=torch.long),
            has_visited_home=torch.zeros(batch_size, device=device, dtype=torch.bool),
            packet_id=torch.arange(batch_size, device=device) + batch_size * writers_per_episode,
        )
        live_buffer.schedule(cfg.task.query_inject_step, query_packets)

        if batch.bootstrap_start_nodes is not None and batch.bootstrap_ttl is not None:
            coverage_start_nodes = batch.bootstrap_start_nodes.reshape(-1)
            coverage_ttl = batch.bootstrap_ttl.reshape(-1)
            coverage_count = coverage_start_nodes.numel()
            coverage_batch_index = (
                torch.arange(batch_size, device=device)
                .unsqueeze(1)
                .expand(-1, batch.bootstrap_start_nodes.size(1))
                .reshape(-1)
            )
            coverage_packets = PacketBatch(
                residual=self._coverage_packet_residual(
                    start_nodes=coverage_start_nodes,
                    ttl=coverage_ttl,
                    device=device,
                    dtype=dtype,
                ),
                routing_key=torch.zeros(coverage_count, cfg.model.key_dim, device=device, dtype=dtype),
                ttl=coverage_ttl,
                batch_index=coverage_batch_index,
                current_node=coverage_start_nodes,
                role=torch.full((coverage_count,), ROLE_SANITY, device=device, dtype=torch.long),
                target_label=torch.zeros(coverage_count, device=device, dtype=torch.long),
                target_home=torch.zeros(coverage_count, device=device, dtype=torch.long),
                hop_index=torch.zeros(coverage_count, device=device, dtype=torch.long),
                has_visited_home=torch.zeros(coverage_count, device=device, dtype=torch.bool),
                packet_id=torch.arange(coverage_count, device=device, dtype=torch.long)
                + batch_size * (writers_per_episode + 1),
            )
            live_buffer.schedule(cfg.task.writer_inject_step, coverage_packets)

        first_query_logits: list[Tensor | None] = [None for _ in range(batch_size)]
        query_delivered = torch.zeros(batch_size, device=device, dtype=torch.bool)
        query_last_hops = torch.zeros(batch_size, device=device, dtype=torch.long)

        loss_main_sum = zero
        loss_writer_sum = zero
        loss_query_sum = zero
        loss_first_hop_aux_sum = zero
        loss_home_out_sum = zero
        loss_delay_sum = zero
        loss_gravity_sum = zero
        loss_missing_sum = zero
        loss_bootstrap_route_sum = zero
        loss_bootstrap_delay_sum = zero

        writer_hit_sum = zero
        writer_hit_count = zero
        query_hit_sum = zero
        query_hit_count = zero
        home_out_hit_sum = zero
        home_out_count = zero
        teacher_force_sum = zero
        teacher_force_count = zero
        delay_sum = zero
        delay_count = zero
        processed_packets_sum = zero
        cache_mean_sum = zero
        cache_mean_count = zero
        cache_max_sum = zero
        cache_max_count = zero
        retrieval_entropy_sum = zero
        retrieval_entropy_count = zero
        retrieval_top_mass_sum = zero
        retrieval_top_mass_count = zero
        retrieval_entry_sum = zero
        retrieval_entry_count = zero
        all_visit_counts_sum = torch.zeros(cfg.model.num_compute_nodes, device=device, dtype=dtype)
        task_visit_counts_sum = torch.zeros_like(all_visit_counts_sum)
        query_visit_counts_sum = torch.zeros_like(all_visit_counts_sum)
        bootstrap_visit_counts_sum = torch.zeros_like(all_visit_counts_sum)
        gradient_buffers = {
            "all": torch.zeros(cfg.model.num_compute_nodes, device=device, dtype=torch.float32),
            "task": torch.zeros(cfg.model.num_compute_nodes, device=device, dtype=torch.float32),
            "query": torch.zeros(cfg.model.num_compute_nodes, device=device, dtype=torch.float32),
            "bootstrap": torch.zeros(cfg.model.num_compute_nodes, device=device, dtype=torch.float32),
        }

        for step in range(cfg.task.max_rollout_steps):
            cache_events = _concat_cache_events(cache_buffer.pop_current())
            if cache_events is not None:
                cache.write(cache_events.residual, cache_events.batch_index, cache_events.node_index)

            output_events = _concat_output_events(output_buffer.pop_current())
            if output_events is not None and len(output_events.batch_index) > 0:
                query_mask = output_events.role == ROLE_QUERY
                if query_mask.any():
                    logits = self._output_logits(output_events.residual[query_mask])
                    batch_index = output_events.batch_index[query_mask]
                    labels = output_events.target_label[query_mask]
                    hops = output_events.hop_index[query_mask]
                    for i in range(logits.size(0)):
                        sample = int(batch_index[i].item())
                        if first_query_logits[sample] is None:
                            first_query_logits[sample] = logits[i]
                            query_delivered[sample] = True
                            query_last_hops[sample] = hops[i]

            cache_mean, cache_max = cache.occupancy_stats()
            cache_mean_sum = cache_mean_sum + cache_mean
            cache_mean_count = cache_mean_count + 1.0
            cache_max_sum = cache_max_sum + cache_max
            cache_max_count = cache_max_count + 1.0

            active_packets = _concat_packets(live_buffer.pop_current())
            if active_packets is None:
                live_buffer.advance()
                cache_buffer.advance()
                output_buffer.advance()
                continue

            processed_packets_sum = processed_packets_sum + float(len(active_packets))

            predictions = self._run_compute_nodes(active_packets, cache, gradient_buffers=gradient_buffers)
            route_logits = predictions["route_logits"]
            delay_logits = predictions["delay_logits"]
            dest_index = predictions["dest_index"]
            predicted_dest_index = predictions["predicted_dest_index"]
            delay_index = predictions["delay_index"]
            next_residual = predictions["next_residual"]
            predicted_address = predictions["predicted_address"]
            first_hop_aux_prediction = predictions["first_hop_aux_prediction"]
            address_norm = predictions["address_norm"]
            teacher_forced_mask = predictions["teacher_forced_mask"]
            retrieval_entropy = predictions["retrieval_entropy"]
            retrieval_top_mass = predictions["retrieval_top_mass"]
            retrieval_entry_count_values = predictions["retrieval_entry_count"]
            clockwise_targets = predictions["clockwise_target"]
            all_visit_counts_sum = all_visit_counts_sum + predictions["visit_counts"].to(dtype)
            task_visit_counts_sum = task_visit_counts_sum + predictions["task_visit_counts"].to(dtype)
            query_visit_counts_sum = query_visit_counts_sum + predictions["query_visit_counts"].to(dtype)
            bootstrap_visit_counts_sum = bootstrap_visit_counts_sum + predictions["bootstrap_visit_counts"].to(dtype)

            writer_first_mask = (active_packets.role == ROLE_WRITER) & (active_packets.hop_index == 0)
            if writer_first_mask.any():
                writer_targets = active_packets.target_home[writer_first_mask]
                writer_logits = route_logits[writer_first_mask]
                loss_writer_sum = loss_writer_sum + F.cross_entropy(writer_logits, writer_targets, reduction="sum")
                if self.uses_legacy_first_hop_router() or not cfg.model.use_learned_first_hop_router:
                    loss_writer_sum = loss_writer_sum + cfg.train.address_aux_weight * F.mse_loss(
                        predicted_address[writer_first_mask],
                        self.address_table[writer_targets],
                        reduction="sum",
                    )
                elif self.uses_first_hop_router_aux():
                    loss_first_hop_aux_sum = loss_first_hop_aux_sum + cfg.train.first_hop_router_aux_weight * F.mse_loss(
                        first_hop_aux_prediction[writer_first_mask],
                        self.address_table[writer_targets],
                        reduction="sum",
                    )
                writer_hit_sum = writer_hit_sum + (predicted_dest_index[writer_first_mask] == writer_targets).sum()
                writer_hit_count = writer_hit_count + writer_targets.numel()
                teacher_force_sum = teacher_force_sum + teacher_forced_mask[writer_first_mask].sum()
                teacher_force_count = teacher_force_count + writer_targets.numel()
                zeros = torch.zeros(writer_targets.numel(), device=device, dtype=torch.long)
                loss_delay_sum = loss_delay_sum + F.cross_entropy(delay_logits[writer_first_mask], zeros, reduction="sum")

            query_first_mask = (active_packets.role == ROLE_QUERY) & (active_packets.hop_index == 0)
            if query_first_mask.any():
                query_targets = active_packets.target_home[query_first_mask]
                query_logits = route_logits[query_first_mask]
                loss_query_sum = loss_query_sum + F.cross_entropy(query_logits, query_targets, reduction="sum")
                if self.uses_legacy_first_hop_router() or not cfg.model.use_learned_first_hop_router:
                    loss_query_sum = loss_query_sum + cfg.train.address_aux_weight * F.mse_loss(
                        predicted_address[query_first_mask],
                        self.address_table[query_targets],
                        reduction="sum",
                    )
                elif self.uses_first_hop_router_aux():
                    loss_first_hop_aux_sum = loss_first_hop_aux_sum + cfg.train.first_hop_router_aux_weight * F.mse_loss(
                        first_hop_aux_prediction[query_first_mask],
                        self.address_table[query_targets],
                        reduction="sum",
                    )
                query_hit_sum = query_hit_sum + (predicted_dest_index[query_first_mask] == query_targets).sum()
                query_hit_count = query_hit_count + query_targets.numel()
                teacher_force_sum = teacher_force_sum + teacher_forced_mask[query_first_mask].sum()
                teacher_force_count = teacher_force_count + query_targets.numel()
                zeros = torch.zeros(query_targets.numel(), device=device, dtype=torch.long)
                loss_delay_sum = loss_delay_sum + F.cross_entropy(delay_logits[query_first_mask], zeros, reduction="sum")

            entered_home = (active_packets.role == ROLE_QUERY) & (active_packets.current_node == active_packets.target_home)
            first_home_mask = entered_home & (~active_packets.has_visited_home)
            if first_home_mask.any():
                loss_home_out_sum = loss_home_out_sum + F.cross_entropy(
                    route_logits[first_home_mask],
                    torch.zeros(first_home_mask.sum(), device=device, dtype=torch.long),
                    reduction="sum",
                )
                loss_home_out_sum = loss_home_out_sum + cfg.train.address_aux_weight * F.mse_loss(
                    predicted_address[first_home_mask],
                    torch.zeros_like(predicted_address[first_home_mask]),
                    reduction="sum",
                )
                home_out_hit_sum = home_out_hit_sum + (dest_index[first_home_mask] == 0).sum()
                home_out_count = home_out_count + first_home_mask.sum()
                zeros = torch.zeros(first_home_mask.sum(), device=device, dtype=torch.long)
                loss_delay_sum = loss_delay_sum + F.cross_entropy(delay_logits[first_home_mask], zeros, reduction="sum")
                retrieval_entropy_sum = retrieval_entropy_sum + retrieval_entropy[first_home_mask].sum()
                retrieval_entropy_count = retrieval_entropy_count + first_home_mask.sum()
                retrieval_top_mass_sum = retrieval_top_mass_sum + retrieval_top_mass[first_home_mask].sum()
                retrieval_top_mass_count = retrieval_top_mass_count + first_home_mask.sum()
                retrieval_entry_sum = retrieval_entry_sum + retrieval_entry_count_values[first_home_mask].sum()
                retrieval_entry_count = retrieval_entry_count + first_home_mask.sum()

            gravity_mask = (active_packets.role == ROLE_QUERY) & (active_packets.has_visited_home | entered_home)
            if gravity_mask.any():
                loss_gravity_sum = loss_gravity_sum + address_norm[gravity_mask].square().sum()

            bootstrap_mask = (active_packets.role == ROLE_SANITY) & torch.tensor(
                self.bootstrap_active,
                device=device,
                dtype=torch.bool,
            )
            if bootstrap_mask.any():
                loss_bootstrap_route_sum = loss_bootstrap_route_sum + F.cross_entropy(
                    route_logits[bootstrap_mask],
                    clockwise_targets[bootstrap_mask],
                    reduction="sum",
                )
                bootstrap_delay_targets = torch.zeros(bootstrap_mask.sum(), device=device, dtype=torch.long)
                loss_bootstrap_delay_sum = loss_bootstrap_delay_sum + F.cross_entropy(
                    delay_logits[bootstrap_mask],
                    bootstrap_delay_targets,
                    reduction="sum",
                )

            delay_sum = delay_sum + delay_index.sum()
            delay_count = delay_count + delay_index.numel()

            query_mask = active_packets.role == ROLE_QUERY
            if query_mask.any():
                query_last_hops[active_packets.batch_index[query_mask]] = active_packets.hop_index[query_mask] + 1

            updated_has_visited = active_packets.has_visited_home | entered_home
            ttl_after = active_packets.ttl - 1
            scheduled_step = step + 1 + delay_index

            output_mask = dest_index == 0
            cache_write_mask = (~output_mask) & (ttl_after == 0)
            live_mask = (~output_mask) & (ttl_after > 0)

            if output_mask.any():
                output_events_to_schedule = OutputEvent(
                    residual=next_residual[output_mask],
                    batch_index=active_packets.batch_index[output_mask],
                    role=active_packets.role[output_mask],
                    target_label=active_packets.target_label[output_mask],
                    hop_index=active_packets.hop_index[output_mask] + 1,
                    packet_id=active_packets.packet_id[output_mask],
                )
                self._schedule_output_events(output_buffer, scheduled_step[output_mask], output_events_to_schedule)

            if cache_write_mask.any():
                cache_events_to_schedule = CacheWriteEvent(
                    residual=next_residual[cache_write_mask],
                    batch_index=active_packets.batch_index[cache_write_mask],
                    node_index=dest_index[cache_write_mask],
                )
                self._schedule_cache_events(cache_buffer, scheduled_step[cache_write_mask], cache_events_to_schedule)

            if live_mask.any():
                next_packets = PacketBatch(
                    residual=next_residual[live_mask],
                    routing_key=active_packets.routing_key[live_mask],
                    ttl=ttl_after[live_mask],
                    batch_index=active_packets.batch_index[live_mask],
                    current_node=dest_index[live_mask],
                    role=active_packets.role[live_mask],
                    target_label=active_packets.target_label[live_mask],
                    target_home=active_packets.target_home[live_mask],
                    hop_index=active_packets.hop_index[live_mask] + 1,
                    has_visited_home=updated_has_visited[live_mask],
                    packet_id=active_packets.packet_id[live_mask],
                )
                self._schedule_packets(live_buffer, scheduled_step[live_mask], next_packets)

            live_buffer.advance()
            cache_buffer.advance()
            output_buffer.advance()

        delivered_mask = torch.tensor([item is not None for item in first_query_logits], device=device, dtype=torch.bool)
        if delivered_mask.any():
            logits = torch.stack([item for item in first_query_logits if item is not None], dim=0)
            labels = batch.query_labels[delivered_mask]
            loss_main_sum = F.cross_entropy(logits, labels, reduction="sum")
            query_correct = (logits.argmax(dim=-1) == labels).sum().to(dtype)
        else:
            query_correct = zero
        loss_missing_sum = (~query_delivered).to(dtype).sum() * cfg.train.missing_output_penalty
        success_ratio = query_correct / max(float(delivered_mask.sum().item()), 1.0)
        success_visit_counts_sum = query_visit_counts_sum * success_ratio

        total_loss_sum = (
            loss_main_sum
            + cfg.train.aux_writer_weight * loss_writer_sum
            + cfg.train.aux_query_weight * loss_query_sum
            + loss_first_hop_aux_sum
            + cfg.train.aux_home_out_weight * loss_home_out_sum
            + cfg.train.delay_reg_weight * loss_delay_sum
            + cfg.train.gravity_weight * loss_gravity_sum
            + cfg.growth.bootstrap_route_weight * loss_bootstrap_route_sum
            + cfg.growth.bootstrap_delay_weight * loss_bootstrap_delay_sum
            + loss_missing_sum
        )
        total_loss = total_loss_sum / batch_size

        return {
            "loss": total_loss,
            "metric_sums": {
                "loss_total": total_loss_sum.detach(),
                "loss_main": loss_main_sum.detach(),
                "loss_writer_route": loss_writer_sum.detach(),
                "loss_query_route": loss_query_sum.detach(),
                "loss_first_hop_aux": loss_first_hop_aux_sum.detach(),
                "loss_home_to_output": loss_home_out_sum.detach(),
                "loss_delay": loss_delay_sum.detach(),
                "loss_gravity": loss_gravity_sum.detach(),
                "loss_bootstrap_route": loss_bootstrap_route_sum.detach(),
                "loss_bootstrap_delay": loss_bootstrap_delay_sum.detach(),
                "loss_missing_output": loss_missing_sum.detach(),
                "query_accuracy_hit": query_correct.detach(),
                "query_accuracy_count": delivered_mask.sum().to(dtype),
                "query_delivery_hit": query_delivered.to(dtype).sum(),
                "query_delivery_count": torch.tensor(float(batch_size), device=device, dtype=dtype),
                "writer_home_hit": writer_hit_sum.detach().to(dtype),
                "writer_home_count": writer_hit_count.detach().to(dtype),
                "query_home_hit": query_hit_sum.detach().to(dtype),
                "query_home_count": query_hit_count.detach().to(dtype),
                "query_home_output_hit": home_out_hit_sum.detach().to(dtype),
                "query_home_output_count": home_out_count.detach().to(dtype),
                "first_hop_teacher_force_sum": teacher_force_sum.detach().to(dtype),
                "first_hop_teacher_force_count": teacher_force_count.detach().to(dtype),
                "avg_hops_sum": query_last_hops.to(dtype).sum(),
                "avg_hops_count": torch.tensor(float(batch_size), device=device, dtype=dtype),
                "delay_sum": delay_sum.detach().to(dtype),
                "delay_count": delay_count.detach().to(dtype),
                "cache_mean_sum": cache_mean_sum.detach(),
                "cache_mean_count": cache_mean_count.detach().to(dtype),
                "cache_max_sum": cache_max_sum.detach(),
                "cache_max_count": cache_max_count.detach().to(dtype),
                "retrieval_entropy_sum": retrieval_entropy_sum.detach(),
                "retrieval_entropy_count": retrieval_entropy_count.detach().to(dtype),
                "retrieval_top_mass_sum": retrieval_top_mass_sum.detach(),
                "retrieval_top_mass_count": retrieval_top_mass_count.detach().to(dtype),
                "retrieval_entry_sum": retrieval_entry_sum.detach(),
                "retrieval_entry_count": retrieval_entry_count.detach().to(dtype),
                "packets_processed_sum": processed_packets_sum.detach().to(dtype),
            },
            "diagnostics": {
                "visit_counts": all_visit_counts_sum.detach(),
                "all_visit_counts": all_visit_counts_sum.detach(),
                "task_visit_counts": task_visit_counts_sum.detach(),
                "query_visit_counts": query_visit_counts_sum.detach(),
                "bootstrap_visit_counts": bootstrap_visit_counts_sum.detach(),
                "success_visit_counts": success_visit_counts_sum.detach(),
                "all_gradient_signal": gradient_buffers["all"],
                "task_gradient_signal": gradient_buffers["task"],
                "query_gradient_signal": gradient_buffers["query"],
                "bootstrap_gradient_signal": gradient_buffers["bootstrap"],
            },
        }

    def forward_sanity(self, batch: SanityBatch) -> dict[str, object]:
        device = batch.inputs.device
        dtype = batch.inputs.dtype
        cfg = self.config
        zero = torch.zeros((), device=device, dtype=dtype)
        batch_size = batch.batch_size

        live_buffer: TemporalRingBuffer[PacketBatch] = TemporalRingBuffer(cfg.model.delay_bins)
        cache_buffer: TemporalRingBuffer[CacheWriteEvent] = TemporalRingBuffer(cfg.model.delay_bins)
        output_buffer: TemporalRingBuffer[OutputEvent] = TemporalRingBuffer(cfg.model.delay_bins)
        cache = NodeCache(
            batch_size=batch_size,
            nodes_total=cfg.model.nodes_total,
            capacity=cfg.model.cache_capacity,
            d_model=cfg.model.d_model,
            device=device,
            dtype=dtype,
            enabled=cfg.model.enable_cache,
        )

        sanity_packets = PacketBatch(
            residual=self.encode_sanity(batch),
            routing_key=torch.zeros(batch_size, cfg.model.key_dim, device=device, dtype=dtype),
            ttl=batch.ttl,
            batch_index=torch.arange(batch_size, device=device),
            current_node=batch.start_nodes,
            role=torch.full((batch_size,), ROLE_SANITY, device=device, dtype=torch.long),
            target_label=torch.zeros(batch_size, device=device, dtype=torch.long),
            target_home=torch.zeros(batch_size, device=device, dtype=torch.long),
            hop_index=torch.zeros(batch_size, device=device, dtype=torch.long),
            has_visited_home=torch.zeros(batch_size, device=device, dtype=torch.bool),
            packet_id=torch.arange(batch_size, device=device),
        )
        live_buffer.schedule(0, sanity_packets)

        delivered = torch.zeros(batch_size, device=device, dtype=torch.bool)
        hops = torch.zeros(batch_size, device=device, dtype=torch.long)
        route_loss_sum = zero
        delay_loss_sum = zero
        delay_sum = zero
        delay_count = zero
        processed_packets_sum = zero

        for step in range(cfg.task.max_rollout_steps):
            cache_events = _concat_cache_events(cache_buffer.pop_current())
            if cache_events is not None:
                cache.write(cache_events.residual, cache_events.batch_index, cache_events.node_index)
            output_events = _concat_output_events(output_buffer.pop_current())
            if output_events is not None and len(output_events.batch_index) > 0:
                delivered[output_events.batch_index] = True
                hops[output_events.batch_index] = output_events.hop_index

            active_packets = _concat_packets(live_buffer.pop_current())
            if active_packets is None:
                live_buffer.advance()
                cache_buffer.advance()
                output_buffer.advance()
                continue

            processed_packets_sum = processed_packets_sum + float(len(active_packets))
            predictions = self._run_compute_nodes(active_packets, cache)
            route_logits = predictions["route_logits"]
            delay_logits = predictions["delay_logits"]
            dest_index = predictions["dest_index"]
            delay_index = predictions["delay_index"]
            next_residual = predictions["next_residual"]

            output_targets = torch.zeros(route_logits.size(0), device=device, dtype=torch.long)
            route_loss_sum = route_loss_sum + F.cross_entropy(route_logits, output_targets, reduction="sum")
            delay_targets = torch.zeros(delay_logits.size(0), device=device, dtype=torch.long)
            delay_loss_sum = delay_loss_sum + F.cross_entropy(delay_logits, delay_targets, reduction="sum")
            delay_sum = delay_sum + delay_index.sum()
            delay_count = delay_count + delay_index.numel()
            hops[active_packets.batch_index] = active_packets.hop_index + 1

            ttl_after = active_packets.ttl - 1
            scheduled_step = step + 1 + delay_index
            output_mask = dest_index == 0
            cache_write_mask = (~output_mask) & (ttl_after == 0)
            live_mask = (~output_mask) & (ttl_after > 0)

            if output_mask.any():
                output_events_to_schedule = OutputEvent(
                    residual=next_residual[output_mask],
                    batch_index=active_packets.batch_index[output_mask],
                    role=active_packets.role[output_mask],
                    target_label=active_packets.target_label[output_mask],
                    hop_index=active_packets.hop_index[output_mask] + 1,
                    packet_id=active_packets.packet_id[output_mask],
                )
                self._schedule_output_events(output_buffer, scheduled_step[output_mask], output_events_to_schedule)

            if cache_write_mask.any():
                cache_events_to_schedule = CacheWriteEvent(
                    residual=next_residual[cache_write_mask],
                    batch_index=active_packets.batch_index[cache_write_mask],
                    node_index=dest_index[cache_write_mask],
                )
                self._schedule_cache_events(cache_buffer, scheduled_step[cache_write_mask], cache_events_to_schedule)

            if live_mask.any():
                next_packets = PacketBatch(
                    residual=next_residual[live_mask],
                    routing_key=active_packets.routing_key[live_mask],
                    ttl=ttl_after[live_mask],
                    batch_index=active_packets.batch_index[live_mask],
                    current_node=dest_index[live_mask],
                    role=active_packets.role[live_mask],
                    target_label=active_packets.target_label[live_mask],
                    target_home=active_packets.target_home[live_mask],
                    hop_index=active_packets.hop_index[live_mask] + 1,
                    has_visited_home=active_packets.has_visited_home[live_mask],
                    packet_id=active_packets.packet_id[live_mask],
                )
                self._schedule_packets(live_buffer, scheduled_step[live_mask], next_packets)

            live_buffer.advance()
            cache_buffer.advance()
            output_buffer.advance()

        missing_sum = (~delivered).to(dtype).sum() * cfg.train.missing_output_penalty
        total_loss_sum = (
            cfg.train.sanity_route_weight * route_loss_sum
            + cfg.train.sanity_delay_weight * delay_loss_sum
            + missing_sum
        )
        total_loss = total_loss_sum / batch_size

        return {
            "loss": total_loss,
            "metric_sums": {
                "loss_total": total_loss_sum.detach(),
                "loss_sanity_route": route_loss_sum.detach(),
                "loss_delay": delay_loss_sum.detach(),
                "loss_missing_output": missing_sum.detach(),
                "query_delivery_hit": delivered.to(dtype).sum(),
                "query_delivery_count": torch.tensor(float(batch_size), device=device, dtype=dtype),
                "avg_hops_sum": hops.to(dtype).sum(),
                "avg_hops_count": torch.tensor(float(batch_size), device=device, dtype=dtype),
                "delay_sum": delay_sum.detach().to(dtype),
                "delay_count": delay_count.detach().to(dtype),
                "packets_processed_sum": processed_packets_sum.detach().to(dtype),
            },
        }

    def _build_first_hop_route_logits(self, logits_compute: Tensor, dtype: torch.dtype) -> Tensor:
        route_logits = torch.full(
            (logits_compute.size(0), self.config.model.nodes_total),
            -1.0e9,
            device=logits_compute.device,
            dtype=dtype,
        )
        route_logits[:, 1:] = logits_compute.to(dtype)
        return route_logits

    def _predict_first_hop(
        self,
        packets: PacketBatch,
        first_hop_mask: Tensor,
        route_dtype: torch.dtype,
        address_dtype: torch.dtype,
    ) -> dict[str, Tensor]:
        cfg = self.config.model
        masked_packets = packets.select(first_hop_mask)
        if self.uses_legacy_first_hop_router():
            router_features = self.first_hop_router_ln(
                masked_packets.residual
                + self.key_proj(masked_packets.routing_key)
                + self.role_embed(masked_packets.role)
                + self.ttl_embed(masked_packets.ttl.clamp(min=0, max=cfg.max_ttl))
                + self.start_node_embed(masked_packets.current_node)
            )
            logits_compute = self.first_hop_router(router_features)
            aux_prediction = None
        else:
            router_outputs = self.first_hop_router(
                routing_key=masked_packets.routing_key,
                aux_features=(
                    self.role_embed(masked_packets.role)
                    + self.ttl_embed(masked_packets.ttl.clamp(min=0, max=cfg.max_ttl))
                    + self.start_node_embed(masked_packets.current_node)
                ),
                residual=masked_packets.residual,
                role=masked_packets.role,
            )
            logits_compute = router_outputs["logits"]
            aux_prediction = router_outputs.get("aux_address")

        one_hot = straight_through_sample(
            logits_compute,
            temperature=cfg.route_temperature,
            training=self.training,
        )
        node_index = 1 + one_hot.argmax(dim=-1)
        routed_address = one_hot @ self.address_table[1:].to(one_hot.dtype)
        route_logits = self._build_first_hop_route_logits(logits_compute, route_dtype)
        aux_address = routed_address.to(address_dtype) if aux_prediction is None else aux_prediction.to(address_dtype)
        return {
            "route_logits": route_logits,
            "dest_index": node_index,
            "predicted_address": routed_address.to(address_dtype),
            "aux_address": aux_address,
        }

    def _explicit_cache_read(
        self,
        *,
        packet_tensor: Tensor,
        role_tensor: Tensor,
        routing_key_tensor: Tensor,
        cache_tensor: Tensor,
        cache_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config.model
        full_explicit_read = torch.zeros_like(packet_tensor)
        rows_with_cache = cache_mask.any(dim=1)
        if not rows_with_cache.any():
            return packet_tensor, full_explicit_read

        query_mask = role_tensor[rows_with_cache] == ROLE_QUERY
        if not query_mask.any():
            return packet_tensor, full_explicit_read

        cache_tensor_with_cache = cache_tensor[rows_with_cache]
        cache_mask_with_cache = cache_mask[rows_with_cache]
        routing_key_with_cache = routing_key_tensor[rows_with_cache]
        cache_keys = cache_tensor_with_cache[:, :, : cfg.key_dim]
        attention_scores = torch.einsum("gqk,gck->gqc", routing_key_with_cache, cache_keys) / math.sqrt(cfg.key_dim)
        attention_scores = attention_scores.masked_fill(~cache_mask_with_cache[:, None, :], -1.0e9)
        attention_weights = attention_scores.softmax(dim=-1)
        explicit_read = torch.einsum("gqc,gcd->gqd", attention_weights, cache_tensor_with_cache)
        masked_read = explicit_read * query_mask.unsqueeze(-1).to(packet_tensor.dtype)

        updated_packet_tensor = packet_tensor.clone()
        updated_packet_tensor[rows_with_cache] = updated_packet_tensor[rows_with_cache] + masked_read
        full_explicit_read[rows_with_cache] = masked_read
        return updated_packet_tensor, full_explicit_read

    def _learned_cache_read(
        self,
        *,
        hidden_tensor: Tensor,
        role_tensor: Tensor,
        routing_key_tensor: Tensor,
        cache_tensor: Tensor,
        cache_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        full_read = torch.zeros_like(hidden_tensor)
        entropy = torch.zeros(hidden_tensor.size(0), hidden_tensor.size(1), device=hidden_tensor.device, dtype=hidden_tensor.dtype)
        top_mass = torch.zeros_like(entropy)
        entry_count = torch.zeros_like(entropy)
        rows_with_cache = cache_mask.any(dim=1)
        if not rows_with_cache.any() or self.cache_retriever is None:
            return full_read, entropy, top_mass, entry_count

        query_mask = role_tensor[rows_with_cache] == ROLE_QUERY
        if not query_mask.any():
            return full_read, entropy, top_mass, entry_count

        hidden_rows = hidden_tensor[rows_with_cache]
        routing_key_rows = routing_key_tensor[rows_with_cache]
        cache_rows = cache_tensor[rows_with_cache]
        cache_mask_rows = cache_mask[rows_with_cache]

        query_positions = query_mask.nonzero(as_tuple=False)
        selected_hidden = hidden_rows[query_positions[:, 0], query_positions[:, 1]]
        selected_keys = routing_key_rows[query_positions[:, 0], query_positions[:, 1]]
        selected_cache = cache_rows[query_positions[:, 0]]
        selected_cache_mask = cache_mask_rows[query_positions[:, 0]]

        retrieval_outputs = self.cache_retriever(
            hidden=selected_hidden,
            routing_key=selected_keys,
            cache_entries=selected_cache,
            cache_mask=selected_cache_mask,
        )
        full_read_rows = torch.zeros_like(hidden_rows)
        entropy_rows = torch.zeros(hidden_rows.size(0), hidden_rows.size(1), device=hidden_rows.device, dtype=hidden_rows.dtype)
        top_mass_rows = torch.zeros_like(entropy_rows)
        entry_count_rows = torch.zeros_like(entropy_rows)
        full_read_rows[query_positions[:, 0], query_positions[:, 1]] = retrieval_outputs["update"].to(full_read_rows.dtype)
        entropy_rows[query_positions[:, 0], query_positions[:, 1]] = retrieval_outputs["entropy"].to(entropy_rows.dtype)
        top_mass_rows[query_positions[:, 0], query_positions[:, 1]] = retrieval_outputs["top_mass"].to(top_mass_rows.dtype)
        entry_count_rows[query_positions[:, 0], query_positions[:, 1]] = retrieval_outputs["entry_count"].to(
            entry_count_rows.dtype
        )

        full_read[rows_with_cache] = full_read_rows
        entropy[rows_with_cache] = entropy_rows
        top_mass[rows_with_cache] = top_mass_rows
        entry_count[rows_with_cache] = entry_count_rows
        return full_read, entropy, top_mass, entry_count

    def _run_compute_nodes(
        self,
        packets: PacketBatch,
        cache: NodeCache,
        *,
        gradient_buffers: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        cfg = self.config.model
        device = packets.residual.device

        contextual_residual = (
            packets.residual
            + self.role_embed(packets.role)
            + self.ttl_embed(packets.ttl.clamp(min=0, max=cfg.max_ttl))
        )
        hidden = torch.zeros_like(packets.residual)
        memory_update = torch.zeros_like(packets.residual)
        retrieval_entropy = torch.zeros(packets.residual.size(0), device=device, dtype=packets.residual.dtype)
        retrieval_top_mass = torch.zeros_like(retrieval_entropy)
        retrieval_entry_count = torch.zeros_like(retrieval_entropy)
        visit_counts = torch.zeros(cfg.num_compute_nodes, device=device, dtype=packets.residual.dtype)
        task_visit_counts = torch.zeros_like(visit_counts)
        query_visit_counts = torch.zeros_like(visit_counts)
        bootstrap_visit_counts = torch.zeros_like(visit_counts)

        grouped: dict[tuple[int, int], list[int]] = {}
        batch_index_cpu = packets.batch_index.detach().cpu().tolist()
        node_index_cpu = packets.current_node.detach().cpu().tolist()
        for idx, (batch_index, node_index) in enumerate(zip(batch_index_cpu, node_index_cpu, strict=True)):
            grouped.setdefault((node_index, batch_index), []).append(idx)

        by_node: dict[int, list[tuple[int, list[int]]]] = {}
        for (node_index, batch_index), indices in sorted(grouped.items()):
            by_node.setdefault(node_index, []).append((batch_index, indices))

        for node_index, groups in by_node.items():
            if node_index == 0:
                continue
            cell = self.node_cells[node_index - 1]
            group_count = len(groups)
            max_packets = max(len(indices) for _, indices in groups)
            packet_tensor = torch.zeros(
                group_count,
                max_packets,
                cfg.d_model,
                device=device,
                dtype=packets.residual.dtype,
            )
            packet_mask = torch.zeros(group_count, max_packets, device=device, dtype=torch.bool)
            group_batch_index = torch.tensor([batch_index for batch_index, _ in groups], device=device, dtype=torch.long)
            group_node_index = torch.full((group_count,), node_index, device=device, dtype=torch.long)
            role_tensor = torch.zeros(group_count, max_packets, device=device, dtype=torch.long)
            routing_key_tensor = torch.zeros(
                group_count,
                max_packets,
                cfg.key_dim,
                device=device,
                dtype=packets.routing_key.dtype,
            )

            for group_idx, (_, indices) in enumerate(groups):
                index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                count = len(indices)
                packet_tensor[group_idx, :count] = contextual_residual[index_tensor]
                packet_mask[group_idx, :count] = True
                role_tensor[group_idx, :count] = packets.role[index_tensor]
                routing_key_tensor[group_idx, :count] = packets.routing_key[index_tensor]

            active_roles = role_tensor[packet_mask]
            visit_counts[node_index - 1] = visit_counts[node_index - 1] + active_roles.numel()
            task_visit_counts[node_index - 1] = task_visit_counts[node_index - 1] + (active_roles != ROLE_SANITY).sum()
            query_visit_counts[node_index - 1] = query_visit_counts[node_index - 1] + (active_roles == ROLE_QUERY).sum()
            bootstrap_visit_counts[node_index - 1] = (
                bootstrap_visit_counts[node_index - 1] + (active_roles == ROLE_SANITY).sum()
            )

            cache_tensor, cache_mask = cache.gather(group_batch_index, group_node_index)
            if self.uses_explicit_cache_read():
                packet_tensor, full_memory_update = self._explicit_cache_read(
                    packet_tensor=packet_tensor,
                    role_tensor=role_tensor,
                    routing_key_tensor=routing_key_tensor,
                    cache_tensor=cache_tensor,
                    cache_mask=cache_mask,
                )
            else:
                full_memory_update = torch.zeros(
                    group_count,
                    max_packets,
                    cfg.d_model,
                    device=device,
                    dtype=packet_tensor.dtype,
                )
            outputs = cell(packet_tensor, packet_mask, cache_tensor, cache_mask)
            if self.uses_learned_cache_read():
                (
                    full_memory_update,
                    group_entropy,
                    group_top_mass,
                    group_entry_count,
                ) = self._learned_cache_read(
                    hidden_tensor=outputs["hidden"],
                    role_tensor=role_tensor,
                    routing_key_tensor=routing_key_tensor,
                    cache_tensor=cache_tensor,
                    cache_mask=cache_mask,
                )
            else:
                group_entropy = torch.zeros(group_count, max_packets, device=device, dtype=packet_tensor.dtype)
                group_top_mass = torch.zeros_like(group_entropy)
                group_entry_count = torch.zeros_like(group_entropy)

            self._register_gradient_probe(
                outputs["hidden"],
                packet_mask=packet_mask,
                role_tensor=role_tensor,
                node_index=node_index,
                gradient_buffers=gradient_buffers,
            )

            for group_idx, (_, indices) in enumerate(groups):
                count = len(indices)
                index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                hidden[index_tensor] = outputs["hidden"][group_idx, :count].to(hidden.dtype)
                memory_update[index_tensor] = full_memory_update[group_idx, :count].to(memory_update.dtype)
                retrieval_entropy[index_tensor] = group_entropy[group_idx, :count].to(retrieval_entropy.dtype)
                retrieval_top_mass[index_tensor] = group_top_mass[group_idx, :count].to(retrieval_top_mass.dtype)
                retrieval_entry_count[index_tensor] = group_entry_count[group_idx, :count].to(retrieval_entry_count.dtype)

        delta = self.delta_head(hidden)
        direction = self.direction_head(hidden)
        magnitude = torch.sigmoid(self.magnitude_head(hidden))
        delay_logits = self.delay_head(hidden)
        next_residual = packets.residual + memory_update + delta.to(packets.residual.dtype)

        direction_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        unit_direction = direction / direction_norm
        address = magnitude * unit_direction
        if cfg.use_first_hop_key_hint and not cfg.use_learned_first_hop_router:
            first_hop_mask = (packets.hop_index == 0) & (
                (packets.role == ROLE_WRITER) | (packets.role == ROLE_QUERY)
            )
            if first_hop_mask.any():
                hint = self._first_hop_key_hint(packets.routing_key[first_hop_mask])
                address = address.clone()
                address[first_hop_mask] = hint + cfg.first_hop_hint_residual_scale * address[first_hop_mask]
        route_logits, _, dest_index = route_from_address(
            address,
            self.address_table,
            temperature=cfg.route_temperature,
            training=self.training,
        )
        predicted_dest_index = dest_index
        first_hop_aux_prediction = address
        teacher_forced_mask = torch.zeros_like(dest_index, dtype=torch.bool)
        clockwise_target = self._clockwise_target(packets.current_node)
        if cfg.use_learned_first_hop_router:
            first_hop_mask = (packets.hop_index == 0) & (
                (packets.role == ROLE_WRITER) | (packets.role == ROLE_QUERY)
            )
            if first_hop_mask.any():
                first_hop_outputs = self._predict_first_hop(
                    packets=packets,
                    first_hop_mask=first_hop_mask,
                    route_dtype=route_logits.dtype,
                    address_dtype=address.dtype,
                )
                route_logits = route_logits.clone()
                route_logits[first_hop_mask] = first_hop_outputs["route_logits"]
                address = address.clone()
                address[first_hop_mask] = first_hop_outputs["predicted_address"]
                dest_index = dest_index.clone()
                dest_index[first_hop_mask] = first_hop_outputs["dest_index"]
                predicted_dest_index = predicted_dest_index.clone()
                predicted_dest_index[first_hop_mask] = first_hop_outputs["dest_index"]
                first_hop_aux_prediction = first_hop_aux_prediction.clone()
                first_hop_aux_prediction[first_hop_mask] = first_hop_outputs["aux_address"]

                if self.training and self.first_hop_teacher_force_ratio > 0.0:
                    force_mask_local = (
                        torch.rand(first_hop_outputs["dest_index"].size(0), device=device) < self.first_hop_teacher_force_ratio
                    )
                    if force_mask_local.any():
                        dest_index = dest_index.clone()
                        teacher_forced_mask = teacher_forced_mask.clone()
                        first_hop_indices = first_hop_mask.nonzero(as_tuple=False).squeeze(-1)
                        forced_indices = first_hop_indices[force_mask_local]
                        dest_index[forced_indices] = packets.target_home[forced_indices]
                        teacher_forced_mask[forced_indices] = True
        if self.uses_growth_curriculum():
            route_logits = route_logits.clone()
            active_mask = self._active_node_mask(device=device, dtype=route_logits.dtype)
            route_logits = route_logits.masked_fill(active_mask.unsqueeze(0) == 0, -1.0e9)

            compute_mask = packets.current_node > 0
            if compute_mask.any() and self.config.growth.clock_prior_bias > 0.0:
                route_logits[compute_mask, clockwise_target[compute_mask]] = (
                    route_logits[compute_mask, clockwise_target[compute_mask]]
                    + self.config.growth.clock_prior_bias
                )

            delay_logits = delay_logits.clone()
            if self.config.growth.delay_zero_bias > 0.0:
                delay_logits[:, 0] = delay_logits[:, 0] + self.config.growth.delay_zero_bias

            coverage_mask = (packets.role == ROLE_SANITY) & torch.tensor(
                self.bootstrap_active,
                device=device,
                dtype=torch.bool,
            )
            if coverage_mask.any():
                if self.config.growth.bootstrap_clock_prior_bias > 0.0:
                    route_logits[coverage_mask, clockwise_target[coverage_mask]] = (
                        route_logits[coverage_mask, clockwise_target[coverage_mask]]
                        + self.config.growth.bootstrap_clock_prior_bias
                    )
                if self.config.growth.bootstrap_delay_zero_bias > 0.0:
                    delay_logits[coverage_mask, 0] = (
                        delay_logits[coverage_mask, 0] + self.config.growth.bootstrap_delay_zero_bias
                    )

            route_choice = straight_through_sample(
                route_logits,
                temperature=cfg.route_temperature,
                training=self.training,
            )
            predicted_dest_index = route_choice.argmax(dim=-1)
            dest_index = predicted_dest_index.clone()

        if teacher_forced_mask.any():
            dest_index = dest_index.clone()
            dest_index[teacher_forced_mask] = packets.target_home[teacher_forced_mask]

        _, delay_index = sample_delay(delay_logits, temperature=cfg.delay_temperature, training=self.training)
        if self.uses_growth_curriculum():
            coverage_mask = (packets.role == ROLE_SANITY) & torch.tensor(
                self.bootstrap_active,
                device=device,
                dtype=torch.bool,
            )
            if coverage_mask.any():
                if self.config.growth.bootstrap_force_clockwise:
                    dest_index = dest_index.clone()
                    dest_index[coverage_mask] = clockwise_target[coverage_mask]
                if self.config.growth.bootstrap_force_delay_zero:
                    delay_index = delay_index.clone()
                    delay_index[coverage_mask] = 0
        return {
            "next_residual": next_residual,
            "predicted_address": address,
            "route_logits": route_logits,
            "delay_logits": delay_logits,
            "dest_index": dest_index,
            "predicted_dest_index": predicted_dest_index,
            "delay_index": delay_index,
            "address_norm": address.norm(dim=-1),
            "first_hop_aux_prediction": first_hop_aux_prediction,
            "teacher_forced_mask": teacher_forced_mask,
            "retrieval_entropy": retrieval_entropy,
            "retrieval_top_mass": retrieval_top_mass,
            "retrieval_entry_count": retrieval_entry_count,
            "clockwise_target": clockwise_target,
            "visit_counts": visit_counts,
            "task_visit_counts": task_visit_counts,
            "query_visit_counts": query_visit_counts,
            "bootstrap_visit_counts": bootstrap_visit_counts,
        }

    def _schedule_packets(
        self,
        buffer: TemporalRingBuffer[PacketBatch],
        scheduled_steps: Tensor,
        packets: PacketBatch,
    ) -> None:
        for step in torch.unique(scheduled_steps).detach().cpu().tolist():
            mask = scheduled_steps == step
            buffer.schedule(int(step), packets.select(mask))

    def _first_hop_key_hint(self, routing_key: Tensor) -> Tensor:
        compute_scores = routing_key @ self.home_hash
        compute_index = compute_scores.argmax(dim=-1)
        return self.address_table[1:][compute_index]

    def _output_logits(self, residual: Tensor) -> Tensor:
        class_start = self.config.model.key_dim
        class_end = class_start + self.config.model.num_classes
        class_slice = residual[:, class_start:class_end]
        return self.output_head(residual) + self.config.model.readout_class_scale * class_slice

    def _schedule_cache_events(
        self,
        buffer: TemporalRingBuffer[CacheWriteEvent],
        scheduled_steps: Tensor,
        events: CacheWriteEvent,
    ) -> None:
        for step in torch.unique(scheduled_steps).detach().cpu().tolist():
            mask = scheduled_steps == step
            buffer.schedule(int(step), events.select(mask))

    def _schedule_output_events(
        self,
        buffer: TemporalRingBuffer[OutputEvent],
        scheduled_steps: Tensor,
        events: OutputEvent,
    ) -> None:
        for step in torch.unique(scheduled_steps).detach().cpu().tolist():
            mask = scheduled_steps == step
            buffer.schedule(int(step), events.select(mask))
