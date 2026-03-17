from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from apsgnn.buffer import NodeCache, TemporalRingBuffer
from apsgnn.config import ExperimentConfig
from apsgnn.node import ComputeNodeCell
from apsgnn.routing import build_address_table, route_from_address, sample_delay
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

        first_query_logits: list[Tensor | None] = [None for _ in range(batch_size)]
        query_delivered = torch.zeros(batch_size, device=device, dtype=torch.bool)
        query_last_hops = torch.zeros(batch_size, device=device, dtype=torch.long)

        loss_main_sum = zero
        loss_writer_sum = zero
        loss_query_sum = zero
        loss_home_out_sum = zero
        loss_delay_sum = zero
        loss_gravity_sum = zero
        loss_missing_sum = zero

        writer_hit_sum = zero
        writer_hit_count = zero
        query_hit_sum = zero
        query_hit_count = zero
        home_out_hit_sum = zero
        home_out_count = zero
        delay_sum = zero
        delay_count = zero
        processed_packets_sum = zero
        cache_mean_sum = zero
        cache_mean_count = zero
        cache_max_sum = zero
        cache_max_count = zero

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

            predictions = self._run_compute_nodes(active_packets, cache)
            route_logits = predictions["route_logits"]
            delay_logits = predictions["delay_logits"]
            dest_index = predictions["dest_index"]
            delay_index = predictions["delay_index"]
            next_residual = predictions["next_residual"]
            predicted_address = predictions["predicted_address"]
            address_norm = predictions["address_norm"]

            writer_first_mask = (active_packets.role == ROLE_WRITER) & (active_packets.hop_index == 0)
            if writer_first_mask.any():
                writer_targets = active_packets.target_home[writer_first_mask]
                writer_logits = route_logits[writer_first_mask]
                loss_writer_sum = loss_writer_sum + F.cross_entropy(writer_logits, writer_targets, reduction="sum")
                loss_writer_sum = loss_writer_sum + cfg.train.address_aux_weight * F.mse_loss(
                    predicted_address[writer_first_mask],
                    self.address_table[writer_targets],
                    reduction="sum",
                )
                writer_hit_sum = writer_hit_sum + (dest_index[writer_first_mask] == writer_targets).sum()
                writer_hit_count = writer_hit_count + writer_targets.numel()
                zeros = torch.zeros(writer_targets.numel(), device=device, dtype=torch.long)
                loss_delay_sum = loss_delay_sum + F.cross_entropy(delay_logits[writer_first_mask], zeros, reduction="sum")

            query_first_mask = (active_packets.role == ROLE_QUERY) & (active_packets.hop_index == 0)
            if query_first_mask.any():
                query_targets = active_packets.target_home[query_first_mask]
                query_logits = route_logits[query_first_mask]
                loss_query_sum = loss_query_sum + F.cross_entropy(query_logits, query_targets, reduction="sum")
                loss_query_sum = loss_query_sum + cfg.train.address_aux_weight * F.mse_loss(
                    predicted_address[query_first_mask],
                    self.address_table[query_targets],
                    reduction="sum",
                )
                query_hit_sum = query_hit_sum + (dest_index[query_first_mask] == query_targets).sum()
                query_hit_count = query_hit_count + query_targets.numel()
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

            gravity_mask = (active_packets.role == ROLE_QUERY) & (active_packets.has_visited_home | entered_home)
            if gravity_mask.any():
                loss_gravity_sum = loss_gravity_sum + address_norm[gravity_mask].square().sum()

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

        total_loss_sum = (
            loss_main_sum
            + cfg.train.aux_writer_weight * loss_writer_sum
            + cfg.train.aux_query_weight * loss_query_sum
            + cfg.train.aux_home_out_weight * loss_home_out_sum
            + cfg.train.delay_reg_weight * loss_delay_sum
            + cfg.train.gravity_weight * loss_gravity_sum
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
                "loss_home_to_output": loss_home_out_sum.detach(),
                "loss_delay": loss_delay_sum.detach(),
                "loss_gravity": loss_gravity_sum.detach(),
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
                "avg_hops_sum": query_last_hops.to(dtype).sum(),
                "avg_hops_count": torch.tensor(float(batch_size), device=device, dtype=dtype),
                "delay_sum": delay_sum.detach().to(dtype),
                "delay_count": delay_count.detach().to(dtype),
                "cache_mean_sum": cache_mean_sum.detach(),
                "cache_mean_count": cache_mean_count.detach().to(dtype),
                "cache_max_sum": cache_max_sum.detach(),
                "cache_max_count": cache_max_count.detach().to(dtype),
                "packets_processed_sum": processed_packets_sum.detach().to(dtype),
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

    def _run_compute_nodes(self, packets: PacketBatch, cache: NodeCache) -> dict[str, Tensor]:
        cfg = self.config.model
        device = packets.residual.device

        contextual_residual = (
            packets.residual
            + self.role_embed(packets.role)
            + self.ttl_embed(packets.ttl.clamp(min=0, max=cfg.max_ttl))
        )
        hidden = torch.zeros_like(packets.residual)
        memory_update = torch.zeros_like(packets.residual)

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

            cache_tensor, cache_mask = cache.gather(group_batch_index, group_node_index)
            rows_with_cache = cache_mask.any(dim=1)
            if rows_with_cache.any():
                query_mask = role_tensor[rows_with_cache] == ROLE_QUERY
                if query_mask.any():
                    cache_tensor_with_cache = cache_tensor[rows_with_cache]
                    cache_mask_with_cache = cache_mask[rows_with_cache]
                    routing_key_with_cache = routing_key_tensor[rows_with_cache]
                    cache_keys = cache_tensor_with_cache[:, :, : cfg.key_dim]
                    attention_scores = torch.einsum("gqk,gck->gqc", routing_key_with_cache, cache_keys) / math.sqrt(
                        cfg.key_dim
                    )
                    attention_scores = attention_scores.masked_fill(~cache_mask_with_cache[:, None, :], -1.0e9)
                    attention_weights = attention_scores.softmax(dim=-1)
                    explicit_read = torch.einsum("gqc,gcd->gqd", attention_weights, cache_tensor_with_cache)
                    packet_tensor = packet_tensor.clone()
                    packet_tensor[rows_with_cache] = packet_tensor[rows_with_cache] + explicit_read * query_mask.unsqueeze(
                        -1
                    ).to(packet_tensor.dtype)
                    full_explicit_read = torch.zeros(
                        group_count,
                        max_packets,
                        cfg.d_model,
                        device=device,
                        dtype=packet_tensor.dtype,
                    )
                    full_explicit_read[rows_with_cache] = explicit_read * query_mask.unsqueeze(-1).to(packet_tensor.dtype)
                else:
                    full_explicit_read = torch.zeros(
                        group_count,
                        max_packets,
                        cfg.d_model,
                        device=device,
                        dtype=packet_tensor.dtype,
                    )
            else:
                full_explicit_read = torch.zeros(
                    group_count,
                    max_packets,
                    cfg.d_model,
                    device=device,
                    dtype=packet_tensor.dtype,
                )
            outputs = cell(packet_tensor, packet_mask, cache_tensor, cache_mask)

            for group_idx, (_, indices) in enumerate(groups):
                count = len(indices)
                index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                hidden[index_tensor] = outputs["hidden"][group_idx, :count].to(hidden.dtype)
                memory_update[index_tensor] = full_explicit_read[group_idx, :count].to(memory_update.dtype)

        delta = self.delta_head(hidden)
        direction = self.direction_head(hidden)
        magnitude = torch.sigmoid(self.magnitude_head(hidden))
        delay_logits = self.delay_head(hidden)
        next_residual = packets.residual + memory_update + delta.to(packets.residual.dtype)

        direction_norm = direction.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        unit_direction = direction / direction_norm
        address = magnitude * unit_direction
        if cfg.use_first_hop_key_hint:
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
        _, delay_index = sample_delay(
            delay_logits,
            temperature=cfg.delay_temperature,
            training=self.training,
        )
        return {
            "next_residual": next_residual,
            "predicted_address": address,
            "route_logits": route_logits,
            "delay_logits": delay_logits,
            "dest_index": dest_index,
            "delay_index": delay_index,
            "address_norm": address.norm(dim=-1),
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
