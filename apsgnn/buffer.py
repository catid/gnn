from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import torch
from torch import Tensor


T = TypeVar("T")


@dataclass
class _ScheduledItem(Generic[T]):
    step: int
    item: T


class TemporalRingBuffer(Generic[T]):
    def __init__(self, size: int) -> None:
        self.size = size
        self.current_step = 0
        self._slots: list[list[_ScheduledItem[T]]] = [[] for _ in range(size)]

    def schedule(self, step: int, item: T) -> None:
        if step < self.current_step:
            raise ValueError(f"Cannot schedule item in the past: {step} < {self.current_step}")
        slot = step % self.size
        self._slots[slot].append(_ScheduledItem(step=step, item=item))

    def pop_current(self) -> list[T]:
        slot = self.current_step % self.size
        items = self._slots[slot]
        keep: list[_ScheduledItem[T]] = []
        ready: list[T] = []
        for scheduled in items:
            if scheduled.step == self.current_step:
                ready.append(scheduled.item)
            else:
                keep.append(scheduled)
        self._slots[slot] = keep
        return ready

    def advance(self) -> None:
        self.current_step += 1


class NodeCache:
    def __init__(
        self,
        batch_size: int,
        nodes_total: int,
        capacity: int,
        d_model: int,
        device: torch.device,
        dtype: torch.dtype,
        enabled: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.nodes_total = nodes_total
        self.capacity = capacity
        self.d_model = d_model
        self.enabled = enabled
        self.values = torch.zeros(batch_size, nodes_total, capacity, d_model, device=device, dtype=dtype)
        self.counts = torch.zeros(batch_size, nodes_total, device=device, dtype=torch.long)
        self.write_ptr = torch.zeros(batch_size, nodes_total, device=device, dtype=torch.long)

    def write(self, residuals: Tensor, batch_index: Tensor, node_index: Tensor) -> None:
        if not self.enabled or residuals.numel() == 0:
            return
        for i in range(residuals.size(0)):
            b = int(batch_index[i].item())
            n = int(node_index[i].item())
            pos = int(self.write_ptr[b, n].item())
            self.values[b, n, pos] = residuals[i]
            if self.counts[b, n] < self.capacity:
                self.counts[b, n] += 1
            self.write_ptr[b, n] = (self.write_ptr[b, n] + 1) % self.capacity

    def gather(self, batch_index: Tensor, node_index: Tensor) -> tuple[Tensor, Tensor]:
        groups = batch_index.numel()
        cache = torch.zeros(groups, self.capacity, self.d_model, device=self.values.device, dtype=self.values.dtype)
        mask = torch.zeros(groups, self.capacity, device=self.values.device, dtype=torch.bool)
        if not self.enabled:
            return cache, mask
        order = torch.arange(self.capacity, device=self.values.device)
        for i in range(groups):
            b = int(batch_index[i].item())
            n = int(node_index[i].item())
            count = int(self.counts[b, n].item())
            if count == 0:
                continue
            if count < self.capacity:
                cache[i, :count] = self.values[b, n, :count]
                mask[i, :count] = True
            else:
                start = int(self.write_ptr[b, n].item())
                indices = (order + start) % self.capacity
                cache[i] = self.values[b, n, indices]
                mask[i] = True
        return cache, mask

    def occupancy_stats(self) -> tuple[Tensor, Tensor]:
        counts = self.counts.to(torch.float32)
        return counts.mean(), counts.max()

    def detach_(self) -> None:
        self.values = self.values.detach()
