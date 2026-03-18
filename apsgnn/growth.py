from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
import math
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from apsgnn.config import ExperimentConfig

if TYPE_CHECKING:
    from apsgnn.model import APSGNNModel


@dataclass(frozen=True)
class GrowthStage:
    index: int
    active_compute_nodes: int
    start_step: int
    end_step: int
    bootstrap_steps: int

    @property
    def name(self) -> str:
        return f"stage_{self.active_compute_nodes}"

    def local_step(self, global_step: int) -> int:
        return global_step - self.start_step + 1

    def bootstrap_active(self, global_step: int) -> bool:
        return self.bootstrap_steps > 0 and self.local_step(global_step) <= self.bootstrap_steps


class GrowthSchedule:
    def __init__(self, stages: list[GrowthStage]) -> None:
        if not stages:
            raise ValueError("GrowthSchedule requires at least one stage.")
        self.stages = stages

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "GrowthSchedule":
        active_counts = list(config.growth.stage_active_counts)
        stage_steps = list(config.growth.stage_steps)

        if not config.growth.enabled or not active_counts:
            active_counts = [config.model.num_compute_nodes]
            stage_steps = [config.train.train_steps]

        if len(active_counts) != len(stage_steps):
            raise ValueError("growth.stage_active_counts and growth.stage_steps must have the same length.")
        if active_counts[-1] != config.model.num_compute_nodes:
            raise ValueError("Final growth stage must match model.num_compute_nodes.")

        total_stage_steps = sum(stage_steps)
        if total_stage_steps != config.train.train_steps:
            if total_stage_steps <= 0:
                raise ValueError("Sum of growth.stage_steps must be positive.")
            scaled: list[int] = []
            remaining = config.train.train_steps
            for index, duration in enumerate(stage_steps):
                if index == len(stage_steps) - 1:
                    scaled_duration = remaining
                else:
                    scaled_duration = max(1, round(duration * config.train.train_steps / total_stage_steps))
                    remaining -= scaled_duration
                scaled.append(scaled_duration)
            if sum(scaled) != config.train.train_steps or any(duration <= 0 for duration in scaled):
                raise ValueError("Could not rescale growth.stage_steps to match train.train_steps.")
            stage_steps = scaled

        stages: list[GrowthStage] = []
        start = 1
        for index, (active_count, duration) in enumerate(zip(active_counts, stage_steps, strict=True)):
            if active_count <= 0 or active_count > config.model.num_compute_nodes:
                raise ValueError(f"Invalid active compute node count: {active_count}")
            end = start + duration - 1
            stages.append(
                GrowthStage(
                    index=index,
                    active_compute_nodes=active_count,
                    start_step=start,
                    end_step=end,
                    bootstrap_steps=config.growth.bootstrap_steps,
                )
            )
            start = end + 1
        return cls(stages)

    @property
    def final_active_compute_nodes(self) -> int:
        return self.stages[-1].active_compute_nodes

    def stage_for_step(self, step: int) -> GrowthStage:
        for stage in self.stages:
            if stage.start_step <= step <= stage.end_step:
                return stage
        raise ValueError(f"No growth stage covers training step {step}.")


@dataclass(frozen=True)
class LeafInterval:
    start: int
    end: int

    @property
    def size(self) -> int:
        return self.end - self.start + 1

    def split(self) -> tuple["LeafInterval", "LeafInterval"]:
        if self.size <= 1:
            raise ValueError("Cannot split a singleton leaf interval.")
        midpoint = (self.start + self.end) // 2
        return LeafInterval(self.start, midpoint), LeafInterval(midpoint + 1, self.end)


@dataclass(frozen=True)
class GrowthTopology:
    final_compute_nodes: int
    ring_node_ids: tuple[int, ...]
    node_intervals: dict[int, LeafInterval]
    inactive_node_ids: tuple[int, ...]

    @property
    def active_compute_nodes(self) -> int:
        return len(self.ring_node_ids)

    def active_node_tensor(self, *, device: torch.device | None = None) -> Tensor:
        return torch.tensor(self.ring_node_ids, device=device, dtype=torch.long)

    def successor_lookup(self, *, device: torch.device | None = None) -> Tensor:
        lookup = torch.zeros(self.final_compute_nodes + 1, device=device, dtype=torch.long)
        for index, node_id in enumerate(self.ring_node_ids):
            lookup[node_id] = self.ring_node_ids[(index + 1) % len(self.ring_node_ids)]
        return lookup

    def interval_size(self, node_id: int) -> int:
        return self.node_intervals[node_id].size

    def ring_index(self, node_id: int) -> int:
        return self.ring_node_ids.index(node_id)

    def eligible_split_parents(self) -> list[int]:
        return [node_id for node_id in self.ring_node_ids if self.interval_size(node_id) > 1]

    def project_home_leaves(self, home_leaf: Tensor) -> Tensor:
        flat_home = home_leaf.reshape(-1)
        projected = torch.empty_like(flat_home)
        for node_id in self.ring_node_ids:
            interval = self.node_intervals[node_id]
            mask = (flat_home >= interval.start) & (flat_home <= interval.end)
            projected[mask] = node_id
        return projected.view_as(home_leaf)

    def split_selected(
        self,
        selected_parents: list[int],
    ) -> tuple["GrowthTopology", dict[int, int], dict[int, list[int]], dict[int, tuple[int, int]]]:
        if not selected_parents:
            return self, {}, {}, {}

        available_children = list(self.inactive_node_ids)
        if len(available_children) < len(selected_parents):
            raise ValueError("Not enough inactive nodes available for selective growth.")

        selected_set = set(selected_parents)
        new_ring: list[int] = []
        new_intervals: dict[int, LeafInterval] = {}
        parent_to_child: dict[int, int] = {}
        parent_to_children: dict[int, list[int]] = {}
        child_intervals: dict[int, tuple[int, int]] = {}

        for node_id in self.ring_node_ids:
            interval = self.node_intervals[node_id]
            if node_id not in selected_set:
                new_ring.append(node_id)
                new_intervals[node_id] = interval
                continue

            left_interval, right_interval = interval.split()
            child_id = available_children.pop(0)
            new_ring.extend([node_id, child_id])
            new_intervals[node_id] = left_interval
            new_intervals[child_id] = right_interval
            parent_to_child[node_id] = child_id
            parent_to_children[node_id] = [node_id, child_id]
            child_intervals[child_id] = (right_interval.start, right_interval.end)

        return (
            GrowthTopology(
                final_compute_nodes=self.final_compute_nodes,
                ring_node_ids=tuple(new_ring),
                node_intervals=new_intervals,
                inactive_node_ids=tuple(available_children),
            ),
            parent_to_child,
            parent_to_children,
            child_intervals,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "final_compute_nodes": self.final_compute_nodes,
            "ring_node_ids": list(self.ring_node_ids),
            "node_intervals": {
                str(node_id): {"start": interval.start, "end": interval.end}
                for node_id, interval in self.node_intervals.items()
            },
            "inactive_node_ids": list(self.inactive_node_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GrowthTopology":
        return cls(
            final_compute_nodes=int(payload["final_compute_nodes"]),
            ring_node_ids=tuple(int(node_id) for node_id in payload["ring_node_ids"]),
            node_intervals={
                int(node_id): LeafInterval(int(interval["start"]), int(interval["end"]))
                for node_id, interval in payload["node_intervals"].items()
            },
            inactive_node_ids=tuple(int(node_id) for node_id in payload["inactive_node_ids"]),
        )


def build_uniform_topology(final_compute_nodes: int, active_compute_nodes: int) -> GrowthTopology:
    if final_compute_nodes % active_compute_nodes != 0:
        raise ValueError("final_compute_nodes must be divisible by active_compute_nodes.")
    bucket_size = final_compute_nodes // active_compute_nodes
    intervals = {
        node_id: LeafInterval((node_id - 1) * bucket_size + 1, node_id * bucket_size)
        for node_id in range(1, active_compute_nodes + 1)
    }
    return GrowthTopology(
        final_compute_nodes=final_compute_nodes,
        ring_node_ids=tuple(range(1, active_compute_nodes + 1)),
        node_intervals=intervals,
        inactive_node_ids=tuple(range(active_compute_nodes + 1, final_compute_nodes + 1)),
    )


def build_initial_topology(config: ExperimentConfig, active_compute_nodes: int) -> GrowthTopology:
    return build_uniform_topology(config.model.num_compute_nodes, active_compute_nodes)


def project_home_leaves(home_leaf: Tensor, active_compute_nodes: int, final_compute_nodes: int) -> Tensor:
    if final_compute_nodes % active_compute_nodes != 0:
        raise ValueError("final_compute_nodes must be divisible by active_compute_nodes.")
    bucket_size = final_compute_nodes // active_compute_nodes
    return 1 + (home_leaf - 1) // bucket_size


def project_home_leaves_topology(home_leaf: Tensor, topology: GrowthTopology) -> Tensor:
    return topology.project_home_leaves(home_leaf)


def clockwise_successor(current_node: Tensor, active_compute_nodes: int) -> Tensor:
    successor = current_node.clone()
    compute_mask = current_node > 0
    successor[compute_mask] = ((current_node[compute_mask] - 1 + 1) % active_compute_nodes) + 1
    return successor


def active_node_ids(active_compute_nodes: int) -> Tensor:
    return torch.arange(1, active_compute_nodes + 1, dtype=torch.long)


def active_node_ids_from_topology(topology: GrowthTopology) -> Tensor:
    return torch.tensor(topology.ring_node_ids, dtype=torch.long)


def _zscore_dict(values: dict[int, float]) -> dict[int, float]:
    if not values:
        return {}
    tensor = torch.tensor(list(values.values()), dtype=torch.float64)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    if std < 1.0e-9:
        return {node_id: 0.0 for node_id in values}
    return {node_id: (value - mean) / std for node_id, value in values.items()}


def _topology_state(topology: GrowthTopology) -> tuple[int, ...]:
    return tuple(sorted((topology.interval_size(node_id) for node_id in topology.ring_node_ids), reverse=True))


def _split_interval_size(size: int) -> tuple[int, int]:
    if size <= 1:
        raise ValueError("Cannot split a singleton interval size.")
    return (size + 1) // 2, size // 2


def _split_state(state: tuple[int, ...], selected_sizes: tuple[int, ...]) -> tuple[int, ...]:
    counts = Counter(state)
    for size in selected_sizes:
        if counts[size] <= 0:
            raise ValueError("Selected split sizes do not match current topology state.")
        counts[size] -= 1
        if counts[size] == 0:
            del counts[size]
        left_size, right_size = _split_interval_size(size)
        counts[left_size] += 1
        counts[right_size] += 1
    expanded: list[int] = []
    for size, count in counts.items():
        expanded.extend([size] * count)
    return tuple(sorted(expanded, reverse=True))


@lru_cache(maxsize=None)
def _is_schedule_feasible(state: tuple[int, ...], remaining_active_counts: tuple[int, ...]) -> bool:
    if not remaining_active_counts:
        return True
    next_active = remaining_active_counts[0]
    if next_active < len(state):
        return False
    delta = next_active - len(state)
    eligible_indices = tuple(index for index, size in enumerate(state) if size > 1)
    if delta > len(eligible_indices):
        return False
    if delta == 0:
        return _is_schedule_feasible(state, remaining_active_counts[1:])
    for selected_indices in combinations(eligible_indices, delta):
        selected_sizes = tuple(state[index] for index in selected_indices)
        if _is_schedule_feasible(_split_state(state, selected_sizes), remaining_active_counts[1:]):
            return True
    return False


def _feasible_parent_subsets(
    topology: GrowthTopology,
    count: int,
    *,
    remaining_active_counts: tuple[int, ...],
) -> list[tuple[int, ...]]:
    eligible = topology.eligible_split_parents()
    if count > len(eligible):
        return []
    if count == 0:
        return [tuple()]
    state = _topology_state(topology)
    feasible: list[tuple[int, ...]] = []
    for selected_indices in combinations(range(len(eligible)), count):
        selected = tuple(eligible[index] for index in selected_indices)
        selected_sizes = tuple(topology.interval_size(node_id) for node_id in selected)
        next_state = _split_state(state, selected_sizes)
        if _is_schedule_feasible(next_state, remaining_active_counts):
            feasible.append(selected)
    return feasible


def _balanced_subset_key(topology: GrowthTopology, subset: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    ordered = sorted(
        subset,
        key=lambda node_id: (-topology.interval_size(node_id), topology.ring_index(node_id)),
    )
    return tuple((-topology.interval_size(node_id), topology.ring_index(node_id)) for node_id in ordered)


def balanced_split_parents(topology: GrowthTopology, feasible_subsets: list[tuple[int, ...]]) -> list[int]:
    if not feasible_subsets:
        raise ValueError("Balanced split requested with no feasible parent subsets.")
    selected = min(feasible_subsets, key=lambda subset: _balanced_subset_key(topology, subset))
    return list(selected)


def random_split_parents(topology: GrowthTopology, feasible_subsets: list[tuple[int, ...]], *, seed: int) -> list[int]:
    if not feasible_subsets:
        raise ValueError("Random split requested with no feasible parent subsets.")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    choice = int(torch.randint(len(feasible_subsets), (1,), generator=generator).item())
    return list(feasible_subsets[choice])


def utility_split_parents(
    topology: GrowthTopology,
    feasible_subsets: list[tuple[int, ...]],
    *,
    utility_scores: dict[int, float],
) -> list[int]:
    if not feasible_subsets:
        raise ValueError("Utility split requested with no feasible parent subsets.")
    selected = min(
        feasible_subsets,
        key=lambda subset: (
            -sum(utility_scores.get(node_id, 0.0) for node_id in subset),
            _balanced_subset_key(topology, subset),
        ),
    )
    return list(selected)


def transition_topology_for_growth(
    topology: GrowthTopology,
    next_active_compute_nodes: int,
    *,
    split_parent_policy: str,
    utility_components: dict[int, dict[str, float]] | None,
    utility_alpha: float,
    utility_visit_weight: float = 1.0,
    utility_grad_weight: float = 1.0,
    utility_query_visit_weight: float = 0.0,
    utility_query_grad_weight: float = 0.0,
    seed: int,
    future_active_counts: list[int] | tuple[int, ...] = (),
) -> tuple[GrowthTopology, dict[str, Any]]:
    if next_active_compute_nodes <= topology.active_compute_nodes:
        return topology, default_split_stats()

    delta = next_active_compute_nodes - topology.active_compute_nodes
    eligible = topology.eligible_split_parents()
    if delta > len(eligible):
        raise ValueError("Selective growth requested more splits than eligible parents.")
    future_active_counts_tuple = tuple(int(count) for count in future_active_counts)
    feasible_subsets = _feasible_parent_subsets(
        topology,
        delta,
        remaining_active_counts=future_active_counts_tuple,
    )
    if not feasible_subsets:
        raise ValueError("No schedule-feasible selective split set exists for the requested transition.")

    utility_components = utility_components or {}
    utility_scores = {
        node_id: utility_components.get(node_id, {}).get("score", 0.0)
        for node_id in eligible
    }

    if split_parent_policy == "balanced":
        selected = balanced_split_parents(topology, feasible_subsets)
    elif split_parent_policy == "random":
        selected = random_split_parents(topology, feasible_subsets, seed=seed)
    elif split_parent_policy == "utility":
        selected = utility_split_parents(topology, feasible_subsets, utility_scores=utility_scores)
    else:
        raise ValueError(f"Unsupported growth.split_parent_policy: {split_parent_policy}")

    selected_set = set(selected)
    new_topology, parent_to_child, parent_to_children, child_intervals = topology.split_selected(selected)
    selected_scores = {node_id: utility_scores.get(node_id, 0.0) for node_id in selected}
    unselected = [node_id for node_id in eligible if node_id not in selected_set]
    unselected_scores = {node_id: utility_scores.get(node_id, 0.0) for node_id in unselected}

    split_stats = {
        "selection_policy": split_parent_policy,
        "utility_alpha": utility_alpha,
        "utility_visit_weight": utility_visit_weight,
        "utility_grad_weight": utility_grad_weight,
        "utility_query_visit_weight": utility_query_visit_weight,
        "utility_query_grad_weight": utility_query_grad_weight,
        "eligible_parents": eligible,
        "selected_parents": selected,
        "unselected_parents": unselected,
        "remaining_future_active_counts": list(future_active_counts_tuple),
        "feasible_parent_subset_count": len(feasible_subsets),
        "selected_parent_scores": selected_scores,
        "unselected_parent_scores": unselected_scores,
        "parent_components": {
            node_id: utility_components.get(
                node_id,
                {
                    "visit": 0.0,
                    "grad": 0.0,
                    "success": 0.0,
                    "query_visit": 0.0,
                    "query_grad": 0.0,
                    "visit_z": 0.0,
                    "grad_z": 0.0,
                    "success_z": 0.0,
                    "query_visit_z": 0.0,
                    "query_grad_z": 0.0,
                    "score": 0.0,
                },
            )
            for node_id in eligible
        },
        "random_seed": seed if split_parent_policy == "random" else None,
        "parent_to_child": parent_to_child,
        "parent_to_children": parent_to_children,
        "child_intervals": child_intervals,
        "ring_node_ids_before": list(topology.ring_node_ids),
        "ring_node_ids_after": list(new_topology.ring_node_ids),
        "sibling_pairs": [(parent_id, child_id) for parent_id, child_id in parent_to_child.items()],
    }
    return new_topology, split_stats


def _noise_like(reference: Tensor, generator: torch.Generator, scale: float) -> Tensor:
    if reference.numel() < 2:
        std = 0.0
    else:
        std = float(reference.detach().float().std(unbiased=False).item())
    sigma = scale * (std if std > 1.0e-6 else 1.0)
    noise = torch.randn(reference.shape, generator=generator, dtype=torch.float32)
    return noise.to(device=reference.device, dtype=reference.dtype) * sigma


def _router_heads(model: APSGNNModel) -> list[torch.nn.Linear]:
    router = model.first_hop_router
    heads: list[torch.nn.Linear] = []
    if hasattr(router, "writer_head"):
        heads.append(router.writer_head)
    if hasattr(router, "query_head"):
        heads.append(router.query_head)
    if hasattr(router, "shared_head"):
        heads.append(router.shared_head)
    return heads


def split_model_for_growth(
    model: APSGNNModel,
    previous_active_compute_nodes: int,
    next_active_compute_nodes: int,
    *,
    split_mode: str,
    mutation_scale: float,
    seed: int,
) -> dict[str, Any]:
    if next_active_compute_nodes <= previous_active_compute_nodes:
        return {"sibling_divergence": 0.0, "sibling_pairs": [], "mutated_children": []}
    if next_active_compute_nodes != previous_active_compute_nodes * 2:
        raise ValueError("Growth splits must double the active compute node count.")

    source_cells = [
        deepcopy(model.node_cells[parent_index].state_dict()) for parent_index in range(previous_active_compute_nodes)
    ]
    source_start_embeddings = model.start_node_embed.weight.detach().clone()
    source_router_rows = [head.weight.detach().clone() for head in _router_heads(model)]
    source_router_biases = [head.bias.detach().clone() if head.bias is not None else None for head in _router_heads(model)]

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    sibling_pairs: list[tuple[int, int]] = []
    mutated_children: list[int] = []
    sibling_divergence = 0.0

    for parent_index in range(previous_active_compute_nodes):
        parent_node_id = parent_index + 1
        left_child = parent_index * 2 + 1
        right_child = left_child + 1
        sibling_pairs.append((left_child, right_child))

        model.node_cells[left_child - 1].load_state_dict(source_cells[parent_index])
        model.node_cells[right_child - 1].load_state_dict(source_cells[parent_index])
        model.start_node_embed.weight.data[left_child].copy_(source_start_embeddings[parent_node_id])
        model.start_node_embed.weight.data[right_child].copy_(source_start_embeddings[parent_node_id])

        for head_index, head in enumerate(_router_heads(model)):
            head.weight.data[left_child - 1].copy_(source_router_rows[head_index][parent_index])
            head.weight.data[right_child - 1].copy_(source_router_rows[head_index][parent_index])
            if source_router_biases[head_index] is not None and head.bias is not None:
                head.bias.data[left_child - 1].copy_(source_router_biases[head_index][parent_index])
                head.bias.data[right_child - 1].copy_(source_router_biases[head_index][parent_index])

        if split_mode != "mutate":
            continue

        mutated_children.append(right_child)
        divergence_terms: list[Tensor] = []
        mutated_cell = model.node_cells[right_child - 1]
        sibling_cell = model.node_cells[left_child - 1]
        for name, parameter in mutated_cell.named_parameters():
            if "ff.net" not in name:
                continue
            parameter.data.add_(_noise_like(parameter.data, generator, mutation_scale))
            sibling_parameter = dict(sibling_cell.named_parameters())[name]
            divergence_terms.append((parameter.detach() - sibling_parameter.detach()).float().pow(2).mean())

        model.start_node_embed.weight.data[right_child].add_(
            _noise_like(model.start_node_embed.weight.data[right_child], generator, mutation_scale)
        )
        divergence_terms.append(
            (
                model.start_node_embed.weight.data[right_child].detach()
                - model.start_node_embed.weight.data[left_child].detach()
            ).float().pow(2).mean()
        )

        for head in _router_heads(model):
            head.weight.data[right_child - 1].add_(
                _noise_like(head.weight.data[right_child - 1], generator, mutation_scale)
            )
            divergence_terms.append(
                (
                    head.weight.data[right_child - 1].detach()
                    - head.weight.data[left_child - 1].detach()
                ).float().pow(2).mean()
            )
            if head.bias is not None:
                head.bias.data[right_child - 1].add_(
                    _noise_like(head.bias.data[right_child - 1], generator, mutation_scale)
                )
                divergence_terms.append(
                    (
                        head.bias.data[right_child - 1].detach()
                        - head.bias.data[left_child - 1].detach()
                    ).float().pow(2).mean()
                )

        sibling_divergence += float(torch.stack(divergence_terms).mean().sqrt().item())

    if mutated_children:
        sibling_divergence /= len(mutated_children)

    return {
        "sibling_divergence": sibling_divergence,
        "sibling_pairs": sibling_pairs,
        "mutated_children": mutated_children,
    }


def selective_split_model_for_growth(
    model: APSGNNModel,
    *,
    parent_child_pairs: list[tuple[int, int]],
    split_mode: str,
    mutation_scale: float,
    seed: int,
    mutate_parent_ids: set[int] | None = None,
) -> dict[str, Any]:
    if not parent_child_pairs:
        return default_split_stats()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    mutated_children: list[int] = []
    sibling_divergence = 0.0

    for parent_node_id, child_node_id in parent_child_pairs:
        parent_index = parent_node_id - 1
        child_index = child_node_id - 1
        source_cell = deepcopy(model.node_cells[parent_index].state_dict())
        model.node_cells[child_index].load_state_dict(source_cell)
        model.start_node_embed.weight.data[child_node_id].copy_(model.start_node_embed.weight.data[parent_node_id].detach())
        for head in _router_heads(model):
            head.weight.data[child_index].copy_(head.weight.data[parent_index].detach())
            if head.bias is not None:
                head.bias.data[child_index].copy_(head.bias.data[parent_index].detach())

        if split_mode != "mutate":
            continue
        if mutate_parent_ids is not None and parent_node_id not in mutate_parent_ids:
            continue

        mutated_children.append(child_node_id)
        divergence_terms: list[Tensor] = []
        mutated_cell = model.node_cells[child_index]
        sibling_cell = model.node_cells[parent_index]
        sibling_parameters = dict(sibling_cell.named_parameters())
        for name, parameter in mutated_cell.named_parameters():
            if "ff.net" not in name:
                continue
            parameter.data.add_(_noise_like(parameter.data, generator, mutation_scale))
            divergence_terms.append((parameter.detach() - sibling_parameters[name].detach()).float().pow(2).mean())

        model.start_node_embed.weight.data[child_node_id].add_(
            _noise_like(model.start_node_embed.weight.data[child_node_id], generator, mutation_scale)
        )
        divergence_terms.append(
            (
                model.start_node_embed.weight.data[child_node_id].detach()
                - model.start_node_embed.weight.data[parent_node_id].detach()
            ).float().pow(2).mean()
        )

        for head in _router_heads(model):
            head.weight.data[child_index].add_(_noise_like(head.weight.data[child_index], generator, mutation_scale))
            divergence_terms.append(
                (
                    head.weight.data[child_index].detach()
                    - head.weight.data[parent_index].detach()
                ).float().pow(2).mean()
            )
            if head.bias is not None:
                head.bias.data[child_index].add_(_noise_like(head.bias.data[child_index], generator, mutation_scale))
                divergence_terms.append(
                    (
                        head.bias.data[child_index].detach()
                        - head.bias.data[parent_index].detach()
                    ).float().pow(2).mean()
                )

        sibling_divergence += float(torch.stack(divergence_terms).mean().sqrt().item())

    if mutated_children:
        sibling_divergence /= len(mutated_children)

    return {
        "sibling_divergence": sibling_divergence,
        "sibling_pairs": parent_child_pairs,
        "mutated_children": mutated_children,
        "mutated_parent_ids": sorted((mutate_parent_ids or set()) & {parent_id for parent_id, _ in parent_child_pairs}),
    }


def default_split_stats() -> dict[str, Any]:
    return {
        "sibling_divergence": 0.0,
        "sibling_pairs": [],
        "mutated_children": [],
        "mutated_parent_ids": [],
        "mutation_score_margin": None,
        "mutation_reference_score": None,
        "mutation_reference_kind": None,
        "mutation_requires_stagnation": False,
        "mutation_stagnated": None,
        "mutation_stagnation_window": None,
        "mutation_stagnation_delta": None,
        "mutation_stage_val_tail": [],
        "mutation_stage_val_range": None,
    }


def mutation_stagnation_info(
    stage_val_history: list[float],
    *,
    window: int,
    delta: float,
) -> dict[str, Any]:
    effective_window = max(int(window), 2)
    tail = [float(value) for value in stage_val_history[-effective_window:]]
    if len(tail) < 2:
        return {
            "window": effective_window,
            "delta": float(delta),
            "tail_values": tail,
            "range": None,
            "stagnated": False,
        }
    value_range = max(tail) - min(tail)
    return {
        "window": effective_window,
        "delta": float(delta),
        "tail_values": tail,
        "range": float(value_range),
        "stagnated": bool(value_range <= float(delta)),
    }


def _select_mutated_parents(
    parent_child_pairs: list[tuple[int, int]],
    *,
    split_mode: str,
    next_stage_index: int | None,
    mutation_stage_index_min: int,
    mutation_selected_fraction: float,
    mutation_score_margin: float,
    mutation_min_visit_z: float,
    mutation_min_query_grad_z: float,
    mutation_require_stagnation: bool,
    transition_stats: dict[str, Any] | None,
) -> tuple[set[int], float | None, str | None]:
    if split_mode != "mutate":
        return set(), None, None
    if next_stage_index is not None and next_stage_index < mutation_stage_index_min:
        return set(), None, None
    if mutation_require_stagnation:
        if transition_stats is None or not bool(transition_stats.get("stage_stagnated", False)):
            return set(), None, None

    parent_ids = [parent_id for parent_id, _ in parent_child_pairs]
    if not parent_ids:
        return set(), None, None

    fraction = float(mutation_selected_fraction)
    if fraction <= 0.0:
        return set(), None, None

    parent_scores = {}
    unselected_scores = {}
    if transition_stats is not None:
        parent_scores = {
            int(parent_id): float(score)
            for parent_id, score in transition_stats.get("selected_parent_scores", {}).items()
        }
        unselected_scores = {
            int(parent_id): float(score)
            for parent_id, score in transition_stats.get("unselected_parent_scores", {}).items()
        }
    ordered = sorted(parent_ids, key=lambda parent_id: (-parent_scores.get(parent_id, 0.0), parent_id))
    reference_score: float | None = None
    reference_kind: str | None = None
    filtered = ordered
    if float(mutation_score_margin) > -1.0e8 and ordered:
        if unselected_scores:
            reference_score = max(unselected_scores.values())
            reference_kind = "best_unselected"
        else:
            reference_score = sum(parent_scores.get(parent_id, 0.0) for parent_id in ordered) / float(len(ordered))
            reference_kind = "selected_mean"
        threshold = reference_score + float(mutation_score_margin)
        filtered = [parent_id for parent_id in ordered if parent_scores.get(parent_id, 0.0) >= threshold]
        if not filtered:
            return set(), reference_score, reference_kind
    if transition_stats is not None and (
        float(mutation_min_visit_z) > -1.0e8 or float(mutation_min_query_grad_z) > -1.0e8
    ):
        parent_components = {
            int(parent_id): components
            for parent_id, components in transition_stats.get("parent_components", {}).items()
        }
        component_filtered = []
        for parent_id in filtered:
            components = parent_components.get(parent_id, {})
            visit_z = float(components.get("visit_z", 0.0))
            query_grad_z = float(components.get("query_grad_z", 0.0))
            if visit_z < float(mutation_min_visit_z):
                continue
            if query_grad_z < float(mutation_min_query_grad_z):
                continue
            component_filtered.append(parent_id)
        filtered = component_filtered
        if not filtered:
            return set(), reference_score, reference_kind
    if fraction >= 1.0:
        return set(filtered), reference_score, reference_kind
    keep = max(1, math.ceil(len(filtered) * fraction))
    return set(filtered[:keep]), reference_score, reference_kind


def transition_model_for_growth(
    model: APSGNNModel,
    previous_active_compute_nodes: int,
    next_active_compute_nodes: int,
    *,
    transition_mode: str,
    split_mode: str,
    mutation_scale: float,
    seed: int,
    selective_parent_child_pairs: list[tuple[int, int]] | None = None,
    transition_stats: dict[str, Any] | None = None,
    next_stage_index: int | None = None,
    mutation_stage_index_min: int = 0,
    mutation_selected_fraction: float = 1.0,
    mutation_score_margin: float = -1.0e9,
    mutation_min_visit_z: float = -1.0e9,
    mutation_min_query_grad_z: float = -1.0e9,
    mutation_require_stagnation: bool = False,
) -> dict[str, Any]:
    if next_active_compute_nodes <= previous_active_compute_nodes:
        return default_split_stats()
    if selective_parent_child_pairs is not None:
        base_stats = transition_stats or {}
        if transition_mode == "activate":
            return {
                **default_split_stats(),
                **base_stats,
                "sibling_pairs": selective_parent_child_pairs,
                "mutated_children": [],
                "mutated_parent_ids": [],
            }
        if transition_mode != "split":
            raise ValueError(f"Unsupported growth.transition_mode: {transition_mode}")
        mutate_parent_ids, mutation_reference_score, mutation_reference_kind = _select_mutated_parents(
            selective_parent_child_pairs,
            split_mode=split_mode,
            next_stage_index=next_stage_index,
            mutation_stage_index_min=mutation_stage_index_min,
            mutation_selected_fraction=mutation_selected_fraction,
            mutation_score_margin=mutation_score_margin,
            mutation_min_visit_z=mutation_min_visit_z,
            mutation_min_query_grad_z=mutation_min_query_grad_z,
            mutation_require_stagnation=mutation_require_stagnation,
            transition_stats=transition_stats,
        )
        model_stats = selective_split_model_for_growth(
            model,
            parent_child_pairs=selective_parent_child_pairs,
            split_mode=split_mode,
            mutation_scale=mutation_scale,
            seed=seed,
            mutate_parent_ids=mutate_parent_ids,
        )
        return {
            **base_stats,
            **model_stats,
            "mutation_stage_index_min": mutation_stage_index_min,
            "mutation_selected_fraction": mutation_selected_fraction,
            "mutation_score_margin": mutation_score_margin,
            "mutation_min_visit_z": mutation_min_visit_z,
            "mutation_min_query_grad_z": mutation_min_query_grad_z,
            "mutation_requires_stagnation": mutation_require_stagnation,
            "mutation_stagnated": None if transition_stats is None else transition_stats.get("stage_stagnated"),
            "mutation_stagnation_window": None if transition_stats is None else transition_stats.get("stage_stagnation_window"),
            "mutation_stagnation_delta": None if transition_stats is None else transition_stats.get("stage_stagnation_delta"),
            "mutation_stage_val_tail": [] if transition_stats is None else transition_stats.get("stage_val_tail", []),
            "mutation_stage_val_range": None if transition_stats is None else transition_stats.get("stage_val_range"),
            "mutation_reference_score": mutation_reference_score,
            "mutation_reference_kind": mutation_reference_kind,
            "next_stage_index": next_stage_index,
        }
    if transition_mode == "activate":
        return default_split_stats()
    if transition_mode != "split":
        raise ValueError(f"Unsupported growth.transition_mode: {transition_mode}")
    return split_model_for_growth(
        model,
        previous_active_compute_nodes,
        next_active_compute_nodes,
        split_mode=split_mode,
        mutation_scale=mutation_scale,
        seed=seed,
    )


def collect_node_gradient_norms(model: APSGNNModel) -> Tensor:
    norms = torch.zeros(model.config.model.num_compute_nodes, device=model.address_table.device, dtype=torch.float32)
    for node_index, cell in enumerate(model.node_cells, start=1):
        norm_sq = torch.zeros((), device=norms.device, dtype=torch.float32)
        for parameter in cell.parameters():
            if parameter.grad is None:
                continue
            grad = parameter.grad.detach().float()
            norm_sq = norm_sq + grad.pow(2).sum()
        norms[node_index - 1] = norm_sq.sqrt()
    return norms


class CoverageTracker:
    CHECKPOINTS = (10, 50, 100, 200)
    COVERAGE_LEVELS = (0.5, 0.75, 1.0)

    def __init__(
        self,
        *,
        num_compute_nodes: int,
        gradient_norm_threshold: float,
        utility_ema_decay: float,
        utility_tail_fraction: float = 1.0,
    ) -> None:
        self.num_compute_nodes = num_compute_nodes
        self.gradient_norm_threshold = gradient_norm_threshold
        self.utility_ema_decay = utility_ema_decay
        self.utility_tail_fraction = min(max(utility_tail_fraction, 1.0e-6), 1.0)
        self.completed_stages: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.current_stage_summary: dict[str, Any] | None = None

    @staticmethod
    def _active_view(values: Tensor, active_node_ids: list[int]) -> Tensor:
        if not active_node_ids:
            return values.new_zeros((0,))
        indices = torch.tensor([node_id - 1 for node_id in active_node_ids], dtype=torch.long)
        return values.index_select(0, indices)

    @classmethod
    def _coverage_fraction(cls, values: Tensor, active_node_ids: list[int], *, threshold: float) -> float:
        active_values = cls._active_view(values, active_node_ids)
        if active_values.numel() == 0:
            return 0.0
        return float((active_values > threshold).to(torch.float32).mean().item())

    @staticmethod
    def _count_fraction(values: Tensor, active_node_ids: list[int], *, threshold: float) -> float:
        active_values = CoverageTracker._active_view(values, active_node_ids)
        if active_values.numel() == 0:
            return 0.0
        return float((active_values >= threshold).to(torch.float32).mean().item())

    @staticmethod
    def _normalized_entropy(values: Tensor, active_node_ids: list[int]) -> float:
        trimmed = CoverageTracker._active_view(values, active_node_ids).clamp_min(0.0)
        total = float(trimmed.sum().item())
        if total <= 0.0 or trimmed.numel() <= 1:
            return 0.0
        probabilities = trimmed / total
        safe = probabilities.clamp_min(1.0e-12)
        entropy = -(probabilities * safe.log()).sum()
        return float((entropy / math.log(trimmed.numel())).item())

    @staticmethod
    def _gini(values: Tensor, active_node_ids: list[int]) -> float:
        trimmed = CoverageTracker._active_view(values, active_node_ids).clamp_min(0.0)
        total = float(trimmed.sum().item())
        if total <= 0.0 or trimmed.numel() <= 1:
            return 0.0
        sorted_values = torch.sort(trimmed).values
        n = sorted_values.numel()
        index = torch.arange(1, n + 1, dtype=sorted_values.dtype)
        gini = (2.0 * (index * sorted_values).sum() / (n * sorted_values.sum())) - (n + 1) / n
        return float(gini.item())

    @staticmethod
    def _post_bootstrap_slope(history: list[dict[str, float | int]], metric_key: str) -> float:
        post_bootstrap = [row for row in history if int(row["post_bootstrap_step"]) >= 0]
        if len(post_bootstrap) < 2:
            return 0.0
        first = post_bootstrap[0]
        window = [row for row in post_bootstrap if int(row["post_bootstrap_step"]) <= 20]
        last = window[-1] if len(window) >= 2 else post_bootstrap[-1]
        step_delta = max(int(last["post_bootstrap_step"]) - int(first["post_bootstrap_step"]), 1)
        value_delta = float(last[metric_key]) - float(first[metric_key])
        return value_delta / step_delta

    @staticmethod
    def _maybe_set_threshold_times(
        summary: dict[str, Any],
        *,
        local_step: int,
        metric_name: str,
        value: float,
    ) -> None:
        post_bootstrap_step = max(local_step - int(summary["bootstrap_steps"]), 0)
        thresholds = summary[metric_name]
        for level in CoverageTracker.COVERAGE_LEVELS:
            key = f"{int(level * 100)}"
            if thresholds[key] is None and value >= level:
                thresholds[key] = post_bootstrap_step

    def start_stage(
        self,
        stage: GrowthStage,
        *,
        split_stats: dict[str, Any] | None = None,
        topology: GrowthTopology | None = None,
    ) -> None:
        if self.current_stage_summary is not None:
            self.completed_stages.append(self._finalize_stage())
        active_node_ids = list(topology.ring_node_ids) if topology is not None else list(range(1, stage.active_compute_nodes + 1))
        non_bootstrap_steps = max(stage.end_step - stage.start_step + 1 - stage.bootstrap_steps, 1)
        tail_steps = max(1, math.ceil(non_bootstrap_steps * self.utility_tail_fraction))
        tail_start_local_step = stage.bootstrap_steps + max(1, non_bootstrap_steps - tail_steps + 1)
        self.current_stage_summary = {
            "stage_index": stage.index,
            "stage_name": stage.name,
            "active_compute_nodes": stage.active_compute_nodes,
            "active_node_ids": active_node_ids,
            "start_step": stage.start_step,
            "bootstrap_steps": stage.bootstrap_steps,
            "utility_tail_start_local_step": tail_start_local_step,
            "all_visit_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "task_visit_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "query_visit_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "bootstrap_visit_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "all_grad_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "task_grad_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "query_grad_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "bootstrap_grad_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "task_visit_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "task_grad_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "query_visit_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "query_grad_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "task_success_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "utility_tail_visit_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "utility_tail_grad_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "utility_tail_success_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "utility_tail_query_visit_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "utility_tail_query_grad_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "visit_coverage_at": {},
            "grad_coverage_at": {},
            "all_visit_coverage_at": {},
            "all_grad_coverage_at": {},
            "task_visit_coverage_at": {},
            "task_grad_coverage_at": {},
            "query_visit_coverage_at": {},
            "query_grad_coverage_at": {},
            "task_visit_ge5_at": {},
            "task_visit_entropy_at": {},
            "task_visit_gini_at": {},
            "time_to_full_visit": None,
            "time_to_full_grad": None,
            "task_time_to_visit": {"50": None, "75": None, "100": None},
            "task_time_to_grad": {"50": None, "75": None, "100": None},
            "history": [],
            "split_stats": split_stats or {
                "sibling_divergence": 0.0,
                "sibling_pairs": [],
                "mutated_children": [],
            },
        }

    def update(
        self,
        *,
        step: int,
        stage: GrowthStage,
        visit_counts: Tensor | None = None,
        gradient_norms: Tensor | None = None,
        all_visit_counts: Tensor | None = None,
        task_visit_counts: Tensor | None = None,
        query_visit_counts: Tensor | None = None,
        bootstrap_visit_counts: Tensor | None = None,
        all_gradient_signal: Tensor | None = None,
        task_gradient_signal: Tensor | None = None,
        query_gradient_signal: Tensor | None = None,
        bootstrap_gradient_signal: Tensor | None = None,
        success_visit_counts: Tensor | None = None,
    ) -> dict[str, float | int]:
        if self.current_stage_summary is None or self.current_stage_summary["stage_index"] != stage.index:
            self.start_stage(stage)
        summary = self.current_stage_summary
        assert summary is not None

        active_node_ids = list(summary["active_node_ids"])
        local_step = stage.local_step(step)
        if all_visit_counts is None:
            if visit_counts is None:
                raise ValueError("CoverageTracker.update requires visit counts.")
            all_visit_counts = visit_counts
        if task_visit_counts is None:
            task_visit_counts = all_visit_counts
        if query_visit_counts is None:
            query_visit_counts = torch.zeros_like(task_visit_counts)
        if bootstrap_visit_counts is None:
            bootstrap_visit_counts = (all_visit_counts - task_visit_counts).clamp_min(0.0)
        if all_gradient_signal is None:
            if gradient_norms is None:
                raise ValueError("CoverageTracker.update requires gradient norms or gradient signals.")
            all_gradient_signal = gradient_norms
        if task_gradient_signal is None:
            task_gradient_signal = all_gradient_signal
        if query_gradient_signal is None:
            query_gradient_signal = torch.zeros_like(task_gradient_signal)
        if bootstrap_gradient_signal is None:
            bootstrap_gradient_signal = (all_gradient_signal - task_gradient_signal).clamp_min(0.0)
        if success_visit_counts is None:
            success_visit_counts = torch.zeros_like(task_visit_counts)

        all_visit_counts = all_visit_counts.detach().cpu().to(torch.float64)
        task_visit_counts = task_visit_counts.detach().cpu().to(torch.float64)
        query_visit_counts = query_visit_counts.detach().cpu().to(torch.float64)
        bootstrap_visit_counts = bootstrap_visit_counts.detach().cpu().to(torch.float64)
        all_gradient_signal = all_gradient_signal.detach().cpu().to(torch.float64)
        task_gradient_signal = task_gradient_signal.detach().cpu().to(torch.float64)
        query_gradient_signal = query_gradient_signal.detach().cpu().to(torch.float64)
        bootstrap_gradient_signal = bootstrap_gradient_signal.detach().cpu().to(torch.float64)
        success_visit_counts = success_visit_counts.detach().cpu().to(torch.float64)

        summary["all_visit_histogram"] += all_visit_counts
        summary["task_visit_histogram"] += task_visit_counts
        summary["query_visit_histogram"] += query_visit_counts
        summary["bootstrap_visit_histogram"] += bootstrap_visit_counts
        summary["all_grad_histogram"] += all_gradient_signal
        summary["task_grad_histogram"] += task_gradient_signal
        summary["query_grad_histogram"] += query_gradient_signal
        summary["bootstrap_grad_histogram"] += bootstrap_gradient_signal
        decay = self.utility_ema_decay
        if not stage.bootstrap_active(step):
            summary["task_visit_ema"] = decay * summary["task_visit_ema"] + (1.0 - decay) * task_visit_counts
            summary["task_grad_ema"] = decay * summary["task_grad_ema"] + (1.0 - decay) * task_gradient_signal
            summary["query_visit_ema"] = decay * summary["query_visit_ema"] + (1.0 - decay) * query_visit_counts
            summary["query_grad_ema"] = decay * summary["query_grad_ema"] + (1.0 - decay) * query_gradient_signal
            summary["task_success_ema"] = decay * summary["task_success_ema"] + (1.0 - decay) * success_visit_counts
            if local_step >= int(summary["utility_tail_start_local_step"]):
                summary["utility_tail_visit_ema"] = decay * summary["utility_tail_visit_ema"] + (1.0 - decay) * task_visit_counts
                summary["utility_tail_grad_ema"] = decay * summary["utility_tail_grad_ema"] + (1.0 - decay) * task_gradient_signal
                summary["utility_tail_success_ema"] = decay * summary["utility_tail_success_ema"] + (1.0 - decay) * success_visit_counts
                summary["utility_tail_query_visit_ema"] = decay * summary["utility_tail_query_visit_ema"] + (1.0 - decay) * query_visit_counts
                summary["utility_tail_query_grad_ema"] = decay * summary["utility_tail_query_grad_ema"] + (1.0 - decay) * query_gradient_signal

        all_visit_fraction = self._coverage_fraction(summary["all_visit_histogram"], active_node_ids, threshold=0.0)
        task_visit_fraction = self._coverage_fraction(summary["task_visit_histogram"], active_node_ids, threshold=0.0)
        query_visit_fraction = self._coverage_fraction(summary["query_visit_histogram"], active_node_ids, threshold=0.0)
        all_grad_fraction = self._coverage_fraction(
            summary["all_grad_histogram"],
            active_node_ids,
            threshold=self.gradient_norm_threshold,
        )
        task_grad_fraction = self._coverage_fraction(
            summary["task_grad_histogram"],
            active_node_ids,
            threshold=self.gradient_norm_threshold,
        )
        query_grad_fraction = self._coverage_fraction(
            summary["query_grad_histogram"],
            active_node_ids,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_ge5_fraction = self._count_fraction(summary["task_visit_histogram"], active_node_ids, threshold=5.0)
        task_visit_entropy = self._normalized_entropy(summary["task_visit_histogram"], active_node_ids)
        task_visit_gini = self._gini(summary["task_visit_histogram"], active_node_ids)

        for checkpoint in self.CHECKPOINTS:
            key = str(checkpoint)
            if local_step == checkpoint:
                summary["visit_coverage_at"][key] = all_visit_fraction
                summary["grad_coverage_at"][key] = all_grad_fraction
                summary["all_visit_coverage_at"][key] = all_visit_fraction
                summary["all_grad_coverage_at"][key] = all_grad_fraction
                summary["task_visit_coverage_at"][key] = task_visit_fraction
                summary["task_grad_coverage_at"][key] = task_grad_fraction
                summary["query_visit_coverage_at"][key] = query_visit_fraction
                summary["query_grad_coverage_at"][key] = query_grad_fraction
                summary["task_visit_ge5_at"][key] = task_visit_ge5_fraction
                summary["task_visit_entropy_at"][key] = task_visit_entropy
                summary["task_visit_gini_at"][key] = task_visit_gini

        if summary["time_to_full_visit"] is None and all_visit_fraction == 1.0:
            summary["time_to_full_visit"] = local_step
        if summary["time_to_full_grad"] is None and all_grad_fraction == 1.0:
            summary["time_to_full_grad"] = local_step

        self._maybe_set_threshold_times(
            summary,
            local_step=local_step,
            metric_name="task_time_to_visit",
            value=task_visit_fraction,
        )
        self._maybe_set_threshold_times(
            summary,
            local_step=local_step,
            metric_name="task_time_to_grad",
            value=task_grad_fraction,
        )

        history_row = {
            "step": step,
            "stage_index": stage.index,
            "active_compute_nodes": len(active_node_ids),
            "stage_local_step": local_step,
            "bootstrap_active": int(stage.bootstrap_active(step)),
            "post_bootstrap_step": max(local_step - stage.bootstrap_steps, 0),
            "visit_coverage": all_visit_fraction,
            "gradient_coverage": all_grad_fraction,
            "all_visit_coverage": all_visit_fraction,
            "all_gradient_coverage": all_grad_fraction,
            "task_visit_coverage": task_visit_fraction,
            "task_gradient_coverage": task_grad_fraction,
            "query_visit_coverage": query_visit_fraction,
            "query_gradient_coverage": query_grad_fraction,
            "task_visit_ge5_fraction": task_visit_ge5_fraction,
            "task_visit_entropy": task_visit_entropy,
            "task_visit_gini": task_visit_gini,
        }
        self.history.append(history_row)
        summary["history"].append(history_row)
        return history_row

    def _finalize_stage(self) -> dict[str, Any]:
        assert self.current_stage_summary is not None
        summary = self.current_stage_summary
        active_node_ids: list[int] = list(summary["active_node_ids"])
        split_stats = summary["split_stats"]
        utility_alpha = float(split_stats.get("utility_alpha", 1.0))
        utility_visit_weight = float(split_stats.get("utility_visit_weight", 1.0))
        utility_grad_weight = float(split_stats.get("utility_grad_weight", 1.0))
        utility_query_visit_weight = float(split_stats.get("utility_query_visit_weight", 0.0))
        utility_query_grad_weight = float(split_stats.get("utility_query_grad_weight", 0.0))
        sibling_pairs: list[tuple[int, int]] = split_stats.get("sibling_pairs", [])
        mutated_children = set(split_stats.get("mutated_children", []))
        mutated_traffic_wins = 0
        mutated_grad_wins = 0
        mutated_usefulness_wins = 0
        mutated_children_with_traffic = 0
        child_visit_share: dict[int, float] = {}
        child_grad_share: dict[int, float] = {}
        child_usefulness_share: dict[int, float] = {}
        mutated_visit_shares: list[float] = []
        mutated_grad_shares: list[float] = []
        mutated_usefulness_shares: list[float] = []
        for left_child, right_child in sibling_pairs:
            left_visit = float(summary["task_visit_ema"][left_child - 1].item())
            right_visit = float(summary["task_visit_ema"][right_child - 1].item())
            total_visit = max(left_visit + right_visit, 1.0e-12)
            child_visit_share[left_child] = left_visit / total_visit
            child_visit_share[right_child] = right_visit / total_visit

            left_grad = float(summary["task_grad_ema"][left_child - 1].item())
            right_grad = float(summary["task_grad_ema"][right_child - 1].item())
            left_query_visit = float(summary["query_visit_ema"][left_child - 1].item())
            right_query_visit = float(summary["query_visit_ema"][right_child - 1].item())
            left_query_grad = float(summary["query_grad_ema"][left_child - 1].item())
            right_query_grad = float(summary["query_grad_ema"][right_child - 1].item())
            total_grad = max(left_grad + right_grad, 1.0e-12)
            child_grad_share[left_child] = left_grad / total_grad
            child_grad_share[right_child] = right_grad / total_grad

            left_usefulness = (
                utility_visit_weight * left_visit
                + utility_grad_weight * left_grad
                + utility_alpha * float(summary["task_success_ema"][left_child - 1].item())
                + utility_query_visit_weight * left_query_visit
                + utility_query_grad_weight * left_query_grad
            )
            right_usefulness = (
                utility_visit_weight * right_visit
                + utility_grad_weight * right_grad
                + utility_alpha * float(summary["task_success_ema"][right_child - 1].item())
                + utility_query_visit_weight * right_query_visit
                + utility_query_grad_weight * right_query_grad
            )
            total_usefulness = max(left_usefulness + right_usefulness, 1.0e-12)
            child_usefulness_share[left_child] = left_usefulness / total_usefulness
            child_usefulness_share[right_child] = right_usefulness / total_usefulness
            if right_child not in mutated_children:
                continue
            if summary["task_visit_histogram"][right_child - 1] > 0:
                mutated_children_with_traffic += 1
            if right_visit > left_visit:
                mutated_traffic_wins += 1
            if right_grad > left_grad:
                mutated_grad_wins += 1
            if right_usefulness > left_usefulness:
                mutated_usefulness_wins += 1
            mutated_visit_shares.append(child_visit_share[right_child])
            mutated_grad_shares.append(child_grad_share[right_child])
            mutated_usefulness_shares.append(child_usefulness_share[right_child])

        utility_parent_components = split_stats.get("parent_components", {})
        parent_to_children = split_stats.get("parent_to_children", {})
        eligible_parents = split_stats.get("eligible_parents", [])
        child_usefulness: dict[int, float] = {}
        for parent_id in eligible_parents:
            child_ids = parent_to_children.get(parent_id, [parent_id])
            usefulness = 0.0
            for child_id in child_ids:
                usefulness += utility_visit_weight * float(summary["task_visit_ema"][child_id - 1].item())
                usefulness += utility_grad_weight * float(summary["task_grad_ema"][child_id - 1].item())
                usefulness += utility_alpha * float(summary["task_success_ema"][child_id - 1].item())
                usefulness += utility_query_visit_weight * float(summary["query_visit_ema"][child_id - 1].item())
                usefulness += utility_query_grad_weight * float(summary["query_grad_ema"][child_id - 1].item())
            child_usefulness[parent_id] = usefulness

        selected = split_stats.get("selected_parents", [])
        unselected = split_stats.get("unselected_parents", [])
        selected_usefulness = [child_usefulness[parent_id] for parent_id in selected]
        unselected_usefulness = [child_usefulness[parent_id] for parent_id in unselected]
        selected_parent_utility = [
            float(utility_parent_components.get(parent_id, {}).get("score", 0.0)) for parent_id in selected
        ]
        unselected_parent_utility = [
            float(utility_parent_components.get(parent_id, {}).get("score", 0.0)) for parent_id in unselected
        ]
        parent_utility = [
            float(utility_parent_components.get(parent_id, {}).get("score", 0.0)) for parent_id in eligible_parents
        ]
        later_usefulness = [child_usefulness[parent_id] for parent_id in eligible_parents]
        utility_usefulness_correlation = 0.0
        if len(parent_utility) >= 2:
            utility_tensor = torch.tensor(parent_utility, dtype=torch.float64)
            usefulness_tensor = torch.tensor(later_usefulness, dtype=torch.float64)
            utility_std = float(utility_tensor.std(unbiased=False).item())
            usefulness_std = float(usefulness_tensor.std(unbiased=False).item())
            if utility_std > 1.0e-12 and usefulness_std > 1.0e-12:
                utility_usefulness_correlation = float(
                    ((utility_tensor - utility_tensor.mean()) * (usefulness_tensor - usefulness_tensor.mean())).mean()
                    / (utility_std * usefulness_std)
                )

        active = len(active_node_ids)

        return {
            "stage_index": summary["stage_index"],
            "stage_name": summary["stage_name"],
            "active_compute_nodes": active,
            "active_node_ids": active_node_ids,
            "start_step": summary["start_step"],
            "bootstrap_steps": summary["bootstrap_steps"],
            "visit_coverage_at": summary["visit_coverage_at"],
            "grad_coverage_at": summary["grad_coverage_at"],
            "all_visit_coverage_at": summary["all_visit_coverage_at"],
            "all_grad_coverage_at": summary["all_grad_coverage_at"],
            "task_visit_coverage_at": summary["task_visit_coverage_at"],
            "task_grad_coverage_at": summary["task_grad_coverage_at"],
            "query_visit_coverage_at": summary["query_visit_coverage_at"],
            "query_grad_coverage_at": summary["query_grad_coverage_at"],
            "task_visit_ge5_at": summary["task_visit_ge5_at"],
            "task_visit_entropy_at": summary["task_visit_entropy_at"],
            "task_visit_gini_at": summary["task_visit_gini_at"],
            "time_to_full_visit": summary["time_to_full_visit"],
            "time_to_full_grad": summary["time_to_full_grad"],
            "task_time_to_visit": summary["task_time_to_visit"],
            "task_time_to_grad": summary["task_time_to_grad"],
            "post_bootstrap_visit_slope": self._post_bootstrap_slope(summary["history"], "task_visit_coverage"),
            "post_bootstrap_grad_slope": self._post_bootstrap_slope(summary["history"], "task_gradient_coverage"),
            "visit_histogram": self._active_view(summary["all_visit_histogram"], active_node_ids).tolist(),
            "grad_histogram": self._active_view(summary["all_grad_histogram"], active_node_ids).tolist(),
            "all_visit_histogram": self._active_view(summary["all_visit_histogram"], active_node_ids).tolist(),
            "task_visit_histogram": self._active_view(summary["task_visit_histogram"], active_node_ids).tolist(),
            "query_visit_histogram": self._active_view(summary["query_visit_histogram"], active_node_ids).tolist(),
            "bootstrap_visit_histogram": self._active_view(summary["bootstrap_visit_histogram"], active_node_ids).tolist(),
            "all_grad_histogram": self._active_view(summary["all_grad_histogram"], active_node_ids).tolist(),
            "task_grad_histogram": self._active_view(summary["task_grad_histogram"], active_node_ids).tolist(),
            "query_grad_histogram": self._active_view(summary["query_grad_histogram"], active_node_ids).tolist(),
            "bootstrap_grad_histogram": self._active_view(summary["bootstrap_grad_histogram"], active_node_ids).tolist(),
            "visit_ema": self._active_view(summary["task_visit_ema"], active_node_ids).tolist(),
            "grad_ema": self._active_view(summary["task_grad_ema"], active_node_ids).tolist(),
            "task_visit_ema": self._active_view(summary["task_visit_ema"], active_node_ids).tolist(),
            "task_grad_ema": self._active_view(summary["task_grad_ema"], active_node_ids).tolist(),
            "query_visit_ema": self._active_view(summary["query_visit_ema"], active_node_ids).tolist(),
            "query_grad_ema": self._active_view(summary["query_grad_ema"], active_node_ids).tolist(),
            "task_success_ema": self._active_view(summary["task_success_ema"], active_node_ids).tolist(),
            "utility_tail_visit_ema": self._active_view(summary["utility_tail_visit_ema"], active_node_ids).tolist(),
            "utility_tail_grad_ema": self._active_view(summary["utility_tail_grad_ema"], active_node_ids).tolist(),
            "utility_tail_success_ema": self._active_view(summary["utility_tail_success_ema"], active_node_ids).tolist(),
            "utility_tail_query_visit_ema": self._active_view(summary["utility_tail_query_visit_ema"], active_node_ids).tolist(),
            "utility_tail_query_grad_ema": self._active_view(summary["utility_tail_query_grad_ema"], active_node_ids).tolist(),
            "split_stats": {
                "sibling_divergence": split_stats.get("sibling_divergence", 0.0),
                "sibling_pairs": sibling_pairs,
                "mutated_children": split_stats.get("mutated_children", []),
                "mutated_parent_ids": split_stats.get("mutated_parent_ids", []),
                "mutated_children_with_traffic": mutated_children_with_traffic,
                "mutated_child_more_traffic_pairs": mutated_traffic_wins,
                "mutated_child_more_gradient_pairs": mutated_grad_wins,
                "mutated_child_more_usefulness_pairs": mutated_usefulness_wins,
                "mutated_child_usefulness_win_rate": (
                    float(mutated_usefulness_wins / len(mutated_usefulness_shares)) if mutated_usefulness_shares else 0.0
                ),
                "mutated_child_visit_share_mean": (
                    float(torch.tensor(mutated_visit_shares, dtype=torch.float64).mean().item()) if mutated_visit_shares else 0.0
                ),
                "mutated_child_grad_share_mean": (
                    float(torch.tensor(mutated_grad_shares, dtype=torch.float64).mean().item()) if mutated_grad_shares else 0.0
                ),
                "mutated_child_usefulness_share_mean": (
                    float(torch.tensor(mutated_usefulness_shares, dtype=torch.float64).mean().item()) if mutated_usefulness_shares else 0.0
                ),
                "selection_policy": split_stats.get("selection_policy"),
                "mutation_stage_index_min": split_stats.get("mutation_stage_index_min"),
                "mutation_selected_fraction": split_stats.get("mutation_selected_fraction"),
                "mutation_score_margin": split_stats.get("mutation_score_margin"),
                "mutation_min_visit_z": split_stats.get("mutation_min_visit_z"),
                "mutation_min_query_grad_z": split_stats.get("mutation_min_query_grad_z"),
                "mutation_requires_stagnation": split_stats.get("mutation_requires_stagnation"),
                "mutation_stagnated": split_stats.get("mutation_stagnated"),
                "mutation_stagnation_window": split_stats.get("mutation_stagnation_window"),
                "mutation_stagnation_delta": split_stats.get("mutation_stagnation_delta"),
                "mutation_stage_val_tail": split_stats.get("mutation_stage_val_tail"),
                "mutation_stage_val_range": split_stats.get("mutation_stage_val_range"),
                "mutation_reference_score": split_stats.get("mutation_reference_score"),
                "mutation_reference_kind": split_stats.get("mutation_reference_kind"),
                "eligible_parents": eligible_parents,
                "selected_parents": selected,
                "unselected_parents": unselected,
                "selected_parent_scores": split_stats.get("selected_parent_scores", {}),
                "unselected_parent_scores": split_stats.get("unselected_parent_scores", {}),
                "selected_parent_utility_mean": float(torch.tensor(selected_parent_utility, dtype=torch.float64).mean().item()) if selected_parent_utility else 0.0,
                "unselected_parent_utility_mean": float(torch.tensor(unselected_parent_utility, dtype=torch.float64).mean().item()) if unselected_parent_utility else 0.0,
                "selected_parent_child_usefulness_mean": float(torch.tensor(selected_usefulness, dtype=torch.float64).mean().item()) if selected_usefulness else 0.0,
                "unselected_parent_child_usefulness_mean": float(torch.tensor(unselected_usefulness, dtype=torch.float64).mean().item()) if unselected_usefulness else 0.0,
                "utility_usefulness_correlation": utility_usefulness_correlation,
                "parent_child_usefulness": child_usefulness,
                "parent_components": utility_parent_components,
                "parent_to_child": split_stats.get("parent_to_child", {}),
                "parent_to_children": parent_to_children,
                "child_intervals": split_stats.get("child_intervals", {}),
                "child_visit_share": child_visit_share,
                "child_grad_share": child_grad_share,
                "child_usefulness_share": child_usefulness_share,
            },
        }

    def finalize(self) -> None:
        if self.current_stage_summary is not None:
            self.completed_stages.append(self._finalize_stage())
            self.current_stage_summary = None

    def current_snapshot(self) -> dict[str, float | int]:
        if self.current_stage_summary is None:
            return {}
        summary = self.current_stage_summary
        active_node_ids: list[int] = list(summary["active_node_ids"])
        visit_fraction = self._coverage_fraction(summary["all_visit_histogram"], active_node_ids, threshold=0.0)
        grad_fraction = self._coverage_fraction(
            summary["all_grad_histogram"],
            active_node_ids,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_fraction = self._coverage_fraction(summary["task_visit_histogram"], active_node_ids, threshold=0.0)
        task_grad_fraction = self._coverage_fraction(
            summary["task_grad_histogram"],
            active_node_ids,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_ge5_fraction = self._count_fraction(summary["task_visit_histogram"], active_node_ids, threshold=5.0)
        task_visit_entropy = self._normalized_entropy(summary["task_visit_histogram"], active_node_ids)
        task_visit_gini = self._gini(summary["task_visit_histogram"], active_node_ids)
        return {
            "active_node_visit_coverage": visit_fraction,
            "active_node_gradient_coverage": grad_fraction,
            "task_node_visit_coverage": task_visit_fraction,
            "task_node_gradient_coverage": task_grad_fraction,
            "task_nodes_ge5_visit_fraction": task_visit_ge5_fraction,
            "task_visit_entropy": task_visit_entropy,
            "task_visit_gini": task_visit_gini,
            "active_compute_nodes": len(active_node_ids),
            "stage_index": int(summary["stage_index"]),
        }

    def selection_components(
        self,
        topology: GrowthTopology,
        *,
        utility_alpha: float,
        utility_visit_weight: float = 1.0,
        utility_grad_weight: float = 1.0,
        utility_query_visit_weight: float = 0.0,
        utility_query_grad_weight: float = 0.0,
    ) -> dict[int, dict[str, float]]:
        if self.current_stage_summary is None:
            return {}
        summary = self.current_stage_summary
        active_node_ids = set(summary["active_node_ids"])
        visit_values = {
            node_id: float(summary["utility_tail_visit_ema"][node_id - 1].item())
            for node_id in topology.eligible_split_parents()
            if node_id in active_node_ids
        }
        grad_values = {
            node_id: float(summary["utility_tail_grad_ema"][node_id - 1].item())
            for node_id in visit_values
        }
        success_values = {
            node_id: float(summary["utility_tail_success_ema"][node_id - 1].item())
            for node_id in visit_values
        }
        query_visit_values = {
            node_id: float(summary["utility_tail_query_visit_ema"][node_id - 1].item())
            for node_id in visit_values
        }
        query_grad_values = {
            node_id: float(summary["utility_tail_query_grad_ema"][node_id - 1].item())
            for node_id in visit_values
        }
        visit_z = _zscore_dict(visit_values)
        grad_z = _zscore_dict(grad_values)
        success_z = _zscore_dict(success_values)
        query_visit_z = _zscore_dict(query_visit_values)
        query_grad_z = _zscore_dict(query_grad_values)
        components: dict[int, dict[str, float]] = {}
        for node_id in visit_values:
            score = (
                float(utility_visit_weight) * visit_z.get(node_id, 0.0)
                + float(utility_grad_weight) * grad_z.get(node_id, 0.0)
                + float(utility_alpha) * success_z.get(node_id, 0.0)
                + float(utility_query_visit_weight) * query_visit_z.get(node_id, 0.0)
                + float(utility_query_grad_weight) * query_grad_z.get(node_id, 0.0)
            )
            components[node_id] = {
                "visit": visit_values[node_id],
                "grad": grad_values[node_id],
                "success": success_values[node_id],
                "query_visit": query_visit_values[node_id],
                "query_grad": query_grad_values[node_id],
                "visit_z": visit_z.get(node_id, 0.0),
                "grad_z": grad_z.get(node_id, 0.0),
                "success_z": success_z.get(node_id, 0.0),
                "query_visit_z": query_visit_z.get(node_id, 0.0),
                "query_grad_z": query_grad_z.get(node_id, 0.0),
                "score": score,
            }
        return components

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "history": self.history,
            "stages": self.completed_stages,
        }
