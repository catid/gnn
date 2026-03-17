from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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


def project_home_leaves(home_leaf: Tensor, active_compute_nodes: int, final_compute_nodes: int) -> Tensor:
    if final_compute_nodes % active_compute_nodes != 0:
        raise ValueError("final_compute_nodes must be divisible by active_compute_nodes.")
    bucket_size = final_compute_nodes // active_compute_nodes
    return 1 + (home_leaf - 1) // bucket_size


def clockwise_successor(current_node: Tensor, active_compute_nodes: int) -> Tensor:
    successor = current_node.clone()
    compute_mask = current_node > 0
    successor[compute_mask] = ((current_node[compute_mask] - 1 + 1) % active_compute_nodes) + 1
    return successor


def active_node_ids(active_compute_nodes: int) -> Tensor:
    return torch.arange(1, active_compute_nodes + 1, dtype=torch.long)


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
    def __init__(
        self,
        *,
        num_compute_nodes: int,
        gradient_norm_threshold: float,
        utility_ema_decay: float,
    ) -> None:
        self.num_compute_nodes = num_compute_nodes
        self.gradient_norm_threshold = gradient_norm_threshold
        self.utility_ema_decay = utility_ema_decay
        self.completed_stages: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.current_stage_summary: dict[str, Any] | None = None

    def start_stage(self, stage: GrowthStage, *, split_stats: dict[str, Any] | None = None) -> None:
        if self.current_stage_summary is not None:
            self.completed_stages.append(self._finalize_stage())
        self.current_stage_summary = {
            "stage_index": stage.index,
            "stage_name": stage.name,
            "active_compute_nodes": stage.active_compute_nodes,
            "start_step": stage.start_step,
            "visit_seen": torch.zeros(self.num_compute_nodes, dtype=torch.bool),
            "grad_seen": torch.zeros(self.num_compute_nodes, dtype=torch.bool),
            "visit_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "grad_histogram": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "visit_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "grad_ema": torch.zeros(self.num_compute_nodes, dtype=torch.float64),
            "visit_coverage_at": {},
            "grad_coverage_at": {},
            "time_to_full_visit": None,
            "time_to_full_grad": None,
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
        visit_counts: Tensor,
        gradient_norms: Tensor,
    ) -> dict[str, float | int]:
        if self.current_stage_summary is None or self.current_stage_summary["stage_index"] != stage.index:
            self.start_stage(stage)
        summary = self.current_stage_summary
        assert summary is not None

        active = stage.active_compute_nodes
        local_step = stage.local_step(step)
        visit_counts = visit_counts.detach().cpu().to(torch.float64)
        gradient_norms = gradient_norms.detach().cpu().to(torch.float64)

        summary["visit_seen"][:active] |= visit_counts[:active] > 0
        summary["grad_seen"][:active] |= gradient_norms[:active] > self.gradient_norm_threshold
        summary["visit_histogram"][:active] += visit_counts[:active]
        summary["grad_histogram"][:active] += gradient_norms[:active]
        decay = self.utility_ema_decay
        summary["visit_ema"][:active] = decay * summary["visit_ema"][:active] + (1.0 - decay) * visit_counts[:active]
        summary["grad_ema"][:active] = decay * summary["grad_ema"][:active] + (1.0 - decay) * gradient_norms[:active]

        visit_fraction = float(summary["visit_seen"][:active].to(torch.float32).mean().item())
        grad_fraction = float(summary["grad_seen"][:active].to(torch.float32).mean().item())

        for checkpoint in (10, 50, 100):
            key = str(checkpoint)
            if local_step == checkpoint:
                summary["visit_coverage_at"][key] = visit_fraction
                summary["grad_coverage_at"][key] = grad_fraction

        if summary["time_to_full_visit"] is None and visit_fraction == 1.0:
            summary["time_to_full_visit"] = local_step
        if summary["time_to_full_grad"] is None and grad_fraction == 1.0:
            summary["time_to_full_grad"] = local_step

        history_row = {
            "step": step,
            "stage_index": stage.index,
            "active_compute_nodes": active,
            "stage_local_step": local_step,
            "visit_coverage": visit_fraction,
            "gradient_coverage": grad_fraction,
        }
        self.history.append(history_row)
        return history_row

    def _finalize_stage(self) -> dict[str, Any]:
        assert self.current_stage_summary is not None
        summary = self.current_stage_summary
        active = summary["active_compute_nodes"]
        split_stats = summary["split_stats"]
        sibling_pairs: list[tuple[int, int]] = split_stats.get("sibling_pairs", [])
        mutated_children = set(split_stats.get("mutated_children", []))
        mutated_traffic_wins = 0
        mutated_grad_wins = 0
        mutated_children_with_traffic = 0
        for left_child, right_child in sibling_pairs:
            if right_child not in mutated_children:
                continue
            if summary["visit_histogram"][right_child - 1] > 0:
                mutated_children_with_traffic += 1
            if summary["visit_ema"][right_child - 1] > summary["visit_ema"][left_child - 1]:
                mutated_traffic_wins += 1
            if summary["grad_ema"][right_child - 1] > summary["grad_ema"][left_child - 1]:
                mutated_grad_wins += 1

        return {
            "stage_index": summary["stage_index"],
            "stage_name": summary["stage_name"],
            "active_compute_nodes": active,
            "start_step": summary["start_step"],
            "visit_coverage_at": summary["visit_coverage_at"],
            "grad_coverage_at": summary["grad_coverage_at"],
            "time_to_full_visit": summary["time_to_full_visit"],
            "time_to_full_grad": summary["time_to_full_grad"],
            "visit_histogram": summary["visit_histogram"][:active].tolist(),
            "grad_histogram": summary["grad_histogram"][:active].tolist(),
            "visit_ema": summary["visit_ema"][:active].tolist(),
            "grad_ema": summary["grad_ema"][:active].tolist(),
            "split_stats": {
                "sibling_divergence": split_stats.get("sibling_divergence", 0.0),
                "sibling_pairs": sibling_pairs,
                "mutated_children": split_stats.get("mutated_children", []),
                "mutated_children_with_traffic": mutated_children_with_traffic,
                "mutated_child_more_traffic_pairs": mutated_traffic_wins,
                "mutated_child_more_gradient_pairs": mutated_grad_wins,
            },
        }

    def finalize(self) -> None:
        if self.current_stage_summary is not None:
            self.completed_stages.append(self._finalize_stage())
            self.current_stage_summary = None

    def current_snapshot(self) -> dict[str, float | int]:
        if self.current_stage_summary is None:
            return {}
        active = self.current_stage_summary["active_compute_nodes"]
        visit_fraction = float(self.current_stage_summary["visit_seen"][:active].to(torch.float32).mean().item())
        grad_fraction = float(self.current_stage_summary["grad_seen"][:active].to(torch.float32).mean().item())
        return {
            "active_node_visit_coverage": visit_fraction,
            "active_node_gradient_coverage": grad_fraction,
            "active_compute_nodes": active,
            "stage_index": int(self.current_stage_summary["stage_index"]),
        }

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "history": self.history,
            "stages": self.completed_stages,
        }
