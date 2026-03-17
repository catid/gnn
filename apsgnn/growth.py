from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
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


def default_split_stats() -> dict[str, Any]:
    return {
        "sibling_divergence": 0.0,
        "sibling_pairs": [],
        "mutated_children": [],
    }


def transition_model_for_growth(
    model: APSGNNModel,
    previous_active_compute_nodes: int,
    next_active_compute_nodes: int,
    *,
    transition_mode: str,
    split_mode: str,
    mutation_scale: float,
    seed: int,
) -> dict[str, Any]:
    if next_active_compute_nodes <= previous_active_compute_nodes:
        return default_split_stats()
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
    ) -> None:
        self.num_compute_nodes = num_compute_nodes
        self.gradient_norm_threshold = gradient_norm_threshold
        self.utility_ema_decay = utility_ema_decay
        self.completed_stages: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.current_stage_summary: dict[str, Any] | None = None

    @staticmethod
    def _coverage_fraction(values: Tensor, active: int, *, threshold: float) -> float:
        if active <= 0:
            return 0.0
        return float((values[:active] > threshold).to(torch.float32).mean().item())

    @staticmethod
    def _count_fraction(values: Tensor, active: int, *, threshold: float) -> float:
        if active <= 0:
            return 0.0
        return float((values[:active] >= threshold).to(torch.float32).mean().item())

    @staticmethod
    def _normalized_entropy(values: Tensor, active: int) -> float:
        trimmed = values[:active].clamp_min(0.0)
        total = float(trimmed.sum().item())
        if total <= 0.0 or active <= 1:
            return 0.0
        probabilities = trimmed / total
        safe = probabilities.clamp_min(1.0e-12)
        entropy = -(probabilities * safe.log()).sum()
        return float((entropy / math.log(active)).item())

    @staticmethod
    def _gini(values: Tensor, active: int) -> float:
        trimmed = values[:active].clamp_min(0.0)
        total = float(trimmed.sum().item())
        if total <= 0.0 or active <= 1:
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

    def start_stage(self, stage: GrowthStage, *, split_stats: dict[str, Any] | None = None) -> None:
        if self.current_stage_summary is not None:
            self.completed_stages.append(self._finalize_stage())
        self.current_stage_summary = {
            "stage_index": stage.index,
            "stage_name": stage.name,
            "active_compute_nodes": stage.active_compute_nodes,
            "start_step": stage.start_step,
            "bootstrap_steps": stage.bootstrap_steps,
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
    ) -> dict[str, float | int]:
        if self.current_stage_summary is None or self.current_stage_summary["stage_index"] != stage.index:
            self.start_stage(stage)
        summary = self.current_stage_summary
        assert summary is not None

        active = stage.active_compute_nodes
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

        all_visit_counts = all_visit_counts.detach().cpu().to(torch.float64)
        task_visit_counts = task_visit_counts.detach().cpu().to(torch.float64)
        query_visit_counts = query_visit_counts.detach().cpu().to(torch.float64)
        bootstrap_visit_counts = bootstrap_visit_counts.detach().cpu().to(torch.float64)
        all_gradient_signal = all_gradient_signal.detach().cpu().to(torch.float64)
        task_gradient_signal = task_gradient_signal.detach().cpu().to(torch.float64)
        query_gradient_signal = query_gradient_signal.detach().cpu().to(torch.float64)
        bootstrap_gradient_signal = bootstrap_gradient_signal.detach().cpu().to(torch.float64)

        summary["all_visit_histogram"][:active] += all_visit_counts[:active]
        summary["task_visit_histogram"][:active] += task_visit_counts[:active]
        summary["query_visit_histogram"][:active] += query_visit_counts[:active]
        summary["bootstrap_visit_histogram"][:active] += bootstrap_visit_counts[:active]
        summary["all_grad_histogram"][:active] += all_gradient_signal[:active]
        summary["task_grad_histogram"][:active] += task_gradient_signal[:active]
        summary["query_grad_histogram"][:active] += query_gradient_signal[:active]
        summary["bootstrap_grad_histogram"][:active] += bootstrap_gradient_signal[:active]
        decay = self.utility_ema_decay
        summary["task_visit_ema"][:active] = (
            decay * summary["task_visit_ema"][:active] + (1.0 - decay) * task_visit_counts[:active]
        )
        summary["task_grad_ema"][:active] = (
            decay * summary["task_grad_ema"][:active] + (1.0 - decay) * task_gradient_signal[:active]
        )

        all_visit_fraction = self._coverage_fraction(summary["all_visit_histogram"], active, threshold=0.0)
        task_visit_fraction = self._coverage_fraction(summary["task_visit_histogram"], active, threshold=0.0)
        query_visit_fraction = self._coverage_fraction(summary["query_visit_histogram"], active, threshold=0.0)
        all_grad_fraction = self._coverage_fraction(
            summary["all_grad_histogram"],
            active,
            threshold=self.gradient_norm_threshold,
        )
        task_grad_fraction = self._coverage_fraction(
            summary["task_grad_histogram"],
            active,
            threshold=self.gradient_norm_threshold,
        )
        query_grad_fraction = self._coverage_fraction(
            summary["query_grad_histogram"],
            active,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_ge5_fraction = self._count_fraction(summary["task_visit_histogram"], active, threshold=5.0)
        task_visit_entropy = self._normalized_entropy(summary["task_visit_histogram"], active)
        task_visit_gini = self._gini(summary["task_visit_histogram"], active)

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
            "active_compute_nodes": active,
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
            if summary["task_visit_histogram"][right_child - 1] > 0:
                mutated_children_with_traffic += 1
            if summary["task_visit_ema"][right_child - 1] > summary["task_visit_ema"][left_child - 1]:
                mutated_traffic_wins += 1
            if summary["task_grad_ema"][right_child - 1] > summary["task_grad_ema"][left_child - 1]:
                mutated_grad_wins += 1

        return {
            "stage_index": summary["stage_index"],
            "stage_name": summary["stage_name"],
            "active_compute_nodes": active,
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
            "visit_histogram": summary["all_visit_histogram"][:active].tolist(),
            "grad_histogram": summary["all_grad_histogram"][:active].tolist(),
            "all_visit_histogram": summary["all_visit_histogram"][:active].tolist(),
            "task_visit_histogram": summary["task_visit_histogram"][:active].tolist(),
            "query_visit_histogram": summary["query_visit_histogram"][:active].tolist(),
            "bootstrap_visit_histogram": summary["bootstrap_visit_histogram"][:active].tolist(),
            "all_grad_histogram": summary["all_grad_histogram"][:active].tolist(),
            "task_grad_histogram": summary["task_grad_histogram"][:active].tolist(),
            "query_grad_histogram": summary["query_grad_histogram"][:active].tolist(),
            "bootstrap_grad_histogram": summary["bootstrap_grad_histogram"][:active].tolist(),
            "visit_ema": summary["task_visit_ema"][:active].tolist(),
            "grad_ema": summary["task_grad_ema"][:active].tolist(),
            "task_visit_ema": summary["task_visit_ema"][:active].tolist(),
            "task_grad_ema": summary["task_grad_ema"][:active].tolist(),
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
        summary = self.current_stage_summary
        active = summary["active_compute_nodes"]
        visit_fraction = self._coverage_fraction(summary["all_visit_histogram"], active, threshold=0.0)
        grad_fraction = self._coverage_fraction(
            summary["all_grad_histogram"],
            active,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_fraction = self._coverage_fraction(summary["task_visit_histogram"], active, threshold=0.0)
        task_grad_fraction = self._coverage_fraction(
            summary["task_grad_histogram"],
            active,
            threshold=self.gradient_norm_threshold,
        )
        task_visit_ge5_fraction = self._count_fraction(summary["task_visit_histogram"], active, threshold=5.0)
        task_visit_entropy = self._normalized_entropy(summary["task_visit_histogram"], active)
        task_visit_gini = self._gini(summary["task_visit_histogram"], active)
        return {
            "active_node_visit_coverage": visit_fraction,
            "active_node_gradient_coverage": grad_fraction,
            "task_node_visit_coverage": task_visit_fraction,
            "task_node_gradient_coverage": task_grad_fraction,
            "task_nodes_ge5_visit_fraction": task_visit_ge5_fraction,
            "task_visit_entropy": task_visit_entropy,
            "task_visit_gini": task_visit_gini,
            "active_compute_nodes": active,
            "stage_index": int(summary["stage_index"]),
        }

    def to_dict(self) -> dict[str, Any]:
        self.finalize()
        return {
            "history": self.history,
            "stages": self.completed_stages,
        }
