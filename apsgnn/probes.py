from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass
class ProbeResult:
    train_accuracy: float
    valid_accuracy: float
    test_accuracy: float


def _accuracy(logits: Tensor, labels: Tensor) -> float:
    if logits.numel() == 0 or labels.numel() == 0:
        return 0.0
    return float((logits.argmax(dim=-1) == labels).to(torch.float32).mean().item())


def fit_linear_probe(
    *,
    train_x: Tensor,
    train_y: Tensor,
    valid_x: Tensor,
    valid_y: Tensor,
    test_x: Tensor,
    test_y: Tensor,
    num_classes: int,
    steps: int = 200,
    lr: float = 0.05,
    weight_decay: float = 1.0e-4,
) -> ProbeResult:
    if train_x.ndim != 2:
        raise ValueError("train_x must be rank-2")
    if train_x.size(0) != train_y.size(0):
        raise ValueError("train_x/train_y length mismatch")
    if train_x.size(1) != valid_x.size(1) or train_x.size(1) != test_x.size(1):
        raise ValueError("probe feature dimensions must match")

    device = train_x.device
    model = nn.Linear(train_x.size(1), num_classes, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_valid = float("-inf")

    for _ in range(max(int(steps), 1)):
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            valid_acc = _accuracy(model(valid_x), valid_y)
            if valid_acc >= best_valid:
                best_valid = valid_acc
                best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        return ProbeResult(
            train_accuracy=_accuracy(model(train_x), train_y),
            valid_accuracy=_accuracy(model(valid_x), valid_y),
            test_accuracy=_accuracy(model(test_x), test_y),
        )


def hard_slice_summary(
    rows: list[dict[str, Any]],
    *,
    difficulty_key: str,
    ambiguity_key: str,
    correct_key: str,
    hard_difficulty_threshold: float,
    hard_ambiguity_threshold: float,
) -> dict[str, float]:
    if not rows:
        return {
            "base_accuracy": 0.0,
            "hard_accuracy": 0.0,
            "base_count": 0.0,
            "hard_count": 0.0,
        }

    base_rows = [
        row
        for row in rows
        if float(row.get(difficulty_key, 0.0)) < hard_difficulty_threshold
        or float(row.get(ambiguity_key, 0.0)) < hard_ambiguity_threshold
    ]
    hard_rows = [
        row
        for row in rows
        if float(row.get(difficulty_key, 0.0)) >= hard_difficulty_threshold
        and float(row.get(ambiguity_key, 0.0)) >= hard_ambiguity_threshold
    ]

    def accuracy(subset: list[dict[str, Any]]) -> float:
        if not subset:
            return 0.0
        return float(sum(float(row.get(correct_key, 0.0)) for row in subset) / len(subset))

    return {
        "base_accuracy": accuracy(base_rows),
        "hard_accuracy": accuracy(hard_rows),
        "base_count": float(len(base_rows)),
        "hard_count": float(len(hard_rows)),
    }


def bucketed_accuracy(
    rows: list[dict[str, Any]],
    *,
    bucket_key: str,
    correct_key: str,
) -> list[dict[str, float]]:
    buckets: dict[int, list[float]] = {}
    for row in rows:
        bucket = int(round(float(row.get(bucket_key, 0.0))))
        buckets.setdefault(bucket, []).append(float(row.get(correct_key, 0.0)))
    return [
        {
            "bucket": float(bucket),
            "accuracy": float(sum(values) / len(values)),
            "count": float(len(values)),
        }
        for bucket, values in sorted(buckets.items())
    ]
