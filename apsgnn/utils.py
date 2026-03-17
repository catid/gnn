from __future__ import annotations

import json
import os
import random
import socket
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from apsgnn.config import ExperimentConfig

matplotlib.use("Agg")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_run_dir(root: str | Path, run_name: str) -> Path:
    return ensure_dir(Path(root) / f"{timestamp()}-{run_name}")


def save_json(data: dict[str, Any], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def git_info() -> dict[str, str]:
    info: dict[str, str] = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).strip()
        info["status"] = subprocess.check_output(
            ["git", "status", "--short"],
            text=True,
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        info["commit"] = "unavailable"
        info["status"] = "unavailable"
    return info


def environment_info() -> dict[str, Any]:
    return {
        "python": sys.version,
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "bf16_supported": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "git": git_info(),
    }


class MetricsWriter:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.rows: list[dict[str, Any]] = []
        self.jsonl_path = run_dir / "metrics.jsonl"

    def append(self, row: dict[str, Any]) -> None:
        self.rows.append(row)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    def flush_csv(self) -> Path:
        csv_path = self.run_dir / "metrics.csv"
        pd.DataFrame(self.rows).to_csv(csv_path, index=False)
        return csv_path


def save_run_metadata(run_dir: Path, config: ExperimentConfig) -> None:
    save_json(config.to_dict(), run_dir / "config.json")
    save_json(environment_info(), run_dir / "environment.json")


def plot_metrics(csv_path: str | Path, output_prefix: str | Path) -> list[Path]:
    df = pd.read_csv(csv_path)
    created: list[Path] = []
    output_prefix = Path(output_prefix)
    if df.empty:
        return created

    if {"step", "train/loss"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        plt.plot(df["step"], df["train/loss"], label="train loss")
        if "val/loss" in df.columns:
            valid = df["val/loss"].notna()
            plt.plot(df.loc[valid, "step"], df.loc[valid, "val/loss"], label="val loss")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        path = output_prefix.with_name(output_prefix.name + "_loss.png")
        plt.savefig(path, dpi=150)
        plt.close()
        created.append(path)

    if {"step", "val/query_accuracy"}.issubset(df.columns):
        plt.figure(figsize=(6, 4))
        valid = df["val/query_accuracy"].notna()
        plt.plot(df.loc[valid, "step"], df.loc[valid, "val/query_accuracy"], label="val accuracy")
        if "val/query_delivery_rate" in df.columns:
            plt.plot(df.loc[valid, "step"], df.loc[valid, "val/query_delivery_rate"], label="val delivery")
        plt.xlabel("Step")
        plt.ylabel("Rate")
        plt.ylim(0.0, 1.05)
        plt.legend()
        plt.tight_layout()
        path = output_prefix.with_name(output_prefix.name + "_accuracy.png")
        plt.savefig(path, dpi=150)
        plt.close()
        created.append(path)

    return created


def summarize_config(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
