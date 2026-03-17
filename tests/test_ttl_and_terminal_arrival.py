from __future__ import annotations

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.model import APSGNNModel
from apsgnn.tasks import SanityBatch


def _tiny_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.nodes_total = 4
    config.model.cache_capacity = 4
    config.task.name = "sanity"
    config.task.max_rollout_steps = 3
    config.train.batch_size_per_gpu = 1
    return config


def test_ttl_expiry_schedules_terminal_cache_write(monkeypatch) -> None:
    config = _tiny_config()
    model = APSGNNModel(config)
    model.eval()

    writes: list[tuple[list[float], list[int], list[int]]] = []

    def fake_run_compute_nodes(*_args, **_kwargs):
        return {
            "next_residual": torch.tensor([[3.0] * config.model.d_model]),
            "route_logits": torch.zeros(1, config.model.nodes_total),
            "delay_logits": torch.zeros(1, config.model.delay_bins),
            "dest_index": torch.tensor([2]),
            "delay_index": torch.tensor([0]),
            "address_norm": torch.zeros(1),
        }

    def record_write(self, residuals, batch_index, node_index):
        writes.append(
            (
                residuals[0, :4].tolist(),
                batch_index.tolist(),
                node_index.tolist(),
            )
        )

    monkeypatch.setattr(model, "_run_compute_nodes", fake_run_compute_nodes)
    monkeypatch.setattr("apsgnn.buffer.NodeCache.write", record_write)

    batch = SanityBatch(
        inputs=torch.zeros(1, config.model.d_model),
        start_nodes=torch.tensor([1]),
        ttl=torch.tensor([1]),
    )
    model(batch)

    assert writes == [([3.0, 3.0, 3.0, 3.0], [0], [2])]


def test_output_sink_terminates_packets(monkeypatch) -> None:
    config = _tiny_config()
    model = APSGNNModel(config)
    model.eval()

    live_calls: list[int] = []
    cache_calls: list[int] = []
    output_calls: list[int] = []

    def fake_run_compute_nodes(*_args, **_kwargs):
        return {
            "next_residual": torch.ones(1, config.model.d_model),
            "route_logits": torch.zeros(1, config.model.nodes_total),
            "delay_logits": torch.zeros(1, config.model.delay_bins),
            "dest_index": torch.tensor([0]),
            "delay_index": torch.tensor([0]),
            "address_norm": torch.zeros(1),
        }

    def capture_live(*args, **kwargs):
        live_calls.append(1)

    def capture_cache(*args, **kwargs):
        cache_calls.append(1)

    def capture_output(*args, **kwargs):
        output_calls.append(1)

    monkeypatch.setattr(model, "_run_compute_nodes", fake_run_compute_nodes)
    monkeypatch.setattr(model, "_schedule_packets", capture_live)
    monkeypatch.setattr(model, "_schedule_cache_events", capture_cache)
    monkeypatch.setattr(model, "_schedule_output_events", capture_output)

    batch = SanityBatch(
        inputs=torch.zeros(1, config.model.d_model),
        start_nodes=torch.tensor([1]),
        ttl=torch.tensor([4]),
    )
    model(batch)

    assert output_calls == [1]
    assert live_calls == []
    assert cache_calls == []
