from __future__ import annotations

import torch

from apsgnn.buffer import NodeCache
from apsgnn.config import ExperimentConfig, load_config
from apsgnn.model import APSGNNModel, PacketBatch, ROLE_QUERY, ROLE_WRITER
from apsgnn.tasks import MemoryRoutingTask


def _v2_config() -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.nodes_total = 8
    config.model.cache_capacity = 8
    config.model.use_first_hop_key_hint = False
    config.model.use_learned_first_hop_router = True
    config.task.writers_per_episode = 2
    config.task.max_rollout_steps = 4
    config.train.batch_size_per_gpu = 2
    return config


def test_v2_config_path_disables_frozen_first_hop_hint(monkeypatch) -> None:
    config = load_config("configs/v2_learned_router.yaml")
    assert not config.model.use_first_hop_key_hint
    assert config.model.use_learned_first_hop_router

    model = APSGNNModel(config)
    task = MemoryRoutingTask(config)
    batch = task.generate(batch_size=1, seed=0).to(torch.device("cpu"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Frozen first-hop hint should not be used in v2.")

    monkeypatch.setattr(model, "_first_hop_key_hint", fail_if_called)
    model.eval()
    model(batch)


def test_first_hop_router_returns_valid_compute_node_destinations() -> None:
    config = _v2_config()
    model = APSGNNModel(config)
    model.eval()
    device = torch.device("cpu")
    cache = NodeCache(
        batch_size=2,
        nodes_total=config.model.nodes_total,
        capacity=config.model.cache_capacity,
        d_model=config.model.d_model,
        device=device,
        dtype=torch.float32,
        enabled=True,
    )
    packets = PacketBatch(
        residual=torch.randn(2, config.model.d_model),
        routing_key=torch.randn(2, config.model.key_dim),
        ttl=torch.tensor([1, 4], dtype=torch.long),
        batch_index=torch.tensor([0, 1], dtype=torch.long),
        current_node=torch.tensor([1, 2], dtype=torch.long),
        role=torch.tensor([ROLE_WRITER, ROLE_QUERY], dtype=torch.long),
        target_label=torch.tensor([0, 1], dtype=torch.long),
        target_home=torch.tensor([3, 4], dtype=torch.long),
        hop_index=torch.zeros(2, dtype=torch.long),
        has_visited_home=torch.zeros(2, dtype=torch.bool),
        packet_id=torch.tensor([0, 1], dtype=torch.long),
    )

    predictions = model._run_compute_nodes(packets, cache)

    assert predictions["route_logits"].shape == (2, config.model.nodes_total)
    assert int(predictions["dest_index"].min().item()) >= 1
    assert int(predictions["dest_index"].max().item()) < config.model.nodes_total
    assert torch.equal(predictions["dest_index"], predictions["predicted_dest_index"])
    assert not bool(predictions["teacher_forced_mask"].any().item())


def test_teacher_forced_first_hop_uses_target_destination(monkeypatch) -> None:
    config = _v2_config()
    model = APSGNNModel(config)
    model.train()
    model.set_first_hop_teacher_force_ratio(1.0)
    device = torch.device("cpu")
    cache = NodeCache(
        batch_size=1,
        nodes_total=config.model.nodes_total,
        capacity=config.model.cache_capacity,
        d_model=config.model.d_model,
        device=device,
        dtype=torch.float32,
        enabled=True,
    )

    def fake_router(_features: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, config.model.num_compute_nodes), -10.0)
        logits[:, 0] = 10.0
        return logits

    monkeypatch.setattr(model.first_hop_router, "forward", fake_router)
    packets = PacketBatch(
        residual=torch.zeros(1, config.model.d_model),
        routing_key=torch.zeros(1, config.model.key_dim),
        ttl=torch.tensor([4], dtype=torch.long),
        batch_index=torch.tensor([0], dtype=torch.long),
        current_node=torch.tensor([1], dtype=torch.long),
        role=torch.tensor([ROLE_QUERY], dtype=torch.long),
        target_label=torch.tensor([0], dtype=torch.long),
        target_home=torch.tensor([3], dtype=torch.long),
        hop_index=torch.zeros(1, dtype=torch.long),
        has_visited_home=torch.zeros(1, dtype=torch.bool),
        packet_id=torch.tensor([0], dtype=torch.long),
    )

    predictions = model._run_compute_nodes(packets, cache)

    assert int(predictions["predicted_dest_index"].item()) == 1
    assert int(predictions["dest_index"].item()) == 3
    assert bool(predictions["teacher_forced_mask"].item())


def test_v2_cached_and_no_cache_configs_share_first_hop_router_settings() -> None:
    cached = load_config("configs/v2_learned_router.yaml")
    no_cache = load_config("configs/v2_learned_router_no_cache.yaml")

    assert cached.model.use_learned_first_hop_router
    assert no_cache.model.use_learned_first_hop_router
    assert not cached.model.use_first_hop_key_hint
    assert not no_cache.model.use_first_hop_key_hint
    assert cached.train.first_hop_teacher_force_start == no_cache.train.first_hop_teacher_force_start
    assert cached.train.first_hop_teacher_force_end == no_cache.train.first_hop_teacher_force_end
    assert cached.train.first_hop_teacher_force_anneal_steps == no_cache.train.first_hop_teacher_force_anneal_steps
    assert cached.model.enable_cache
    assert not no_cache.model.enable_cache
