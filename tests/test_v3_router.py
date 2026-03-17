from __future__ import annotations

import torch

from apsgnn.buffer import NodeCache
from apsgnn.config import ExperimentConfig, load_config
from apsgnn.model import APSGNNModel, KeyCentricFirstHopRouter, PacketBatch, ROLE_QUERY, ROLE_WRITER
from apsgnn.tasks import MemoryRoutingTask


def _v3_config(*, aux_type: str = "none") -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.nodes_total = 8
    config.model.cache_capacity = 8
    config.model.use_first_hop_key_hint = False
    config.model.use_learned_first_hop_router = True
    config.model.first_hop_router_variant = "key_mlp_aux" if aux_type != "none" else "key_mlp_ce"
    config.model.first_hop_router_hidden_dim = 64
    config.model.first_hop_router_layers = 2
    config.model.first_hop_router_use_residual = False
    config.model.first_hop_router_separate_heads = True
    config.model.first_hop_router_aux_type = aux_type
    config.task.writers_per_episode = 2
    config.task.max_rollout_steps = 4
    config.train.batch_size_per_gpu = 2
    config.train.first_hop_router_aux_weight = 0.25 if aux_type != "none" else 0.0
    return config


def _build_cache(config: ExperimentConfig, batch_size: int) -> NodeCache:
    return NodeCache(
        batch_size=batch_size,
        nodes_total=config.model.nodes_total,
        capacity=config.model.cache_capacity,
        d_model=config.model.d_model,
        device=torch.device("cpu"),
        dtype=torch.float32,
        enabled=True,
    )


def _build_packets(config: ExperimentConfig) -> PacketBatch:
    return PacketBatch(
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


def test_v3_config_path_disables_frozen_first_hop_hint(monkeypatch) -> None:
    config = load_config("configs/v3_router_ce_search.yaml")
    assert not config.model.use_first_hop_key_hint
    assert config.model.use_learned_first_hop_router
    assert config.model.first_hop_router_variant != "legacy"

    model = APSGNNModel(config)
    task = MemoryRoutingTask(config)
    batch = task.generate(batch_size=1, seed=0).to(torch.device("cpu"))

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("Frozen first-hop hint should not be used in v3.")

    monkeypatch.setattr(model, "_first_hop_key_hint", fail_if_called)
    model.eval()
    model(batch)


def test_v3_first_hop_router_returns_valid_compute_node_destinations() -> None:
    config = _v3_config()
    model = APSGNNModel(config)
    model.eval()
    predictions = model._run_compute_nodes(_build_packets(config), _build_cache(config, batch_size=2))

    assert predictions["route_logits"].shape == (2, config.model.nodes_total)
    assert predictions["first_hop_aux_prediction"].shape == (2, config.model.address_dim)
    assert int(predictions["dest_index"].min().item()) >= 1
    assert int(predictions["dest_index"].max().item()) < config.model.nodes_total
    assert torch.equal(predictions["dest_index"], predictions["predicted_dest_index"])
    assert not bool(predictions["teacher_forced_mask"].any().item())


def test_v3_separate_writer_and_query_heads_are_used() -> None:
    config = _v3_config()
    model = APSGNNModel(config)
    assert isinstance(model.first_hop_router, KeyCentricFirstHopRouter)

    def writer_forward(_hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, config.model.num_compute_nodes), -10.0)
        logits[:, 1] = 10.0
        return logits

    def query_forward(_hidden: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, config.model.num_compute_nodes), -10.0)
        logits[:, 3] = 10.0
        return logits

    model.first_hop_router.writer_head.forward = writer_forward  # type: ignore[method-assign]
    model.first_hop_router.query_head.forward = query_forward  # type: ignore[method-assign]

    outputs = model.first_hop_router(
        routing_key=torch.zeros(2, config.model.key_dim),
        aux_features=torch.zeros(2, config.model.d_model),
        residual=torch.zeros(2, config.model.d_model),
        role=torch.tensor([ROLE_WRITER, ROLE_QUERY], dtype=torch.long),
    )

    assert int(outputs["logits"][0].argmax().item()) == 1
    assert int(outputs["logits"][1].argmax().item()) == 3


def test_teacher_forcing_changes_executed_first_hop_only_during_training(monkeypatch) -> None:
    config = _v3_config()
    model = APSGNNModel(config)
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

    def fake_router(**_kwargs) -> dict[str, torch.Tensor]:
        logits = torch.full((1, config.model.num_compute_nodes), -10.0)
        logits[:, 0] = 10.0
        return {
            "logits": logits,
            "hidden": torch.zeros(1, config.model.first_hop_router_hidden_dim),
        }

    monkeypatch.setattr(model.first_hop_router, "forward", fake_router)

    model.train()
    model.set_first_hop_teacher_force_ratio(1.0)
    predictions = model._run_compute_nodes(packets, _build_cache(config, batch_size=1))
    assert int(predictions["predicted_dest_index"].item()) == 1
    assert int(predictions["dest_index"].item()) == 3
    assert bool(predictions["teacher_forced_mask"].item())

    model.eval()
    predictions_eval = model._run_compute_nodes(packets, _build_cache(config, batch_size=1))
    assert int(predictions_eval["predicted_dest_index"].item()) == 1
    assert int(predictions_eval["dest_index"].item()) == 1
    assert not bool(predictions_eval["teacher_forced_mask"].item())


def test_v3_auxiliary_address_matches_target_node_address_shape() -> None:
    config = _v3_config(aux_type="address_l2")
    model = APSGNNModel(config)
    model.eval()
    packets = _build_packets(config)
    predictions = model._run_compute_nodes(packets, _build_cache(config, batch_size=2))

    expected_targets = model.address_table[packets.target_home]
    assert model.uses_first_hop_router_aux()
    assert predictions["first_hop_aux_prediction"].shape == expected_targets.shape
    assert expected_targets.shape == (2, config.model.address_dim)
    assert not torch.allclose(expected_targets, torch.zeros_like(expected_targets))


def test_v3_cached_and_no_cache_configs_share_router_settings() -> None:
    cached = load_config("configs/v3_router_best.yaml")
    no_cache = load_config("configs/v3_router_best_no_cache.yaml")

    assert cached.model.use_learned_first_hop_router
    assert no_cache.model.use_learned_first_hop_router
    assert not cached.model.use_first_hop_key_hint
    assert not no_cache.model.use_first_hop_key_hint
    assert cached.model.first_hop_router_variant == no_cache.model.first_hop_router_variant
    assert cached.model.first_hop_router_hidden_dim == no_cache.model.first_hop_router_hidden_dim
    assert cached.model.first_hop_router_layers == no_cache.model.first_hop_router_layers
    assert cached.model.first_hop_router_use_residual == no_cache.model.first_hop_router_use_residual
    assert cached.model.first_hop_router_separate_heads == no_cache.model.first_hop_router_separate_heads
    assert cached.model.first_hop_router_aux_type == no_cache.model.first_hop_router_aux_type
    assert cached.train.first_hop_teacher_force_start == no_cache.train.first_hop_teacher_force_start
    assert cached.train.first_hop_teacher_force_end == no_cache.train.first_hop_teacher_force_end
    assert cached.train.first_hop_teacher_force_anneal_steps == no_cache.train.first_hop_teacher_force_anneal_steps
    assert cached.train.first_hop_router_aux_weight == no_cache.train.first_hop_router_aux_weight
    assert cached.model.enable_cache
    assert not no_cache.model.enable_cache
