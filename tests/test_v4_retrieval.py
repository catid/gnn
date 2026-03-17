from __future__ import annotations

import torch

from apsgnn.config import ExperimentConfig, load_config
from apsgnn.model import APSGNNModel
from apsgnn.tasks import MemoryRoutingTask
from apsgnn.train import freeze_first_hop_router, maybe_initialize_from_checkpoint


def _v4_config(*, variant: str, enable_cache: bool = True) -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.nodes_total = 8
    config.model.cache_capacity = 8
    config.model.enable_cache = enable_cache
    config.model.use_first_hop_key_hint = False
    config.model.use_learned_first_hop_router = True
    config.model.first_hop_router_variant = "key_mlp_ce"
    config.model.first_hop_router_hidden_dim = 64
    config.model.first_hop_router_layers = 2
    config.model.first_hop_router_use_residual = False
    config.model.first_hop_router_separate_heads = True
    config.model.first_hop_router_aux_type = "none"
    config.model.cache_read_variant = variant
    config.model.cache_read_hidden_dim = 64
    config.model.cache_read_layers = 2
    config.task.writers_per_episode = 2
    config.task.max_rollout_steps = 4
    config.train.batch_size_per_gpu = 2
    return config


def test_v4_configs_disable_old_explicit_cache_read(monkeypatch) -> None:
    for path in ("configs/v4_retrieval_implicit_search.yaml", "configs/v4_retrieval_keycond_search.yaml"):
        config = load_config(path)
        assert config.model.cache_read_variant != "explicit"
        assert not config.model.use_first_hop_key_hint
        assert config.model.use_learned_first_hop_router
        assert config.train.freeze_first_hop_router

        model = APSGNNModel(config)
        task = MemoryRoutingTask(config)
        batch = task.generate(batch_size=1, seed=0).to(torch.device("cpu"))

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("Explicit raw-key cache read should not be used in v4 configs.")

        monkeypatch.setattr(model, "_explicit_cache_read", fail_if_called)
        model.eval()
        model(batch)


def test_v4_implicit_retrieval_query_depends_only_on_hidden_state() -> None:
    model = APSGNNModel(_v4_config(variant="learned_implicit"))
    assert model.cache_retriever is not None

    hidden = torch.randn(2, model.config.model.d_model)
    cache_entries = torch.randn(2, 3, model.config.model.d_model)
    cache_mask = torch.tensor([[True, True, False], [True, False, False]])
    routing_key_a = torch.randn(2, model.config.model.key_dim)
    routing_key_b = torch.randn(2, model.config.model.key_dim)

    outputs_a = model.cache_retriever(
        hidden=hidden,
        routing_key=routing_key_a,
        cache_entries=cache_entries,
        cache_mask=cache_mask,
    )
    outputs_b = model.cache_retriever(
        hidden=hidden,
        routing_key=routing_key_b,
        cache_entries=cache_entries,
        cache_mask=cache_mask,
    )

    assert torch.allclose(outputs_a["update"], outputs_b["update"])
    assert torch.allclose(outputs_a["attention_weights"], outputs_b["attention_weights"])


def test_v4_key_conditioned_retrieval_uses_learned_key_features() -> None:
    model = APSGNNModel(_v4_config(variant="learned_keycond"))
    assert model.cache_retriever is not None
    assert model.cache_retriever.uses_key_conditioning
    assert model.cache_retriever.key_condition_proj is not None

    with torch.no_grad():
        model.cache_retriever.key_condition_proj.weight.fill_(0.25)
        model.cache_retriever.key_condition_proj.bias.zero_()

    hidden = torch.randn(2, model.config.model.d_model)
    cache_entries = torch.randn(2, 3, model.config.model.d_model)
    cache_mask = torch.tensor([[True, True, True], [True, True, False]])
    routing_key_a = torch.zeros(2, model.config.model.key_dim)
    routing_key_b = torch.ones(2, model.config.model.key_dim)

    outputs_a = model.cache_retriever(
        hidden=hidden,
        routing_key=routing_key_a,
        cache_entries=cache_entries,
        cache_mask=cache_mask,
    )
    outputs_b = model.cache_retriever(
        hidden=hidden,
        routing_key=routing_key_b,
        cache_entries=cache_entries,
        cache_mask=cache_mask,
    )

    assert not torch.allclose(outputs_a["update"], outputs_b["update"])
    assert not torch.allclose(outputs_a["attention_weights"], outputs_b["attention_weights"])


def test_v4_cached_and_no_cache_configs_share_router_setup() -> None:
    cached = load_config("configs/v4_retrieval_best.yaml")
    no_cache = load_config("configs/v4_retrieval_best_no_cache.yaml")

    assert cached.model.use_learned_first_hop_router
    assert no_cache.model.use_learned_first_hop_router
    assert cached.model.first_hop_router_variant == no_cache.model.first_hop_router_variant
    assert cached.model.first_hop_router_hidden_dim == no_cache.model.first_hop_router_hidden_dim
    assert cached.model.first_hop_router_layers == no_cache.model.first_hop_router_layers
    assert cached.model.first_hop_router_use_residual == no_cache.model.first_hop_router_use_residual
    assert cached.model.first_hop_router_separate_heads == no_cache.model.first_hop_router_separate_heads
    assert cached.model.first_hop_router_aux_type == no_cache.model.first_hop_router_aux_type
    assert cached.model.cache_read_variant == no_cache.model.cache_read_variant
    assert cached.model.cache_read_hidden_dim == no_cache.model.cache_read_hidden_dim
    assert cached.model.cache_read_layers == no_cache.model.cache_read_layers
    assert cached.train.init_checkpoint == no_cache.train.init_checkpoint
    assert cached.train.freeze_first_hop_router == no_cache.train.freeze_first_hop_router
    assert cached.model.enable_cache
    assert not no_cache.model.enable_cache


def test_v4_warm_start_preserves_first_hop_router_weights(tmp_path) -> None:
    source = APSGNNModel(_v4_config(variant="explicit"))
    target = APSGNNModel(_v4_config(variant="learned_keycond"))
    source_parameter = dict(source.named_parameters())["first_hop_router.key_proj.weight"]
    target_parameter = dict(target.named_parameters())["first_hop_router.key_proj.weight"]
    with torch.no_grad():
        source_parameter.fill_(0.75)
        target_parameter.zero_()

    checkpoint_path = tmp_path / "v3_like.pt"
    torch.save({"model": source.state_dict(), "optimizer": {}}, checkpoint_path)

    maybe_initialize_from_checkpoint(target, str(checkpoint_path), torch.device("cpu"))
    loaded_parameter = dict(target.named_parameters())["first_hop_router.key_proj.weight"]

    assert torch.allclose(loaded_parameter, torch.full_like(loaded_parameter, 0.75))
    assert target.cache_retriever is not None


def test_freeze_first_hop_router_leaves_router_weights_frozen() -> None:
    model = APSGNNModel(_v4_config(variant="learned_keycond"))
    freeze_first_hop_router(model)

    frozen_names = {
        name
        for name, parameter in model.named_parameters()
        if (name.startswith("first_hop_router") or name.startswith("first_hop_router_ln")) and not parameter.requires_grad
    }
    assert frozen_names
