from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.model import APSGNNModel, OutputEvent


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def small_config(source: str = "ambiguity_gate") -> ExperimentConfig:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = source
    config.model.d_model = 8
    config.model.num_classes = 4
    config.model.key_dim = 4
    config.model.use_reserved_class_slice = True
    return config


def test_v88_generator_emits_ambig_and_sharpblend_pack() -> None:
    module = load_module("gen_v88_configs", "scripts/gen_v88_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"ambig", "sharpblend"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v88_collision_sharpened_summary_blend"


def test_v88_configs_select_sources_cleanly() -> None:
    module = load_module("gen_v88_cfg", "scripts/gen_v88_configs.py")
    ambig_cfg = module.build_collision_config("c2", "m", condition="ambig")
    sharp_cfg = module.build_collision_config("c2", "m", condition="sharpblend")
    assert ambig_cfg["model"]["cache_output_summary_source"] == "ambiguity_gate"
    assert sharp_cfg["model"]["cache_output_summary_source"] == "collision_sharpened_blend"
    assert sharp_cfg["model"]["cache_output_gate_feature_scale"] == 1.0
    assert sharp_cfg["model"]["cache_output_collision_switch_width"] == 2.0
    assert sharp_cfg["model"]["cache_output_collision_blend_scale"] == 1.0


def test_v88_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_gate_feature_proj is None
    assert model.cache_output_collision_blend_proj is None


def test_v88_sharpblend_mode_uses_gate_and_blend_proj() -> None:
    model = APSGNNModel(small_config("collision_sharpened_blend"))
    assert model.cache_output_gate_feature_proj is not None
    assert model.cache_output_gate_feature_proj.in_features == 2
    assert model.cache_output_collision_blend_proj is not None
    assert model.cache_output_aux_proj is None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_classslice_feature_proj is None


def test_v88_output_logits_uses_cache_summary_when_enabled() -> None:
    config = small_config("ambiguity_gate")
    config.model.use_reserved_class_slice = False
    model = APSGNNModel(config)
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()
        assert model.cache_output_summary_head is not None
        assert model.cache_output_summary_gate is not None
        final_linear = model.cache_output_summary_head[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.weight.zero_()
        final_linear.bias.fill_(1.0)
        model.cache_output_summary_gate.weight.zero_()
        model.cache_output_summary_gate.bias.fill_(10.0)
    residual = torch.zeros(2, config.model.d_model)
    cache_summary = torch.randn(2, config.model.d_model)
    logits = model._output_logits(residual, cache_summary=cache_summary)
    assert torch.all(logits > 0.99)


def test_v88_sharpblend_matches_v75_when_blend_proj_is_zero() -> None:
    ambig_cfg = small_config("ambiguity_gate")
    ambig_cfg.model.use_reserved_class_slice = False
    sharp_cfg = small_config("collision_sharpened_blend")
    sharp_cfg.model.use_reserved_class_slice = False
    ambig = APSGNNModel(ambig_cfg)
    sharp = APSGNNModel(sharp_cfg)
    with torch.no_grad():
        ambig.output_head.weight.zero_()
        ambig.output_head.bias.zero_()
        sharp.output_head.weight.zero_()
        sharp.output_head.bias.zero_()
        assert ambig.cache_output_summary_head is not None
        assert sharp.cache_output_summary_head is not None
        ambig_final = ambig.cache_output_summary_head[-1]
        sharp_final = sharp.cache_output_summary_head[-1]
        assert isinstance(ambig_final, torch.nn.Linear)
        assert isinstance(sharp_final, torch.nn.Linear)
        ambig_final.weight.zero_()
        ambig_final.bias.fill_(1.0)
        sharp_final.weight.copy_(ambig_final.weight)
        sharp_final.bias.copy_(ambig_final.bias)
        assert ambig.cache_output_summary_gate is not None
        assert sharp.cache_output_summary_gate is not None
        ambig.cache_output_summary_gate.weight.zero_()
        ambig.cache_output_summary_gate.bias.zero_()
        sharp.cache_output_summary_gate.weight.zero_()
        sharp.cache_output_summary_gate.bias.zero_()
        assert ambig.cache_output_gate_feature_proj is not None
        assert sharp.cache_output_gate_feature_proj is not None
        ambig.cache_output_gate_feature_proj.weight.zero_()
        ambig.cache_output_gate_feature_proj.bias.zero_()
        ambig.cache_output_gate_feature_proj.weight[0, 0] = 1.0
        sharp.cache_output_gate_feature_proj.weight.zero_()
        sharp.cache_output_gate_feature_proj.bias.zero_()
        sharp.cache_output_gate_feature_proj.weight[0, 0] = 1.0
    residual = torch.zeros(2, ambig_cfg.model.d_model)
    cache_summary = torch.ones(2, ambig_cfg.model.d_model)
    retrieved_summary = torch.full((2, ambig_cfg.model.d_model), 2.0)
    gate_features = torch.tensor([[0.8, 3.0 / 48.0], [0.8, 4.0 / 48.0]], dtype=residual.dtype)
    ambig_logits = ambig._output_logits(
        residual,
        cache_summary=cache_summary,
        retrieved_summary=retrieved_summary,
        gate_features=gate_features,
    )
    sharp_logits = sharp._output_logits(
        residual,
        cache_summary=cache_summary,
        retrieved_summary=retrieved_summary,
        gate_features=gate_features,
    )
    assert torch.allclose(ambig_logits, sharp_logits)


def test_v88_sharpblend_only_changes_high_collision_summary() -> None:
    cfg = small_config("collision_sharpened_blend")
    model = APSGNNModel(cfg)
    assert model.cache_output_collision_blend_proj is not None
    with torch.no_grad():
        model.cache_output_collision_blend_proj.weight.zero_()
        model.cache_output_collision_blend_proj.bias.fill_(1.0)
    cache_summary = torch.zeros(2, cfg.model.d_model)
    retrieved_summary = torch.ones(2, cfg.model.d_model)
    low_collision = torch.tensor([[0.8, 1.0 / 48.0], [0.8, 1.25 / 48.0]], dtype=cache_summary.dtype)
    high_collision = torch.tensor([[0.8, 4.0 / 48.0], [0.8, 5.0 / 48.0]], dtype=cache_summary.dtype)
    low_blend = model._blend_output_cache_summary(
        cache_summary,
        retrieved_summary,
        low_collision,
    )
    high_blend = model._blend_output_cache_summary(
        cache_summary,
        retrieved_summary,
        high_collision,
    )
    assert torch.all(high_blend >= low_blend)
    assert torch.any(high_blend > low_blend)


def test_v88_output_event_carries_two_feature_gate_payload() -> None:
    event = OutputEvent(
        residual=torch.randn(3, 8),
        batch_index=torch.tensor([0, 1, 2]),
        role=torch.tensor([0, 1, 1]),
        target_label=torch.tensor([0, 1, 2]),
        hop_index=torch.tensor([1, 2, 3]),
        packet_id=torch.tensor([10, 11, 12]),
        cache_summary=torch.randn(3, 8),
        aux_summary=torch.randn(3, 8),
        retrieved_summary=torch.randn(3, 8),
        gate_features=torch.randn(3, 2),
    )
    selected = event.select(torch.tensor([True, False, True]))
    assert selected.cache_summary is not None
    assert selected.aux_summary is not None
    assert selected.gate_features is not None
    assert selected.gate_features.shape == (2, 2)


def test_v88_report_parser_accepts_v88_run_names() -> None:
    module = load_module("build_v88_report", "scripts/build_v88_report.py")
    parsed = module.parse_run_name(
        "20260327-040000-v88-collision-c2-sharpblend-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "sharpblend"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
