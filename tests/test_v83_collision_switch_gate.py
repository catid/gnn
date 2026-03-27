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


def test_v83_generator_emits_ambig_and_collswitch_pack() -> None:
    module = load_module("gen_v83_configs", "scripts/gen_v83_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"ambig", "collswitch"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v83_collision_switch_gate"


def test_v83_configs_select_sources_cleanly() -> None:
    module = load_module("gen_v83_cfg", "scripts/gen_v83_configs.py")
    ambig_cfg = module.build_collision_config("c2", "m", condition="ambig")
    switch_cfg = module.build_collision_config("c2", "m", condition="collswitch")
    assert ambig_cfg["model"]["cache_output_summary_source"] == "ambiguity_gate"
    assert switch_cfg["model"]["cache_output_summary_source"] == "collision_switch_gate"
    assert switch_cfg["model"]["cache_output_gate_feature_scale"] == 1.0
    assert switch_cfg["model"]["cache_output_collision_switch_width"] == 2.0


def test_v83_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_gate_feature_proj is None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_classslice_feature_proj is None


def test_v83_collswitch_mode_uses_gate_feature_proj_only() -> None:
    model = APSGNNModel(small_config("collision_switch_gate"))
    assert model.cache_output_gate_feature_proj is not None
    assert model.cache_output_gate_feature_proj.in_features == 2
    assert model.cache_output_aux_proj is None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_classslice_feature_proj is None
    assert model.cache_output_feature_scale_proj is None
    assert model.cache_output_feature_shift_proj is None


def test_v83_output_logits_uses_cache_summary_when_enabled() -> None:
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


def test_v83_collision_switch_matches_v75_when_collision_is_high() -> None:
    ambig_cfg = small_config("ambiguity_gate")
    ambig_cfg.model.use_reserved_class_slice = False
    switch_cfg = small_config("collision_switch_gate")
    switch_cfg.model.use_reserved_class_slice = False
    switch_cfg.model.cache_output_collision_switch_width = 2.0
    ambig = APSGNNModel(ambig_cfg)
    switch = APSGNNModel(switch_cfg)
    switch.load_state_dict(ambig.state_dict(), strict=False)
    with torch.no_grad():
        assert ambig.cache_output_summary_gate is not None
        assert switch.cache_output_summary_gate is not None
        ambig.cache_output_summary_gate.weight.zero_()
        ambig.cache_output_summary_gate.bias.zero_()
        switch.cache_output_summary_gate.weight.zero_()
        switch.cache_output_summary_gate.bias.zero_()
        assert ambig.cache_output_gate_feature_proj is not None
        assert switch.cache_output_gate_feature_proj is not None
        ambig.cache_output_gate_feature_proj.weight.zero_()
        ambig.cache_output_gate_feature_proj.bias.zero_()
        ambig.cache_output_gate_feature_proj.weight[0, 0] = 1.0
        switch.cache_output_gate_feature_proj.weight.copy_(ambig.cache_output_gate_feature_proj.weight)
        switch.cache_output_gate_feature_proj.bias.copy_(ambig.cache_output_gate_feature_proj.bias)
    residual = torch.zeros(2, ambig_cfg.model.d_model)
    cache_summary = torch.ones(2, ambig_cfg.model.d_model)
    high_collision = torch.tensor([[0.8, 3.0 / 48.0], [0.8, 4.0 / 48.0]], dtype=residual.dtype)
    ambig_logits = ambig._output_logits(residual, cache_summary=cache_summary, gate_features=high_collision)
    switch_logits = switch._output_logits(residual, cache_summary=cache_summary, gate_features=high_collision)
    assert torch.allclose(ambig_logits, switch_logits)


def test_v83_collision_switch_reduces_feature_delta_when_entries_are_low() -> None:
    switch_cfg = small_config("collision_switch_gate")
    switch = APSGNNModel(switch_cfg)
    gate_delta = torch.ones(2, 1)
    low_collision = torch.tensor([[0.8, 1.0 / 48.0], [0.8, 1.5 / 48.0]], dtype=torch.float32)
    switched = switch._apply_output_collision_switch(gate_delta, low_collision)
    assert torch.all(switched < gate_delta)


def test_v83_output_event_carries_two_feature_gate_payload() -> None:
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


def test_v83_report_parser_accepts_v83_run_names() -> None:
    module = load_module("build_v83_report", "scripts/build_v83_report.py")
    parsed = module.parse_run_name(
        "20260327-040000-v83-collision-c2-collswitch-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "collswitch"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
