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


def test_v79_generator_emits_ambig_and_ambigbase_pack() -> None:
    module = load_module("gen_v79_configs", "scripts/gen_v79_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"ambig", "ambigbase"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v79_ambiguity_conditioned_base_suppression"


def test_v79_configs_select_sources_cleanly() -> None:
    module = load_module("gen_v79_cfg", "scripts/gen_v79_configs.py")
    ambig_cfg = module.build_collision_config("c2", "m", condition="ambig")
    ambigbase_cfg = module.build_collision_config("c2", "m", condition="ambigbase")
    assert ambig_cfg["model"]["cache_output_summary_source"] == "ambiguity_gate"
    assert ambigbase_cfg["model"]["cache_output_summary_source"] == "ambiguity_base_suppression"
    assert ambigbase_cfg["model"]["cache_output_gate_feature_scale"] == 1.0
    assert ambigbase_cfg["model"]["cache_output_base_suppression_scale"] == 1.0


def test_v79_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_aux_proj is None
    assert model.cache_output_gate_feature_proj is None
    assert model.cache_output_base_feature_proj is None


def test_v79_base_suppression_mode_uses_gate_and_base_feature_projection() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "ambiguity_base_suppression"
    model = APSGNNModel(config)
    assert model.cache_output_gate_feature_proj is not None
    assert model.cache_output_base_feature_proj is not None
    assert model.cache_output_aux_proj is None
    assert model.cache_output_feature_scale_proj is None
    assert model.cache_output_feature_shift_proj is None


def test_v79_output_logits_uses_cache_summary_when_enabled() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.use_reserved_class_slice = False
    config.model.num_classes = 4
    config.model.key_dim = 4
    config.model.d_model = 8
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


def test_v79_base_suppression_defaults_to_v75_behavior() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "ambiguity_base_suppression"
    config.model.use_reserved_class_slice = False
    config.model.num_classes = 4
    config.model.key_dim = 4
    config.model.d_model = 8
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
        model.cache_output_summary_gate.bias.zero_()
    residual = torch.zeros(2, config.model.d_model)
    cache_summary = torch.randn(2, config.model.d_model)
    gate_features = torch.randn(2, 2)
    logits = model._output_logits(
        residual,
        cache_summary=cache_summary,
        gate_features=gate_features,
    )
    assert torch.all(logits > 0.49)


def test_v79_base_suppression_changes_logits_when_trained() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "ambiguity_base_suppression"
    config.model.cache_output_base_suppression_scale = 1.5
    config.model.d_model = 4
    config.model.num_classes = 3
    config.model.key_dim = 3
    config.model.use_reserved_class_slice = False
    model = APSGNNModel(config)
    assert model.cache_output_base_feature_proj is not None
    with torch.no_grad():
        model.output_head.weight.fill_(1.0)
        model.output_head.bias.zero_()
        assert model.cache_output_summary_head is not None
        assert model.cache_output_summary_gate is not None
        final_linear = model.cache_output_summary_head[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        model.cache_output_summary_gate.weight.zero_()
        model.cache_output_summary_gate.bias.zero_()
        model.cache_output_base_feature_proj.weight.zero_()
        model.cache_output_base_feature_proj.bias.fill_(-0.4)
    residual = torch.ones(2, config.model.d_model)
    cache_summary = torch.zeros(2, config.model.d_model)
    gate_features = torch.ones(2, 2)
    base = model._output_logits(residual, cache_summary=cache_summary)
    shifted = model._output_logits(
        residual,
        cache_summary=cache_summary,
        gate_features=gate_features,
    )
    assert torch.all(shifted < base)


def test_v79_output_event_carries_gate_features() -> None:
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


def test_v79_report_parser_accepts_v79_run_names() -> None:
    module = load_module("build_v79_report", "scripts/build_v79_report.py")
    parsed = module.parse_run_name(
        "20260327-000000-v79-collision-c2-ambigbase-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "ambigbase"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
