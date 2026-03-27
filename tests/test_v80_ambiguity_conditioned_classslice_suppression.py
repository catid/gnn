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


def test_v80_generator_emits_ambig_and_ambigclass_pack() -> None:
    module = load_module("gen_v80_configs", "scripts/gen_v80_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"ambig", "ambigclass"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v80_ambiguity_conditioned_classslice_suppression"


def test_v80_configs_select_sources_cleanly() -> None:
    module = load_module("gen_v80_cfg", "scripts/gen_v80_configs.py")
    ambig_cfg = module.build_collision_config("c2", "m", condition="ambig")
    ambigclass_cfg = module.build_collision_config("c2", "m", condition="ambigclass")
    assert ambig_cfg["model"]["cache_output_summary_source"] == "ambiguity_gate"
    assert ambigclass_cfg["model"]["cache_output_summary_source"] == "ambiguity_classslice_suppression"
    assert ambigclass_cfg["model"]["cache_output_gate_feature_scale"] == 1.0
    assert ambigclass_cfg["model"]["cache_output_classslice_suppression_scale"] == 1.0


def test_v80_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_gate_feature_proj is None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_classslice_feature_proj is None


def test_v80_classslice_mode_uses_gate_and_classslice_projection_only() -> None:
    model = APSGNNModel(small_config("ambiguity_classslice_suppression"))
    assert model.cache_output_gate_feature_proj is not None
    assert model.cache_output_classslice_feature_proj is not None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_aux_proj is None
    assert model.cache_output_feature_scale_proj is None
    assert model.cache_output_feature_shift_proj is None


def test_v80_output_logits_uses_cache_summary_when_enabled() -> None:
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


def test_v80_classslice_suppression_defaults_to_v75_behavior() -> None:
    ambig = APSGNNModel(small_config("ambiguity_gate"))
    ambigclass = APSGNNModel(small_config("ambiguity_classslice_suppression"))
    ambigclass.load_state_dict(ambig.state_dict(), strict=False)
    with torch.no_grad():
        assert ambig.cache_output_summary_gate is not None
        assert ambigclass.cache_output_summary_gate is not None
        ambig.cache_output_summary_gate.weight.zero_()
        ambig.cache_output_summary_gate.bias.zero_()
        ambigclass.cache_output_summary_gate.weight.zero_()
        ambigclass.cache_output_summary_gate.bias.zero_()
    residual = torch.randn(2, ambig.config.model.d_model)
    cache_summary = torch.randn(2, ambig.config.model.d_model)
    gate_features = torch.randn(2, 2)
    ambig_logits = ambig._output_logits(residual, cache_summary=cache_summary, gate_features=gate_features)
    ambigclass_logits = ambigclass._output_logits(
        residual,
        cache_summary=cache_summary,
        gate_features=gate_features,
    )
    assert torch.allclose(ambig_logits, ambigclass_logits)


def test_v80_classslice_suppression_changes_logits_when_trained() -> None:
    config = small_config("ambiguity_classslice_suppression")
    config.model.cache_output_classslice_suppression_scale = 1.5
    model = APSGNNModel(config)
    assert model.cache_output_classslice_feature_proj is not None
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()
        assert model.cache_output_summary_head is not None
        assert model.cache_output_summary_gate is not None
        final_linear = model.cache_output_summary_head[-1]
        assert isinstance(final_linear, torch.nn.Linear)
        final_linear.weight.zero_()
        final_linear.bias.zero_()
        model.cache_output_summary_gate.weight.zero_()
        model.cache_output_summary_gate.bias.zero_()
        model.cache_output_classslice_feature_proj.weight.zero_()
        model.cache_output_classslice_feature_proj.bias.fill_(-0.4)
    residual = torch.zeros(2, config.model.d_model)
    residual[:, config.model.key_dim : config.model.key_dim + config.model.num_classes] = 1.0
    cache_summary = torch.zeros(2, config.model.d_model)
    gate_features = torch.ones(2, 2)
    base = model._output_logits(residual, cache_summary=cache_summary)
    shifted = model._output_logits(
        residual,
        cache_summary=cache_summary,
        gate_features=gate_features,
    )
    assert torch.all(shifted < base)


def test_v80_output_event_carries_gate_features() -> None:
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


def test_v80_report_parser_accepts_v80_run_names() -> None:
    module = load_module("build_v80_report", "scripts/build_v80_report.py")
    parsed = module.parse_run_name(
        "20260327-000000-v80-collision-c2-ambigclass-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "ambigclass"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
