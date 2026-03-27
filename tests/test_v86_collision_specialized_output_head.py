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


def test_v86_generator_emits_ambig_and_splithead_pack() -> None:
    module = load_module("gen_v86_configs", "scripts/gen_v86_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"ambig", "splithead"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v86_collision_specialized_output_head"


def test_v86_configs_select_sources_cleanly() -> None:
    module = load_module("gen_v86_cfg", "scripts/gen_v86_configs.py")
    ambig_cfg = module.build_collision_config("c2", "m", condition="ambig")
    split_cfg = module.build_collision_config("c2", "m", condition="splithead")
    assert ambig_cfg["model"]["cache_output_summary_source"] == "ambiguity_gate"
    assert split_cfg["model"]["cache_output_summary_source"] == "collision_specialized_head"
    assert split_cfg["model"]["cache_output_gate_feature_scale"] == 1.0
    assert split_cfg["model"]["cache_output_collision_switch_width"] == 2.0
    assert split_cfg["model"]["cache_output_collision_head_scale"] == 1.0


def test_v86_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_gate_feature_proj is None
    assert model.cache_output_collision_head is None


def test_v86_splithead_mode_uses_gate_and_collision_head() -> None:
    model = APSGNNModel(small_config("collision_specialized_head"))
    assert model.cache_output_gate_feature_proj is not None
    assert model.cache_output_gate_feature_proj.in_features == 2
    assert model.cache_output_collision_head is not None
    assert model.cache_output_aux_proj is None
    assert model.cache_output_base_feature_proj is None
    assert model.cache_output_classslice_feature_proj is None


def test_v86_output_logits_uses_cache_summary_when_enabled() -> None:
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


def test_v86_splithead_matches_v75_when_collision_head_is_zero() -> None:
    ambig_cfg = small_config("ambiguity_gate")
    ambig_cfg.model.use_reserved_class_slice = False
    split_cfg = small_config("collision_specialized_head")
    split_cfg.model.use_reserved_class_slice = False
    ambig = APSGNNModel(ambig_cfg)
    splithead = APSGNNModel(split_cfg)
    with torch.no_grad():
        ambig.output_head.weight.zero_()
        ambig.output_head.bias.zero_()
        splithead.output_head.weight.zero_()
        splithead.output_head.bias.zero_()
        assert ambig.cache_output_summary_head is not None
        assert splithead.cache_output_summary_head is not None
        ambig_final = ambig.cache_output_summary_head[-1]
        split_final = splithead.cache_output_summary_head[-1]
        assert isinstance(ambig_final, torch.nn.Linear)
        assert isinstance(split_final, torch.nn.Linear)
        ambig_final.weight.zero_()
        ambig_final.bias.fill_(1.0)
        split_final.weight.copy_(ambig_final.weight)
        split_final.bias.copy_(ambig_final.bias)
        assert ambig.cache_output_summary_gate is not None
        assert splithead.cache_output_summary_gate is not None
        ambig.cache_output_summary_gate.weight.zero_()
        ambig.cache_output_summary_gate.bias.zero_()
        splithead.cache_output_summary_gate.weight.zero_()
        splithead.cache_output_summary_gate.bias.zero_()
        assert ambig.cache_output_gate_feature_proj is not None
        assert splithead.cache_output_gate_feature_proj is not None
        ambig.cache_output_gate_feature_proj.weight.zero_()
        ambig.cache_output_gate_feature_proj.bias.zero_()
        ambig.cache_output_gate_feature_proj.weight[0, 0] = 1.0
        splithead.cache_output_gate_feature_proj.weight.zero_()
        splithead.cache_output_gate_feature_proj.bias.zero_()
        splithead.cache_output_gate_feature_proj.weight[0, 0] = 1.0
    residual = torch.zeros(2, ambig_cfg.model.d_model)
    cache_summary = torch.ones(2, ambig_cfg.model.d_model)
    gate_features = torch.tensor([[0.8, 3.0 / 48.0], [0.8, 4.0 / 48.0]], dtype=residual.dtype)
    ambig_logits = ambig._output_logits(residual, cache_summary=cache_summary, gate_features=gate_features)
    split_logits = splithead._output_logits(residual, cache_summary=cache_summary, gate_features=gate_features)
    assert torch.allclose(ambig_logits, split_logits)


def test_v86_splithead_only_changes_high_collision_outputs() -> None:
    cfg = small_config("collision_specialized_head")
    cfg.model.use_reserved_class_slice = False
    model = APSGNNModel(cfg)
    with torch.no_grad():
        model.output_head.weight.zero_()
        model.output_head.bias.zero_()
        assert model.cache_output_summary_head is not None
        assert model.cache_output_summary_gate is not None
        assert model.cache_output_collision_head is not None
        base_final = model.cache_output_summary_head[-1]
        collision_final = model.cache_output_collision_head[-1]
        assert isinstance(base_final, torch.nn.Linear)
        assert isinstance(collision_final, torch.nn.Linear)
        base_final.weight.zero_()
        base_final.bias.fill_(1.0)
        collision_final.weight.zero_()
        collision_final.bias.fill_(2.0)
        model.cache_output_summary_gate.weight.zero_()
        model.cache_output_summary_gate.bias.fill_(10.0)
        assert model.cache_output_gate_feature_proj is not None
        model.cache_output_gate_feature_proj.weight.zero_()
        model.cache_output_gate_feature_proj.bias.zero_()
    residual = torch.zeros(2, cfg.model.d_model)
    cache_summary = torch.ones(2, cfg.model.d_model)
    low_collision = torch.tensor([[0.8, 1.0 / 48.0], [0.8, 1.25 / 48.0]], dtype=residual.dtype)
    high_collision = torch.tensor([[0.8, 4.0 / 48.0], [0.8, 5.0 / 48.0]], dtype=residual.dtype)
    low_logits = model._output_logits(residual, cache_summary=cache_summary, gate_features=low_collision)
    high_logits = model._output_logits(residual, cache_summary=cache_summary, gate_features=high_collision)
    assert torch.all(high_logits > low_logits)


def test_v86_output_event_carries_two_feature_gate_payload() -> None:
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


def test_v86_report_parser_accepts_v86_run_names() -> None:
    module = load_module("build_v86_report", "scripts/build_v86_report.py")
    parsed = module.parse_run_name(
        "20260327-040000-v86-collision-c2-splithead-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "splithead"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
