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


def test_v72_generator_emits_query_conditioned_pack():
    module = load_module("gen_v72_configs", "scripts/gen_v72_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"mean", "qcond"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v72_query_conditioned_output_readout"


def test_v72_configs_select_output_summary_source_cleanly():
    module = load_module("gen_v72_cfg", "scripts/gen_v72_configs.py")
    mean_cfg = module.build_collision_config("c2", "m", condition="mean")
    qcond_cfg = module.build_collision_config("c2", "m", condition="qcond")
    assert mean_cfg["model"]["cache_output_summary_readout"] is True
    assert qcond_cfg["model"]["cache_output_summary_readout"] is True
    assert mean_cfg["model"]["cache_output_summary_source"] == "mean"
    assert qcond_cfg["model"]["cache_output_summary_source"] == "query_conditioned"
    assert qcond_cfg["model"]["cache_output_query_condition_scale"] == 1.0


def test_v72_model_has_no_output_readout_modules_by_default():
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_condition_scale is None
    assert model.cache_output_condition_shift is None


def test_v72_query_conditioned_uses_no_extra_dual_modules():
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "query_conditioned"
    model = APSGNNModel(config)
    assert model.cache_output_condition_scale is not None
    assert model.cache_output_condition_shift is not None
    assert model.cache_output_retrieved_head is None
    assert model.cache_output_retrieved_gate is None


def test_v72_output_logits_uses_cache_summary_when_enabled():
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


def test_v72_query_conditioning_defaults_to_identity():
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "query_conditioned"
    config.model.d_model = 8
    model = APSGNNModel(config)
    residual = torch.randn(2, config.model.d_model)
    cache_summary = torch.randn(2, config.model.d_model)
    with torch.no_grad():
        conditioned = model._condition_output_cache_summary(residual, cache_summary)
    assert torch.allclose(conditioned, cache_summary)


def test_v72_query_conditioning_modulates_summary_when_enabled():
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "query_conditioned"
    config.model.cache_output_query_condition_scale = 1.0
    config.model.d_model = 4
    model = APSGNNModel(config)
    assert model.cache_output_condition_scale is not None
    assert model.cache_output_condition_shift is not None
    with torch.no_grad():
        model.cache_output_condition_scale.weight.zero_()
        model.cache_output_condition_scale.bias.fill_(0.5)
        model.cache_output_condition_shift.weight.zero_()
        model.cache_output_condition_shift.bias.fill_(0.25)
    residual = torch.randn(2, config.model.d_model)
    cache_summary = torch.ones(2, config.model.d_model)
    conditioned = model._condition_output_cache_summary(residual, cache_summary)
    expected = torch.full_like(cache_summary, 1.0 + torch.tanh(torch.tensor(0.5)).item() + 0.25)
    assert torch.allclose(conditioned, expected)


def test_v72_output_event_carries_both_summaries():
    event = OutputEvent(
        residual=torch.randn(3, 8),
        batch_index=torch.tensor([0, 1, 2]),
        role=torch.tensor([0, 1, 1]),
        target_label=torch.tensor([0, 1, 2]),
        hop_index=torch.tensor([1, 2, 3]),
        packet_id=torch.tensor([10, 11, 12]),
        cache_summary=torch.randn(3, 8),
        retrieved_summary=torch.randn(3, 8),
    )
    selected = event.select(torch.tensor([True, False, True]))
    assert selected.cache_summary is not None
    assert selected.cache_summary.shape == (2, 8)
    assert selected.retrieved_summary is not None
    assert selected.retrieved_summary.shape == (2, 8)


def test_v72_report_parser_accepts_v72_run_names():
    module = load_module("build_v72_report", "scripts/build_v72_report.py")
    parsed = module.parse_run_name(
        "20260326-230000-v72-collision-c2-qcond-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "qcond"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
