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


def test_v73_generator_emits_dispersion_pack() -> None:
    module = load_module("gen_v73_configs", "scripts/gen_v73_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"mean", "disp"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v73_dispersion_augmented_output_readout"


def test_v73_configs_select_dispersion_source_cleanly() -> None:
    module = load_module("gen_v73_cfg", "scripts/gen_v73_configs.py")
    mean_cfg = module.build_collision_config("c2", "m", condition="mean")
    disp_cfg = module.build_collision_config("c2", "m", condition="disp")
    assert mean_cfg["model"]["cache_output_summary_readout"] is True
    assert disp_cfg["model"]["cache_output_summary_readout"] is True
    assert mean_cfg["model"]["cache_output_summary_source"] == "mean"
    assert disp_cfg["model"]["cache_output_summary_source"] == "dispersion_augmented"
    assert disp_cfg["model"]["cache_output_dispersion_scale"] == 1.0


def test_v73_model_has_no_output_readout_modules_by_default() -> None:
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_output_summary_head is None
    assert model.cache_output_summary_gate is None
    assert model.cache_output_aux_proj is None


def test_v73_dispersion_mode_uses_aux_proj_only() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "dispersion_augmented"
    model = APSGNNModel(config)
    assert model.cache_output_aux_proj is not None
    assert model.cache_output_condition_scale is None
    assert model.cache_output_condition_shift is None
    assert model.cache_output_retrieved_head is None
    assert model.cache_output_retrieved_gate is None


def test_v73_output_logits_uses_cache_summary_when_enabled() -> None:
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


def test_v73_dispersion_augmentation_defaults_to_identity() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "dispersion_augmented"
    config.model.d_model = 8
    model = APSGNNModel(config)
    cache_summary = torch.randn(2, config.model.d_model)
    aux_summary = torch.randn(2, config.model.d_model)
    augmented = model._augment_output_cache_summary(cache_summary, aux_summary)
    assert torch.allclose(augmented, cache_summary)


def test_v73_dispersion_augmentation_adds_learned_correction() -> None:
    config = ExperimentConfig()
    config.model.cache_output_summary_readout = True
    config.model.cache_output_summary_source = "dispersion_augmented"
    config.model.cache_output_dispersion_scale = 2.0
    config.model.d_model = 4
    model = APSGNNModel(config)
    assert model.cache_output_aux_proj is not None
    with torch.no_grad():
        model.cache_output_aux_proj.weight.zero_()
        model.cache_output_aux_proj.bias.fill_(0.25)
    cache_summary = torch.ones(2, config.model.d_model)
    aux_summary = torch.randn(2, config.model.d_model)
    augmented = model._augment_output_cache_summary(cache_summary, aux_summary)
    expected = torch.full_like(cache_summary, 1.5)
    assert torch.allclose(augmented, expected)


def test_v73_output_event_carries_all_summaries() -> None:
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
    )
    selected = event.select(torch.tensor([True, False, True]))
    assert selected.cache_summary is not None
    assert selected.cache_summary.shape == (2, 8)
    assert selected.aux_summary is not None
    assert selected.aux_summary.shape == (2, 8)
    assert selected.retrieved_summary is not None
    assert selected.retrieved_summary.shape == (2, 8)


def test_v73_report_parser_accepts_v73_run_names() -> None:
    module = load_module("build_v73_report", "scripts/build_v73_report.py")
    parsed = module.parse_run_name(
        "20260326-230000-v73-collision-c2-disp-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "disp"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
