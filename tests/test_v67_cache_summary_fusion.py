from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from apsgnn.config import ExperimentConfig
from apsgnn.model import APSGNNModel, ROLE_QUERY, ROLE_WRITER


ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_v67_generator_emits_collision_fusion_pack():
    module = load_module("gen_v67_configs", "scripts/gen_v67_configs.py")
    assert set(module.COLLISION_REGIMES) == {"c1", "c2"}
    assert set(module.CONDITIONS) == {"baseline", "fusion"}
    packs = module.pack_definitions()
    assert packs["experiment"] == "v67_cache_summary_fusion_rescue"


def test_v67_configs_toggle_home_cache_summary_fusion_cleanly():
    module = load_module("gen_v67_cfg", "scripts/gen_v67_configs.py")
    baseline = module.build_collision_config("c2", "m", condition="baseline")
    fusion = module.build_collision_config("c2", "m", condition="fusion")
    assert baseline["model"]["cache_home_summary_fusion"] is False
    assert fusion["model"]["cache_home_summary_fusion"] is True


def test_v67_home_cache_summary_mask_only_hits_query_packets_at_home_with_cache():
    config = ExperimentConfig()
    config.model.cache_home_summary_fusion = True
    model = APSGNNModel(config)
    packet_mask = torch.tensor([[True, True, True]])
    role_tensor = torch.tensor([[ROLE_QUERY, ROLE_QUERY, ROLE_WRITER]])
    target_home_tensor = torch.tensor([[5, 6, 5]])
    current_node_index = torch.tensor([5])
    cache_mask = torch.tensor([[True, False]])
    mask = model._home_cache_summary_mask(
        packet_mask=packet_mask,
        role_tensor=role_tensor,
        target_home_tensor=target_home_tensor,
        current_node_index=current_node_index,
        cache_mask=cache_mask,
    )
    assert torch.equal(mask, torch.tensor([[True, False, False]]))


def test_v67_model_has_no_fusion_modules_by_default():
    config = ExperimentConfig()
    model = APSGNNModel(config)
    assert model.cache_home_summary_proj is None
    assert model.cache_home_summary_gate is None


def test_v67_report_parser_accepts_v67_run_names():
    module = load_module("build_v67_report", "scripts/build_v67_report.py")
    parsed = module.parse_run_name(
        "20260326-220000-v67-collision-c2-fusion-visit_taskgrad_half_d-32-m-s2234"
    )
    assert parsed is not None
    assert parsed["regime"] == "c2"
    assert parsed["condition"] == "fusion"
    assert parsed["pair"] == "visit_taskgrad_half_d"
    assert parsed["schedule"] == "m"
    assert parsed["seed"] == "2234"
