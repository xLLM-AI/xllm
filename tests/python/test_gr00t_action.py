# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise GR00T inference and optionally compare the NVIDIA implementation."""

from dataclasses import asdict

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.feature_extraction_utils import BatchFeature

pytest.importorskip("diffusers")

from xllm.vla.gr00t_action import Gr00tActionConfig, Gr00tActionHead


def _config(alternate: bool = True) -> Gr00tActionConfig:
    return Gr00tActionConfig.from_dict(
        {
            "backbone_embedding_dim": 16,
            "hidden_size": 12,
            "input_embedding_dim": 16,
            "max_state_dim": 3,
            "max_action_dim": 5,
            "max_num_embodiments": 3,
            "action_horizon": 4,
            "max_seq_len": 8,
            "num_inference_timesteps": 4,
            "use_alternate_vl_dit": alternate,
            "diffusion_model_cfg": {
                "num_layers": 4,
                "num_attention_heads": 2,
                "attention_head_dim": 8,
                "output_dim": 12,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
            "vl_self_attention_cfg": {
                "num_layers": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 8,
                "positional_embeddings": None,
            },
        }
    )


def _inputs() -> tuple[torch.Tensor, ...]:
    return (
        torch.randn(2, 7, 16),
        torch.randn(2, 1, 3),
        torch.tensor([0, 2]),
        torch.tensor([[False, True, True, False, False, False, False]]).expand(2, -1),
        torch.tensor([[True, True, True, True, True, False, False]]).expand(2, -1),
    )


@pytest.mark.parametrize("alternate", [True, False])
def test_seeded_actions_match_explicit_noise(alternate: bool) -> None:
    torch.manual_seed(834)
    head = Gr00tActionHead(_config(alternate)).eval()
    inputs = _inputs()
    generator = torch.Generator().manual_seed(781)
    expected = head(*inputs, generator=generator)
    generator.manual_seed(781)
    noise = torch.randn((2, 4, 5), generator=generator)
    before = noise.clone()
    actual = head(*inputs, initial_noise=noise)
    assert actual.shape == (2, 4, 5)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(noise, before, rtol=0, atol=0)


def test_rejects_incompatible_checkpoint() -> None:
    config = asdict(_config())
    config["diffusion_model_cfg"]["output_dim"] = 13
    with pytest.raises(ValueError, match="output_dim"):
        Gr00tActionConfig.from_dict(config)


def test_rejects_invalid_embodiment() -> None:
    head = Gr00tActionHead(_config()).eval()
    features, state, _, images, visible = _inputs()
    with pytest.raises(ValueError, match="category table"):
        head(features, state, torch.tensor([0, 3]), images, visible)


@pytest.mark.parametrize("alternate", [True, False])
@torch.inference_mode()
@sdpa_kernel(SDPBackend.MATH)
def test_matches_independent_nvidia_action_head(alternate: bool) -> None:
    official = pytest.importorskip("gr00t.model.gr00t_n1d7.gr00t_n1d7")
    config_module = pytest.importorskip("gr00t.configs.model.gr00t_n1d7")
    config = _config(alternate)
    reference = official.Gr00tN1d7ActionHead(config_module.Gr00tN1d7Config(**asdict(config))).eval()
    head = Gr00tActionHead(config).eval()
    head.load_state_dict(reference.state_dict(), strict=True)
    features, state, ids, images, visible = _inputs()
    torch.manual_seed(781)
    expected = reference.get_action(
        BatchFeature({"backbone_features": features, "image_mask": images, "backbone_attention_mask": visible}),
        BatchFeature({"state": state, "embodiment_id": ids}),
    )["action_pred"]
    actual = head(features, state, ids, images, visible, generator=torch.Generator().manual_seed(781))
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
