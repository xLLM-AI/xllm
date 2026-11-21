# Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0
"""Exercise real Flash computation and fresh-input HIP graph replay."""

import pytest
import torch

pytest.importorskip("diffusers")

from xllm.vla.gr00t_action import Gr00tActionConfig, Gr00tActionHead
from xllm.vla.gr00t_execution import _Gr00tActionRunner


@pytest.mark.parametrize("execution", ["eager", "graph"])
@torch.inference_mode()
def test_flash_actions_and_fresh_inputs(execution: str) -> None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("Flash/graph execution requires DCU")
    torch.manual_seed(731)
    config = Gr00tActionConfig.from_dict(
        {
            "backbone_embedding_dim": 64,
            "hidden_size": 32,
            "input_embedding_dim": 64,
            "max_state_dim": 3,
            "max_action_dim": 5,
            "max_num_embodiments": 3,
            "action_horizon": 4,
            "max_seq_len": 8,
            "num_inference_timesteps": 4,
            "diffusion_model_cfg": {
                "num_layers": 4,
                "num_attention_heads": 2,
                "attention_head_dim": 32,
                "output_dim": 32,
                "interleave_self_attention": True,
                "norm_type": "ada_norm",
                "positional_embeddings": None,
            },
            "vl_self_attention_cfg": {
                "num_layers": 2,
                "num_attention_heads": 2,
                "attention_head_dim": 32,
                "positional_embeddings": None,
            },
        }
    )
    flash = Gr00tActionHead(config, attention_backend="flash").to(device="cuda", dtype=torch.bfloat16).eval()
    reference = Gr00tActionHead(config).to(device="cuda", dtype=torch.bfloat16).eval()
    reference.load_state_dict(flash.state_dict())
    runner = _Gr00tActionRunner(flash, execution)
    outputs = []
    for step in range(3):
        # Same specialization, different masks, noise, state and embodiment:
        # replay must update values and indices, not just reuse the first call.
        features = torch.randn(1, 12, 64, device="cuda", dtype=torch.bfloat16)
        state = torch.randn(1, 1, 3, device="cuda", dtype=torch.bfloat16)
        ids = torch.tensor([step], device="cuda")
        images = torch.zeros(1, 12, device="cuda", dtype=torch.bool)
        images[:, step : step + 4] = True
        visible = torch.ones_like(images)
        visible[:, -1] = False
        noise = torch.randn(1, 4, 5, device="cuda", dtype=torch.bfloat16)
        before = noise.clone()
        inputs = (features, state, ids, images, visible)
        expected = reference(*inputs, initial_noise=noise)
        actual = runner.run(*inputs, initial_noise=noise)
        torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        torch.testing.assert_close(noise, before, atol=0, rtol=0)
        outputs.append(actual.clone())
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[1], outputs[2])
    assert runner.requests == 3
    assert runner.captures == (1 if execution == "graph" else 0)
    assert runner.replays == (3 if execution == "graph" else 0)
