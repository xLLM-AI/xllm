# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu", reason="requires the real NPU backend")
pytest.importorskip("diffusers", reason="requires the MiniMax-H3 model components")

from diffusers import MiniMaxH3Transformer3DModel

from xllm.python.dit.minimax_h3 import _build_layout, _VideoGeometry
from xllm.python.dit.minimax_h3_acl_graph import MiniMaxH3AclGraphRunner
from xllm.python.dit.minimax_h3_model import MiniMaxH3Denoiser, SequenceParallel, TensorParallel

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)


class _TensorShard:
    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        self._tensors = tensors

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._tensors[name]

    def keys(self) -> list[str]:
        return list(self._tensors)


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _exchange(tensor: torch.Tensor, scatter_heads: bool) -> torch.Tensor:
    return tensor


@pytest.mark.parametrize("laser_attention_enabled", (False, True))
@torch.no_grad()
def test_graph_matches_eager_across_requests_and_timesteps(tmp_path: Path, laser_attention_enabled: bool) -> None:
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    torch.manual_seed(42)
    model = MiniMaxH3Transformer3DModel(
        num_attention_heads=2,
        attention_head_dim=64,
        hidden_size=128,
        num_layers=1,
        num_refiner_layers=1,
        ffn_dim=256,
        in_channels=4,
        audio_in_channels=4,
        freq_dim=64,
        time_embed_hidden_dim=128,
        time_embed_dim=64,
        rope_freq_dim=8,
    )
    model.save_config(tmp_path)
    denoiser = MiniMaxH3Denoiser(
        str(tmp_path),
        [_TensorShard(model.state_dict())],
        device,
        TensorParallel(0, 1, _identity, _identity),
        SequenceParallel(0, 1, _identity, _exchange),
    )
    if laser_attention_enabled:
        # conftest selects the LLM operation set, which excludes Laser.
        # Use the kernel adapter's native binding without changing that policy.
        denoiser._laser = torch.ops.xllm_ops.laser_attention.default
    runner = MiniMaxH3AclGraphRunner(denoiser, device)
    # The second request has new text/tags/rotary values at the same shape.
    # Later requests change the geometry, then revisit the original shape.
    side = 64 if laser_attention_enabled else 4
    for request, extra_width in enumerate((0, 0, 2, 0)):
        geometry = _VideoGeometry(2, side, side + extra_width, 8, 16, 4, 2, 2, 2)
        tags = tuple((row + request) % 2 for row in range(5))
        layout = _build_layout(tags, geometry, audio_latents=3, device=device)
        layout.positions.add_(request)
        prompt = torch.randn(1, len(tags), 5120, device=device, dtype=torch.bfloat16)
        conditioning = denoiser.prepare_conditioning(prompt, layout)
        video = torch.randn(1, layout.video_indices.numel(), 16, device=device)
        audio = torch.randn(1, layout.audio_indices.numel(), 4, device=device)
        for times, mode in (((1.0,), 0), ((0.2, 0.8), 1), ((0.1, 0.7), 2), ((0.4,), 0)):
            timestep = torch.tensor(times, device=device)
            indices = layout.timestep_indices[mode]
            expected = denoiser(video, audio, timestep, indices, conditioning)
            actual = runner(video, audio, timestep, indices, conditioning)
            for result, reference in zip(actual, expected, strict=True):
                torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)
            video = actual[0] * 0.1 + video
            audio = actual[1] * 0.1 + audio
        assert runner.captured_steps == (2 if request < 2 else 2 * request)
    assert runner.replayed_steps == 16
