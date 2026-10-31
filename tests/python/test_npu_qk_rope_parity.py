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

from __future__ import annotations

import os

import pytest
import torch

pytest.importorskip("torch_npu", reason="requires the real NPU backend")

pytestmark = pytest.mark.skipif(
    os.environ.get("XLLM_TEST_NPU") != "1",
    reason="requires an explicitly selected idle NPU and XLLM_TEST_NPU=1",
)


@pytest.mark.parametrize("head_dim", [64, 128], ids=["head64", "head128"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32], ids=["fp16", "bf16", "fp32"])
@pytest.mark.parametrize("num_tokens", [1, 7], ids=["single-token", "strided-multi-token"])
@torch.inference_mode()
def test_apply_qk_rotary_graph_replay_matches_half_rope(head_dim: int, dtype: torch.dtype, num_tokens: int) -> None:
    from xllm.python import kernels
    from xllm.python.layers import apply_rotary_half

    torch.npu.set_device(0)
    device = torch.device("npu:0")
    q_size, kv_size = 6 * head_dim, 2 * head_dim
    generator = torch.Generator(device="cpu").manual_seed(61031 + head_dim)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    positions = torch.randperm(4096, generator=generator)[:num_tokens].to(torch.int64)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)
    half_cos = freqs.cos().to(dtype).to(device)
    half_sin = freqs.sin().to(dtype).to(device)
    qkv = torch.randn(num_tokens, q_size + 2 * kv_size, device=device, dtype=dtype)

    # Strided Q/K views into one packed QKV buffer, as in Qwen2Attention.
    q = qkv[:, :q_size].view(num_tokens, 6, head_dim)
    k = qkv[:, q_size : q_size + kv_size].view(num_tokens, 2, head_dim)
    full_cos = torch.cat((half_cos, half_cos), dim=-1).view(1, num_tokens, 1, head_dim)
    full_sin = torch.cat((half_sin, half_sin), dim=-1).view(1, num_tokens, 1, head_dim)

    stream = torch.npu.Stream(device=device)
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            kernels.apply_qk_rotary(q, k, full_cos, full_sin)
        stream.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            out_q, out_k = kernels.apply_qk_rotary(q, k, full_cos, full_sin)
    torch.npu.current_stream().wait_stream(stream)

    bound = 2 * torch.finfo(dtype).eps

    for phase in (0.0, 0.5):
        qkv_input = torch.randn(num_tokens, q_size + 2 * kv_size, device=device, dtype=dtype)
        cos = torch.cos(freqs + phase).to(dtype).to(device)
        sin = torch.sin(freqs + phase).to(dtype).to(device)
        qkv.copy_(qkv_input)
        full_cos.copy_(torch.cat((cos, cos), dim=-1).view(1, num_tokens, 1, head_dim))
        full_sin.copy_(torch.cat((sin, sin), dim=-1).view(1, num_tokens, 1, head_dim))
        graph.replay()
        torch.npu.synchronize()
        # V is read after RoPE in production; the op's in-place mutation is
        # confined to its internal clones, so the packed storage must be intact.
        torch.testing.assert_close(qkv.cpu(), qkv_input.cpu(), rtol=0, atol=0)
        expected_q = apply_rotary_half(qkv_input[:, :q_size].view(num_tokens, 6, head_dim), cos, sin)
        expected_k = apply_rotary_half(qkv_input[:, q_size : q_size + kv_size].view(num_tokens, 2, head_dim), cos, sin)
        torch.testing.assert_close(out_q.cpu(), expected_q.cpu(), rtol=bound, atol=bound)
        torch.testing.assert_close(out_k.cpu(), expected_k.cpu(), rtol=bound, atol=bound)
