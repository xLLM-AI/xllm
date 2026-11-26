# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check fused mRoPE head dispatch, layout and live graph inputs."""

from importlib.util import find_spec

import pytest
import torch
from xllm_kernel.ops.positional_encoding import (
    build_split_qkv_rmsnorm_mrope_gather_pattern,
    has_split_qkv_rmsnorm_mrope_specialization,
    prepare_split_qkv_rmsnorm_mrope,
    split_qkv_rmsnorm_mrope,
)


@pytest.mark.skipif(find_spec("tilelang") is None, reason="requires TileLang")
@pytest.mark.parametrize("tokens", [1, 3, 53])
@torch.inference_mode()
def test_fused_mrope_replays_strided_inputs(tokens: int) -> None:
    device = torch.device("npu:0")
    q_heads, kv_heads, head_size = 4, 1, 256
    assert has_split_qkv_rmsnorm_mrope_specialization(q_heads, kv_heads, head_size)
    assert not has_split_qkv_rmsnorm_mrope_specialization(q_heads, kv_heads, 128)
    assert not has_split_qkv_rmsnorm_mrope_specialization(5, kv_heads, head_size)
    prepare_split_qkv_rmsnorm_mrope(q_heads, kv_heads, head_size)
    dtype = torch.bfloat16
    q_width = q_heads * head_size
    kv_width = kv_heads * head_size
    width = 2 * (q_width + kv_width)
    qkvg = torch.randn((tokens, width + 16), dtype=dtype, device=device)[:, :width]
    q_weight = torch.randn((head_size,), dtype=dtype, device=device)
    k_weight = torch.randn_like(q_weight)
    cos_sin = torch.zeros((tokens, 192), dtype=dtype, device=device)
    for axis in range(3):
        cos_sin[:, axis * 64 : axis * 64 + 32].fill_(axis + 1)
    pattern = build_split_qkv_rmsnorm_mrope_gather_pattern(64, [11, 11, 10], False, device)
    rotary_scale = torch.tensor([1] * 11 + [2] * 11 + [3] * 10, dtype=torch.float32).repeat(2)

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return split_qkv_rmsnorm_mrope(
            qkvg,
            q_weight,
            k_weight,
            cos_sin,
            pattern,
            1e-6,
            q_heads,
            kv_heads,
            head_size,
        )

    def reference() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        q, gate, k, v = qkvg.cpu().split((q_width, q_width, kv_width, kv_width), dim=-1)
        normed = []
        for value, weight in ((q, q_weight), (k, k_weight)):
            rows = value.reshape(tokens, -1, head_size).float()
            rows = rows * torch.rsqrt(rows.square().mean(dim=-1, keepdim=True) + 1e-6)
            rows = rows * weight.cpu().float()
            rows[..., :64] *= rotary_scale
            normed.append(rows.to(dtype))
        return normed[0], normed[1], v.view(tokens, kv_heads, head_size), gate.view(tokens, q_heads, head_size)

    eager = forward()
    torch.testing.assert_close(tuple(value.cpu() for value in eager), reference(), atol=0.04, rtol=0.02)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = forward()

    for _ in range(2):
        qkvg.copy_(torch.randn_like(qkvg))
        graph.replay()
        torch.testing.assert_close(tuple(value.cpu() for value in outputs), reference(), atol=0.04, rtol=0.02)
