# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Verify the fused classic RoPE path and its in-place compatibility API."""

import torch
from xllm_kernel.ops.positional_encoding import apply_rotary_embedding, fused_qk_norm_rope


def _rotate_half(value: torch.Tensor) -> torch.Tensor:
    left, right = value.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


@torch.inference_mode()
def test_apply_rotary_embedding_updates_q_and_k_in_place() -> None:
    query = torch.randn((3, 2, 64), dtype=torch.bfloat16)
    key = torch.randn((3, 1, 64), dtype=torch.bfloat16)
    q = query.npu().reshape(3, -1)
    k = key.npu().reshape(3, -1)
    positions = torch.tensor([1, 0, 1], dtype=torch.int64, device="npu")
    coeffs = torch.zeros((2, 128), dtype=torch.bfloat16, device="npu")
    coeffs[0, :64].fill_(1)
    coeffs[1, 64:].fill_(1)
    pointers = (q.data_ptr(), k.data_ptr())

    assert apply_rotary_embedding(q, k, coeffs, positions) is None

    expected_q, expected_k = query.clone(), key.clone()
    expected_q[::2] = _rotate_half(query[::2])
    expected_k[::2] = _rotate_half(key[::2])
    torch.testing.assert_close(q.cpu().reshape_as(query), expected_q, rtol=0, atol=0)
    torch.testing.assert_close(k.cpu().reshape_as(key), expected_k, rtol=0, atol=0)
    assert (q.data_ptr(), k.data_ptr()) == pointers


@torch.inference_mode()
def test_fused_qk_norm_rope_consumes_live_positions_and_rows() -> None:
    dtype = torch.bfloat16
    q_heads, kv_heads, head_dim = 4, 2, 64
    widths = (q_heads * head_dim, kv_heads * head_dim, kv_heads * head_dim)
    qkv = torch.randn((4, sum(widths)), dtype=dtype, device="npu")
    q_weight = torch.rand((head_dim,), dtype=dtype, device="npu")
    k_weight = torch.rand_like(q_weight)
    positions = torch.zeros((4,), dtype=torch.int64, device="npu")
    cache = torch.zeros((2, head_dim), dtype=dtype, device="npu")
    cache[0, : head_dim // 2].fill_(1)
    cache[1, head_dim // 2 :].fill_(1)

    def forward() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return fused_qk_norm_rope(
            qkv,
            num_heads_q=q_heads,
            num_heads_k=kv_heads,
            num_heads_v=kv_heads,
            head_dim=head_dim,
            eps=1e-6,
            q_weight=q_weight,
            k_weight=k_weight,
            cos_sin_cache=cache,
            position_ids=positions,
        )

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        query, key, value = forward()

    for offset in (0, 1):
        qkv.copy_(torch.randn_like(qkv))
        positions.copy_(torch.tensor([offset, 1 - offset, offset, 1 - offset]))
        graph.replay()
        q, k, v = qkv.cpu().split(widths, dim=-1)
        rotate_mask = positions.cpu().bool().view(-1, 1, 1)
        for actual, expected, weight in ((query, q, q_weight), (key, k, k_weight)):
            expected = expected.reshape(4, -1, head_dim).float()
            expected = expected * torch.rsqrt(expected.square().mean(-1, keepdim=True) + 1e-6)
            expected = (expected * weight.cpu()).to(dtype)
            expected = torch.where(rotate_mask, _rotate_half(expected), expected)
            torch.testing.assert_close(actual.cpu(), expected.flatten(1), rtol=0.02, atol=0.01)
        torch.testing.assert_close(value.cpu(), v, rtol=0, atol=0)
