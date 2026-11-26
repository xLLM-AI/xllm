# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check Laser padding, layout, and ACLGraph replay against dense attention."""

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention
from xllm_kernel.ops.attention.laser import laser_attention


@pytest.mark.parametrize(
    ("query_rows", "kv_rows", "width", "dtype"),
    [(2048, 2048, 128, torch.bfloat16), (2051, 2051, 64, torch.float16), (31, 37, 64, torch.bfloat16)],
)
def test_laser_attention_matches_dense(query_rows: int, kv_rows: int, width: int, dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(2831)
    query, key, value = (
        torch.randn(2, rows, 2, width, generator=generator).to(dtype).transpose(1, 2)
        for rows in (query_rows, kv_rows, kv_rows)
    )
    scale = width**-0.5
    expected = scaled_dot_product_attention(query.float(), key.float(), value.float(), scale=scale)
    actual = laser_attention(query.npu(), key.npu(), value.npu(), scale).cpu()
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected, atol=0.002, rtol=0.01)


def test_laser_attention_replays_fresh_values() -> None:
    inputs = [torch.zeros((1, 2, 2051, 128), dtype=torch.bfloat16, device="npu") for _ in range(3)]
    scale = inputs[0].size(-1) ** -0.5
    laser_attention(*inputs, scale)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        result = laser_attention(*inputs, scale)
    for value in (1.0, -2.0):
        inputs[2].fill_(value)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(result.cpu(), torch.full_like(result, value, device="cpu"), atol=0.016, rtol=0)
