# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise mutation, layout and replay for the package-owned interleaved RoPE."""

import pytest
import torch
from xllm_kernel.ops import inplace_partial_rotary_mul


def _reference(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int) -> torch.Tensor:
    rope = value[..., start:end].float()
    pairs = rope.unflatten(-1, (-1, 2))
    rotated_pairs = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1)
    rotated = rope * cosine.float() + rotated_pairs.flatten(-2) * sine.float()
    return torch.cat((value[..., :start], rotated.to(value.dtype), value[..., end:]), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    ("shape", "start", "end", "strided"),
    [
        ((3, 1, 4, 128), 32, 96, False),
        ((7, 3, 1, 128), 64, 128, False),
        ((2, 4, 1, 96), 16, 80, True),
        ((3, 1, 2, 128), 32, 64, False),
    ],
)
def test_inplace_partial_rotary_mul(
    dtype: torch.dtype, shape: tuple[int, ...], start: int, end: int, strided: bool
) -> None:
    generator = torch.Generator().manual_seed(277)
    storage = torch.rand((*shape, 2), generator=generator).sub(0.5).to(dtype)
    value = storage[..., 0]
    angle = torch.rand((shape[0], 1, 1, end - start), generator=generator)
    cosine, sine = angle.cos().to(dtype), angle.sin().to(dtype)
    cosine_npu, sine_npu = cosine.npu(), sine.npu()
    storage_npu = storage.npu() if strided else value.contiguous().npu()
    value_npu = storage_npu[..., 0] if strided else storage_npu
    pointer, stride = value_npu.data_ptr(), value_npu.stride()

    assert inplace_partial_rotary_mul(value_npu, cosine_npu, sine_npu, start, end) is None
    assert value_npu.data_ptr() == pointer and value_npu.stride() == stride
    actual = value_npu.cpu()
    expected = _reference(value, cosine, sine, start, end)
    tol = {torch.float16: 1e-3, torch.bfloat16: 8e-3, torch.float32: 1e-6}[dtype]
    torch.testing.assert_close(actual[..., start:end], expected[..., start:end], atol=tol, rtol=tol)
    torch.testing.assert_close(actual[..., :start], value[..., :start], atol=0, rtol=0)
    torch.testing.assert_close(actual[..., end:], value[..., end:], atol=0, rtol=0)
    torch.testing.assert_close(cosine_npu.cpu(), cosine, atol=0, rtol=0)
    torch.testing.assert_close(sine_npu.cpu(), sine, atol=0, rtol=0)
    if strided:
        torch.testing.assert_close(storage_npu[..., 1].cpu(), storage[..., 1], atol=0, rtol=0)


def test_inplace_partial_rotary_mul_aclgraph_replays_fresh_inputs() -> None:
    generator = torch.Generator().manual_seed(279)
    value = torch.zeros((7, 3, 1, 128), device="npu", dtype=torch.bfloat16)
    cosine = torch.ones((7, 1, 1, 64), device="npu", dtype=value.dtype)
    sine = torch.zeros_like(cosine)
    inplace_partial_rotary_mul(value, cosine, sine, 32, 96)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        inplace_partial_rotary_mul(value, cosine, sine, 32, 96)

    for _ in range(3):
        original = torch.rand(value.shape, generator=generator).sub(0.5).to(value.dtype)
        angle = torch.rand(cosine.shape, generator=generator)
        value.copy_(original)
        cosine.copy_(angle.cos())
        sine.copy_(angle.sin())
        expected = value.clone()
        inplace_partial_rotary_mul(expected, cosine, sine, 32, 96)
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(value, expected, atol=0, rtol=0)
