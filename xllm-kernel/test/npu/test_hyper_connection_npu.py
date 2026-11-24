# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Numerics, input preservation and ACLGraph replay for HyperConnection."""

import pytest
import torch
from xllm_kernel.ops.hyper_connection import hc_post, hc_pre

_HC_MULT = 4
_SINKHORN_ITERS = 6
_EPS = 1e-6


def _pre_reference(
    x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, base: torch.Tensor, *, fused: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    flattened = x.bfloat16().flatten(-2).float()
    projection = torch.nn.functional.linear(flattened, weight.float())
    if not fused:
        projection = projection.bfloat16().float()
    logits = projection * (flattened.square().mean(-1, keepdim=True) + _EPS).rsqrt()
    pre_logits, post_logits, comb_logits = logits.split((_HC_MULT, _HC_MULT, _HC_MULT**2), dim=-1)
    pre_base, post_base, comb_base = base.split((_HC_MULT, _HC_MULT, _HC_MULT**2))
    pre = torch.sigmoid(pre_logits * scale[0] + pre_base) + _EPS
    post = 2 * torch.sigmoid(post_logits * scale[1] + post_base)
    comb = (comb_logits * scale[2] + comb_base).unflatten(-1, (_HC_MULT, _HC_MULT))
    comb = torch.softmax(comb, dim=-1) + _EPS
    comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    for _ in range(_SINKHORN_ITERS - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + _EPS)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + _EPS)
    collapsed = (x.bfloat16().float() * pre.unsqueeze(-1)).sum(dim=-2).bfloat16().to(x.dtype)
    return collapsed, post, comb


def _post_reference(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    mixed = torch.einsum("bskd,bskj->bsjd", residual.float(), comb.float())
    return (x.float().unsqueeze(-2) * post.float().unsqueeze(-1) + mixed).to(residual.dtype)


def _parameters(width: int, dtype: torch.dtype, generator: torch.Generator) -> tuple[torch.Tensor, ...]:
    mix_count = _HC_MULT * (_HC_MULT + 2)
    weight = (torch.randn((mix_count, _HC_MULT * width), generator=generator) / width**0.5).bfloat16().to(dtype)
    scale = torch.tensor([0.2, 0.3, 0.4])
    base = torch.randn((mix_count,), generator=generator) * 0.7
    return weight, scale, base


@pytest.mark.parametrize(
    ("shape", "dtype", "weight_dtype", "strided"),
    [
        ((5, 4, 4096), torch.bfloat16, torch.float32, False),
        ((2, 3, 4, 4096), torch.float16, torch.bfloat16, True),
    ],
)
def test_hc_pre_matches_reference(
    shape: tuple[int, ...], dtype: torch.dtype, weight_dtype: torch.dtype, strided: bool
) -> None:
    generator = torch.Generator().manual_seed(2786)
    x = torch.randn(shape, generator=generator).to(dtype)
    params = _parameters(shape[-1], weight_dtype, generator)
    x_device = x.npu()
    if strided:
        x_device = torch.stack((x, torch.zeros_like(x)), dim=-1).npu()[..., 0]
    device_params = tuple(param.npu() for param in params)
    expected = _pre_reference(
        x,
        *params,
        fused=torch.npu.get_device_name().startswith("Ascend910_93") and weight_dtype == torch.float32 and not strided,
    )

    actual = hc_pre(x_device, *device_params, _HC_MULT, _SINKHORN_ITERS, _EPS, _EPS)
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0.02, rtol=0.012)
    torch.testing.assert_close(actual[1].cpu(), expected[1], atol=0.003, rtol=0.003)
    torch.testing.assert_close(actual[2].cpu(), expected[2], atol=0.003, rtol=0.003)
    if strided:
        contiguous = hc_pre(x_device.contiguous(), *device_params, _HC_MULT, _SINKHORN_ITERS, _EPS, _EPS)
        for result, contiguous_result in zip(actual, contiguous, strict=True):
            torch.testing.assert_close(result, contiguous_result, atol=0, rtol=0)
    torch.testing.assert_close(x_device.cpu(), x, atol=0, rtol=0)
    for actual_param, expected_param in zip(device_params, params, strict=True):
        torch.testing.assert_close(actual_param.cpu(), expected_param, atol=0, rtol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_hc_post_mixes_residual_with_input_major_comb(dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(2787)
    x = torch.randn((2, 3, 256), generator=generator).to(dtype)
    residual = torch.randn((2, 3, _HC_MULT, 256), generator=generator).to(dtype)
    post = torch.rand((2, 3, _HC_MULT), generator=generator)
    comb = torch.rand((2, 3, _HC_MULT, _HC_MULT), generator=generator)
    inputs = (x, residual, post, comb)
    device_inputs = (x.npu(), residual.npu(), post.npu(), comb.transpose(-2, -1).contiguous().npu().transpose(-2, -1))
    output = hc_post(*device_inputs)

    tolerance = {torch.float16: 0.004, torch.bfloat16: 0.025, torch.float32: 2e-6}[dtype]
    torch.testing.assert_close(output.cpu(), _post_reference(*inputs), atol=tolerance, rtol=0)
    assert output.data_ptr() != device_inputs[1].data_ptr()
    for actual_input, expected_input in zip(device_inputs, inputs, strict=True):
        torch.testing.assert_close(actual_input.cpu(), expected_input, atol=0, rtol=0)


@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_hyper_connection_aclgraph_replays_fresh_inputs(weight_dtype: torch.dtype) -> None:
    generator = torch.Generator().manual_seed(2788)
    residual = torch.zeros((1, 4, _HC_MULT, 4096), device="npu", dtype=torch.bfloat16)
    params = tuple(param.npu() for param in _parameters(residual.shape[-1], weight_dtype, generator))
    linear_output = torch.zeros((1, 4, 4096), device="npu", dtype=residual.dtype)

    def forward() -> tuple[torch.Tensor, ...]:
        collapsed, post, comb = hc_pre(residual, *params, _HC_MULT, _SINKHORN_ITERS, _EPS, _EPS)
        return collapsed, post, comb, hc_post(linear_output, residual, post, comb)

    forward()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = forward()

    for _ in range(3):
        residual.copy_(torch.randn(residual.shape, generator=generator).bfloat16())
        linear_output.copy_(torch.randn(linear_output.shape, generator=generator).bfloat16())
        eager = forward()
        graph.replay()
        torch.npu.synchronize()
        for actual, expected in zip(captured, eager, strict=True):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
