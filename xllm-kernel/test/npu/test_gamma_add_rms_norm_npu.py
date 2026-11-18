# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the package-owned GammaAddRmsNorm kernel on NPU."""

import pytest
import torch
from xllm_kernel.ops import fused_add_gemma_rms_norm


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 1024), (17, 2048), (2, 3, 512)])
def test_gamma_add_rms_norm(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(581)
    value, residual = (torch.rand(shape, generator=generator).sub(0.5).to(dtype) for _ in range(2))
    weight = torch.rand(shape[-1:], generator=generator).sub(0.5).to(dtype)
    value_npu, residual_npu, weight_npu = (tensor.to("npu") for tensor in (value, residual, weight))
    normed, summed = fused_add_gemma_rms_norm(value_npu, residual_npu, weight_npu, 1e-6)

    expected_sum = value + residual
    summed_fp32 = expected_sum.float()
    normalized = summed_fp32 * torch.rsqrt(summed_fp32.square().mean(-1, keepdim=True) + 1e-6)
    expected = normalized.to(dtype) * (weight + 1)
    tolerance = {torch.float16: 1e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(normed.cpu(), expected, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(summed.cpu(), expected_sum, atol=0, rtol=0)
    torch.testing.assert_close(value_npu.cpu(), value, atol=0, rtol=0)
    torch.testing.assert_close(residual_npu.cpu(), residual, atol=0, rtol=0)
    torch.testing.assert_close(weight_npu.cpu(), weight, atol=0, rtol=0)
    assert len({t.data_ptr() for t in (value_npu, residual_npu, weight_npu, normed, summed)}) == 5


def test_gamma_add_rms_norm_aclgraph_replays_fresh_inputs() -> None:
    generator = torch.Generator().manual_seed(583)
    value = torch.zeros((7, 2048), device="npu", dtype=torch.bfloat16)
    residual = torch.zeros_like(value)
    weight = torch.zeros_like(value[0])
    fused_add_gemma_rms_norm(value, residual, weight, 1e-6)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        normed, summed = fused_add_gemma_rms_norm(value, residual, weight, 1e-6)

    for _ in range(3):
        for tensor in (value, residual, weight):
            tensor.copy_(torch.rand(tensor.shape, generator=generator) - 0.5)
        graph.replay()
        torch.npu.synchronize()
        eager_normed, eager_summed = fused_add_gemma_rms_norm(value, residual, weight, 1e-6)
        torch.testing.assert_close(normed, eager_normed, atol=0, rtol=0)
        torch.testing.assert_close(summed, eager_summed, atol=0, rtol=0)
