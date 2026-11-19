# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the package-owned LayerNormFwd and RmsNormDynamicQuant kernels."""

import pytest
import torch
from xllm_kernel.ops import rms_norm_dynamic_quant, rms_norm_gated

_EPS = 1e-6


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 128), (2, 7, 128), (17, 256)])
def test_gated_rms_norm(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(587)
    value = torch.randn((*shape, 2), generator=generator).to(dtype)[..., 0]
    gate = torch.randn((*shape, 2), generator=generator).to(dtype)[..., 0]
    weight = torch.rand(shape[-1], generator=generator).to(dtype)
    value_npu, gate_npu, weight_npu = (tensor.to("npu") for tensor in (value, gate, weight))

    output = rms_norm_gated(value_npu, gate_npu, weight_npu, _EPS)
    value_fp32 = value.float()
    normed = value_fp32 * torch.rsqrt(value_fp32.square().mean(-1, keepdim=True) + _EPS)
    expected = normed * weight.float() * torch.nn.functional.silu(gate.float())
    tolerance = {torch.float16: 2e-3, torch.bfloat16: 1e-2, torch.float32: 1e-5}[dtype]
    torch.testing.assert_close(output.cpu(), expected.to(dtype), atol=tolerance, rtol=tolerance)
    for actual, original in ((value_npu, value), (gate_npu, gate), (weight_npu, weight)):
        torch.testing.assert_close(actual.cpu(), original, atol=0, rtol=0)
        assert output.data_ptr() != actual.data_ptr()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(1536,), (17, 1536), (2, 7, 512), (64, 128), (1, 8192), (2, 10240), (1, 20480)])
def test_rms_norm_dynamic_quant(dtype: torch.dtype, shape: tuple[int, ...]) -> None:
    generator = torch.Generator().manual_seed(589)
    storage = torch.randn((*shape, 2), generator=generator).to(dtype)
    value = storage[..., 0]
    weight = torch.rand(shape[-1], generator=generator).to(dtype)
    value_npu = storage.to("npu")[..., 0]
    weight_npu = weight.to("npu")

    quantized, scales = rms_norm_dynamic_quant(input=value_npu, weight=weight_npu, eps=_EPS)
    value_fp32 = value.float()
    normalized = value_fp32 * torch.rsqrt(value_fp32.square().mean(-1, keepdim=True) + _EPS)
    normalized *= weight.float()
    expected_scales = normalized.abs().amax(-1) / 127
    expected_q = (normalized / expected_scales.unsqueeze(-1)).round().to(torch.int8)
    assert quantized.shape == shape and quantized.dtype == torch.int8
    assert scales.shape == shape[:-1] and scales.dtype == torch.float32
    torch.testing.assert_close(scales.cpu(), expected_scales, atol=1e-6, rtol=5e-3)
    torch.testing.assert_close(quantized.cpu(), expected_q, atol=1, rtol=0)
    torch.testing.assert_close(value_npu.cpu(), value, atol=0, rtol=0)
    torch.testing.assert_close(weight_npu.cpu(), weight, atol=0, rtol=0)
    assert len({tensor.data_ptr() for tensor in (value_npu, weight_npu, quantized, scales)}) == 4


def test_normalization_aclgraph_replays_fresh_inputs() -> None:
    generator = torch.Generator().manual_seed(593)
    value = torch.ones((7, 128), device="npu", dtype=torch.bfloat16)
    gate = torch.ones_like(value)
    weight = torch.ones_like(value[0])
    rms_norm_gated(value, gate, weight, _EPS)
    rms_norm_dynamic_quant(value, weight, _EPS)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        gated = rms_norm_gated(value, gate, weight, _EPS)
        quantized, scale = rms_norm_dynamic_quant(value, weight, _EPS)

    for _ in range(3):
        for tensor in (value, gate, weight):
            tensor.copy_(torch.rand(tensor.shape, generator=generator) + 0.25)
        graph.replay()
        torch.npu.synchronize()
        eager_gated = rms_norm_gated(value, gate, weight, _EPS)
        eager_quantized, eager_scale = rms_norm_dynamic_quant(value, weight, _EPS)
        for actual, expected in ((gated, eager_gated), (quantized, eager_quantized), (scale, eager_scale)):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
