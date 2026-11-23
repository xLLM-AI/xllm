# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Check static and dynamic NPU quantization storage and scale semantics."""

import torch
from xllm_kernel.ops import dynamic_quant, quantize_per_tensor


def test_static_qint8_quantization_uses_scale_and_zero_point() -> None:
    value = torch.tensor([[-4.0, -2.0, 0.0, 2.0, 4.0]], device="npu")
    scale = torch.full((1,), 2.0, device="npu")
    zero_point = torch.full((1,), 1, dtype=torch.int32, device="npu")
    actual = quantize_per_tensor(value, scale, zero_point, torch.qint8, -1)
    expected = torch.tensor([[-1, 0, 1, 2, 3]], dtype=torch.int8)
    torch.testing.assert_close(actual.cpu(), expected)


def test_static_int4_codes_are_packed_into_int32() -> None:
    value = torch.arange(-4, 4, dtype=torch.float16, device="npu").view(1, 8)
    scale = torch.ones((1,), dtype=value.dtype, device=value.device)
    zero_point = torch.zeros_like(scale)
    actual = quantize_per_tensor(value, scale, zero_point, torch.quint4x2, -1)
    expected = torch.tensor([[0x3210FEDC]], dtype=torch.int32)
    torch.testing.assert_close(actual.cpu(), expected)


def test_dynamic_quant_replays_new_scales() -> None:
    value = torch.tensor([[-8.0, 0.0, 8.0], [0.0, 16.0, -16.0]], dtype=torch.bfloat16, device="npu")
    dynamic_quant(value)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        quantized, scale = dynamic_quant(value)

    for multiplier in (2, -4):
        value.mul_(multiplier)
        graph.replay()
        source = value.cpu().float()
        expected_scale = source.abs().amax(dim=-1) / 127.0
        expected_codes = (source / expected_scale[:, None]).round().to(torch.int8)
        torch.testing.assert_close(scale.cpu(), expected_scale)
        torch.testing.assert_close(quantized.cpu(), expected_codes, rtol=0, atol=0)
