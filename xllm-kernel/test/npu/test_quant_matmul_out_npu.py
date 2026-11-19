# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Exercise the package matmul with NZ decode and general CANN layouts."""

from collections.abc import Iterator

import pytest
import torch
import torch_npu
from xllm_kernel.ops import quant_matmul_out


@pytest.fixture
def _allow_nz() -> Iterator[None]:
    previous = torch_npu._C._npu_getOption("ALLOW_INTERNAL_FORMAT")
    torch.npu.config.allow_internal_format = True
    yield
    torch.npu.config.allow_internal_format = previous in ("enable", b"enable")


@pytest.mark.parametrize("m", [1, 16, 17])
def test_quant_matmul_out_nz(m: int, _allow_nz: None) -> None:
    generator = torch.Generator().manual_seed(617)
    k, n = 5120, 1280
    value = torch.randint(-4, 5, (m, k), generator=generator, dtype=torch.int8).npu()
    weight = torch.randint(-4, 5, (k, n), generator=generator, dtype=torch.int8).npu()
    weight_nz = torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ)
    assert torch_npu.get_npu_format(weight_nz) == torch_npu.Format.FRACTAL_NZ
    scale = (torch.rand(n, generator=generator) / 8).npu()
    bias = torch.randint(-32, 33, (n,), generator=generator, dtype=torch.int32).npu()
    out = torch.empty((m, n), device="npu", dtype=torch.bfloat16)

    assert quant_matmul_out(value, weight_nz, False, scale, None, None, bias, torch.bfloat16, out) is out
    reference = torch_npu.npu_quant_matmul(value, weight, scale, bias=bias, output_dtype=torch.bfloat16)
    torch.testing.assert_close(out, reference, rtol=0, atol=0)


def test_quant_matmul_out_transposed_dynamic_scale() -> None:
    generator = torch.Generator().manual_seed(619)
    value = torch.randint(-4, 5, (3, 64), generator=generator, dtype=torch.int8)
    weight = torch.randint(-4, 5, (96, 64), generator=generator, dtype=torch.int8)
    scale = torch.rand(96, generator=generator)
    pertoken_scale = torch.rand(3, generator=generator)
    out = torch.empty((3, 96), device="npu", dtype=torch.bfloat16)
    assert (
        quant_matmul_out(
            value.npu(), weight.npu(), True, scale.npu(), None, pertoken_scale.npu(), None, torch.bfloat16, out
        )
        is out
    )
    expected = (value.float() @ weight.float().T) * scale * pertoken_scale[:, None]
    torch.testing.assert_close(out.cpu(), expected.to(out.dtype), rtol=0, atol=0)


def test_quant_matmul_out_nz_aclgraph_replay(_allow_nz: None) -> None:
    generator = torch.Generator().manual_seed(621)
    k, n = 5120, 1280
    weight = torch.randint(-4, 5, (k, n), generator=generator, dtype=torch.int8).npu()
    weight_nz = torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ)
    value = torch.zeros((3, k), dtype=torch.int8).npu()
    scale = (torch.rand(n, generator=generator) / 8).npu()
    bias = torch.randint(-32, 33, (n,), generator=generator, dtype=torch.int32).npu()
    out = torch.empty((3, n), device="npu", dtype=torch.bfloat16)
    quant_matmul_out(value, weight_nz, False, scale, None, None, bias, out.dtype, out)
    torch.npu.synchronize()

    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        quant_matmul_out(value, weight_nz, False, scale, None, None, bias, out.dtype, out)
    for _ in range(3):
        value.copy_(torch.randint(-4, 5, value.shape, generator=generator, dtype=torch.int8))
        graph.replay()
        torch.npu.synchronize()
        reference = torch_npu.npu_quant_matmul(value, weight, scale, bias=bias, output_dtype=out.dtype)
        torch.testing.assert_close(out, reference, rtol=0, atol=0)
