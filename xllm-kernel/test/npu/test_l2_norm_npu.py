# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Real NPU L2Norm bindings, streams, mutation and graph replay."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_nondefault_stream_and_changed_input_graph_replays(strided: bool, offset: int) -> None:
    import torch
    from xllm_kernel.ops import l2_norm

    stream = torch.npu.Stream()
    storage = torch.randn(17 * 192 * (2 if strided else 1) + offset, dtype=torch.bfloat16, device="npu")
    value = storage[offset :: 2 if strided else 1].reshape(17, 192)
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            output = l2_norm(value)
        for index in range(4):
            value.copy_(torch.randn_like(value) if index < 3 else torch.zeros_like(value))
            graph.replay()
            expected = l2_norm(value)
            stream.synchronize()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
    stream.synchronize()


@pytest.mark.parametrize(
    "shape",
    [(64,), (1, 31, 4, 64), (1, 31, 4, 128), (2, 17, 3, 192), (2, 64, 8, 256), (3, 127), (3, 4096), (4097, 128)],
)
@pytest.mark.parametrize("eps", [1e-6, 0.125])
@pytest.mark.parametrize("strided", [False, True])
def test_numeric_shape_alias_and_mutation(shape: tuple[int, ...], eps: float, strided: bool) -> None:
    import torch
    from xllm_kernel.ops import l2_norm

    torch.manual_seed(42)
    storage = torch.randn((*shape[:-1], shape[-1] * 2 if strided else shape[-1]), dtype=torch.bfloat16, device="npu")
    value = storage[..., ::2] if strided else storage
    before = value.cpu()
    fp32 = before.float()
    expected = (fp32 * torch.rsqrt(fp32.square().sum(-1, keepdim=True) + eps)).bfloat16()
    output = l2_norm(value, eps)
    assert output.shape == value.shape and output.dtype == value.dtype
    assert output.is_contiguous() and output.data_ptr() != value.data_ptr()
    torch.testing.assert_close(output.cpu(), expected, rtol=0.01, atol=0.002)
    torch.testing.assert_close(value.cpu(), before, rtol=0, atol=0)


def test_standalone_package_runs_without_xllm_host(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
from typing import Any

class BlockHostImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split('.')[0] == 'xllm':
            raise AssertionError(f'unexpected host import: {fullname}')

sys.meta_path.insert(0, BlockHostImports())
import torch
import torch_npu
from xllm_kernel import initialize
from xllm_kernel.ops import l2_norm

plan = initialize(device='npu', operations=('l2_norm',))['l2_norm']
assert plan.spec.binding == 'python'
value = torch.randn(17, 192, dtype=torch.bfloat16, device='npu')
value_fp32 = value.cpu().float()
expected = (value_fp32 * torch.rsqrt(value_fp32.square().sum(-1, keepdim=True) + 1e-6)).bfloat16()
torch.testing.assert_close(l2_norm(value).cpu(), expected, rtol=0.01, atol=0.002)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)
