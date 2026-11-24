# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU correctness tests for GLM and DeepSeek TND Q/V projections."""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

_DTYPE = torch.bfloat16
_RTOL, _ATOL = 1e-2, 2e-2
_Project = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@pytest.fixture(scope="module")
def project() -> _Project:
    assert torch.npu.is_available(), "MLA projection tests require an Ascend NPU"
    from xllm_kernel.ops.gemm import matmul_ein_sum

    return matmul_ein_sum


def _make_inputs(
    tokens: int,
    layout: str,
    heads: int,
    q_dim: int,
    v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260922 + tokens + heads)
    input_dim, output_dim = (q_dim, 512) if layout.startswith("q_") else (512, v_dim)
    if layout in ("q_split", "q_offset"):
        offset = int(layout == "q_offset")
        backing = torch.randn((tokens + offset, heads, input_dim + 64), generator=generator, dtype=_DTYPE).to("npu")
        x = backing[offset:, :, :input_dim]
    elif layout == "v_narrow":
        backing = torch.randn((tokens, heads * 2, input_dim), generator=generator, dtype=_DTYPE).to("npu")
        x = backing.narrow(1, heads, heads)
    elif layout in ("q_contiguous", "v_contiguous"):
        backing = torch.randn((tokens, heads, input_dim), generator=generator, dtype=_DTYPE).to("npu")
        x = backing
    else:
        raise ValueError(f"Unknown layout: {layout}")
    weight = torch.randn((heads, input_dim, output_dim), generator=generator)
    weight = (weight / math.sqrt(input_dim)).to(dtype=_DTYPE, device="npu")
    return backing, x, weight


@pytest.mark.parametrize("tokens", (1, 4, 8, 128))
@pytest.mark.parametrize("layout", ("q_split", "q_offset", "q_contiguous", "v_contiguous", "v_narrow"))
@pytest.mark.parametrize("heads,q_dim,v_dim", ((4, 192, 256), (8, 128, 128)), ids=("glm", "deepseek"))
@pytest.mark.parametrize("mode", ("eager", "graph"))
def test_projection(
    project: _Project,
    tokens: int,
    layout: str,
    heads: int,
    q_dim: int,
    v_dim: int,
    mode: str,
) -> None:
    backing, x, weight = _make_inputs(tokens, layout, heads, q_dim, v_dim)
    original_backing, original_weight = backing.cpu(), weight.cpu()
    graph = None
    if mode == "graph":
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                project(x, weight)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            actual = project(x, weight)
        addresses = (x.data_ptr(), weight.data_ptr(), actual.data_ptr())

    previous = None
    for scale in (1.0, -0.5):
        expected_backing = original_backing * scale
        expected_weight = original_weight * (2.0 - scale)
        backing.copy_(expected_backing)
        weight.copy_(expected_weight)
        torch.npu.synchronize()
        if graph is None:
            actual = project(x, weight)
        else:
            graph.replay()
        torch.npu.synchronize()

        # The FP32 reference uses the actual quantized inputs, not the RNG source.
        reference = torch.einsum("thd,hdo->tho", x.cpu().float(), weight.cpu().float())
        if layout.startswith("q_"):
            legacy = torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)
        else:
            legacy = torch_npu.npu_transpose_batchmatmul(x.transpose(0, 1), weight, perm_y=(1, 0, 2))
        assert actual.shape == reference.shape
        assert actual.dtype == x.dtype and actual.device == x.device
        assert actual.is_contiguous()
        current = actual.cpu()
        assert torch.isfinite(current).all()
        torch.testing.assert_close(current.float(), reference, rtol=_RTOL, atol=_ATOL)
        torch.testing.assert_close(actual, legacy, rtol=_RTOL, atol=_ATOL)
        if tokens == 4 and mode == "eager" and scale == 1.0 and layout != "v_narrow":
            baseline = torch.ops.npu.npu_transpose_batchmatmul(
                x, weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
            )
            torch.testing.assert_close(baseline.cpu().float(), reference, rtol=_RTOL, atol=_ATOL)
            max_abs = (current.float() - baseline.cpu().float()).abs().max().item()
            limit = 0 if layout.startswith("q_") else 0.25
            assert max_abs <= limit, f"ATB/TBMM {layout} max_abs={max_abs}"
        torch.testing.assert_close(backing.cpu(), expected_backing, rtol=0, atol=0)
        torch.testing.assert_close(weight.cpu(), expected_weight, rtol=0, atol=0)
        if graph is not None:
            assert addresses == (x.data_ptr(), weight.data_ptr(), actual.data_ptr())
        if previous is not None:
            assert not torch.equal(current, previous), "Projection reused stale input values"
        previous = current
    torch.npu.synchronize()


@pytest.mark.parametrize("layout", ("q_contiguous", "q_split", "q_offset", "v_contiguous"))
@pytest.mark.parametrize("heads,q_dim,v_dim", ((4, 192, 256), (8, 128, 128)), ids=("glm", "deepseek"))
def test_atb_ein_sum_projection(
    project: _Project,
    layout: str,
    heads: int,
    q_dim: int,
    v_dim: int,
) -> None:
    _, x, weight = _make_inputs(4, layout, heads, q_dim, v_dim)
    actual = project(x, weight)
    baseline = torch.ops.npu.npu_transpose_batchmatmul(
        x, weight, perm_x1=(1, 0, 2), perm_x2=(0, 1, 2), perm_y=(1, 0, 2)
    )
    torch.npu.synchronize()
    reference = torch.einsum("thd,hdo->tho", x.cpu().float(), weight.cpu().float())
    assert actual.shape == reference.shape
    assert actual.dtype == x.dtype and actual.device == x.device and actual.is_contiguous()
    torch.testing.assert_close(actual.cpu().float(), reference, rtol=_RTOL, atol=_ATOL)
    torch.testing.assert_close(baseline.cpu().float(), reference, rtol=_RTOL, atol=_ATOL)
    max_abs = (actual.float() - baseline.float()).abs().max().item()
    assert max_abs <= (0 if layout.startswith("q_") else 0.25), f"ATB/TBMM {layout} max_abs={max_abs}"


@pytest.mark.parametrize("tokens", (1, 4, 8))
@pytest.mark.parametrize("layout", ("q_split", "q_offset", "v_contiguous", "v_narrow"))
@pytest.mark.parametrize("mode", ("eager", "graph"))
def test_projection_out_updates_caller_buffer(project: _Project, tokens: int, layout: str, mode: str) -> None:
    from xllm_kernel.ops.gemm import matmul_ein_sum_out as atb_matmul_ein_sum_out

    backing, x, weight = _make_inputs(tokens, layout, 4, 192, 256)
    original = backing.cpu()
    output = torch.empty((tokens, 4, weight.shape[2]), dtype=_DTYPE, device=x.device)
    output_address = output.data_ptr()
    graph = None
    if mode == "graph":
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(3):
                atb_matmul_ein_sum_out(x, weight, output)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph, stream=stream):
            result = atb_matmul_ein_sum_out(x, weight, output)

    previous = None
    for scale in (1.0, -0.5):
        backing.copy_(original * scale)
        torch.npu.synchronize()
        if graph is None:
            result = atb_matmul_ein_sum_out(x, weight, output)
        else:
            graph.replay()
        expected = project(x, weight)
        torch.npu.synchronize()
        assert result.data_ptr() == output.data_ptr() == output_address
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        reference = torch.einsum("thd,hdo->tho", x.cpu().float(), weight.cpu().float())
        torch.testing.assert_close(output.cpu().float(), reference, rtol=_RTOL, atol=_ATOL)
        current = output.cpu()
        if previous is not None:
            assert not torch.equal(current, previous), "Output buffer retained stale projection values"
        previous = current


def test_alternating_graphs_keep_independent_outputs_and_streams(project: _Project) -> None:
    from xllm_kernel.ops.gemm import matmul_ein_sum_out

    captures = []
    for tokens in (1, 17):
        backing, value, weight = _make_inputs(tokens, "q_offset", 4, 192, 256)
        output_storage = torch.full((tokens * 4 * 512 + 1,), float("nan"), device="npu", dtype=_DTYPE)
        output = output_storage[1:].view(tokens, 4, 512)
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            matmul_ein_sum_out(value, weight, output)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                result = matmul_ein_sum_out(value, weight, output)
        stream.synchronize()
        assert result is output
        captures.append((graph, stream, backing, value, weight, output, output.data_ptr()))

    for _ in range(4):
        for graph, stream, backing, value, weight, output, address in reversed(captures):
            with torch.npu.stream(stream):
                backing.mul_(-0.5)
                weight.mul_(0.75)
                graph.replay()
            stream.synchronize()
            reference = torch.einsum("thd,hdo->tho", value.cpu().float(), weight.cpu().float())
            torch.testing.assert_close(output.cpu().float(), reference, rtol=_RTOL, atol=_ATOL)
            assert output.data_ptr() == address
            torch.testing.assert_close(output, project(value, weight), rtol=0, atol=0)


def test_standalone_execution_without_host_import(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = """
import importlib.abc
import sys
from typing import Any
class BlockHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> None:
        if fullname.split('.')[0] == 'xllm':
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockHost())
import torch
import torch_npu
from xllm_kernel import initialize
from xllm_kernel.ops.gemm import matmul_ein_sum, matmul_ein_sum_out
plans = initialize(device='npu', operations=('matmul_ein_sum', 'matmul_ein_sum_out'))
assert all(plan.spec.binding == 'atb' for plan in plans.values())
x = torch.randn(3, 4, 128, dtype=torch.bfloat16, device='npu')
w = torch.randn(4, 128, 64, dtype=torch.bfloat16, device='npu') / 16
out = torch.empty(3, 4, 64, dtype=x.dtype, device=x.device)
actual = matmul_ein_sum(x, w)
assert matmul_ein_sum_out(x, w, out) is out
torch.testing.assert_close(out, actual, rtol=0, atol=0)
reference = torch.einsum('thd,hdo->tho', x.cpu().float(), w.cpu().float())
torch.testing.assert_close(out.cpu().float(), reference, rtol=0.01, atol=0.02)
"""
    subprocess.run([sys.executable, "-c", script], cwd=tmp_path, check=True)


@pytest.mark.parametrize("overlap", ("input", "weight"))
def test_out_rejects_operand_overlap_before_writing(project: _Project, overlap: str) -> None:
    from xllm_kernel.ops.gemm import matmul_ein_sum_out

    value = torch.randn((4, 4, 4), dtype=_DTYPE, device="npu")
    weight = torch.randn_like(value)
    out = value if overlap == "input" else weight
    before = out.cpu()
    with pytest.raises(RuntimeError, match="single memory location"):
        matmul_ein_sum_out(value, weight, out)
    torch.testing.assert_close(out.cpu(), before, rtol=0, atol=0)
