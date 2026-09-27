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

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

_DTYPE = torch.bfloat16
_RTOL, _ATOL = 1e-2, 2e-2
_Project = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@pytest.fixture(scope="module")
def project() -> _Project:
    assert torch.npu.is_available(), "MLA projection tests require an Ascend NPU"
    import xllm

    _ = xllm.xllm_export
    from xllm.python.kernels_npu.linear import atb_matmul_ein_sum

    return atb_matmul_ein_sum


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
