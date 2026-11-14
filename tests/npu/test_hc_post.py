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

from __future__ import annotations

import os
from collections.abc import Callable

import pytest
import torch

HcPost = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


@pytest.fixture(scope="module")
def native_hc_post() -> HcPost:
    if os.environ.get("XLLM_TEST_NPU") != "1":
        pytest.skip("set XLLM_TEST_NPU=1 after allocating an NPU")
    import torch_npu  # noqa: F401

    from xllm import xllm_export  # noqa: F401 - registers native operators
    from xllm.python import initialize_runtime

    initialize_runtime()
    from xllm.python import kernels

    torch.npu.set_device(0)
    return kernels.hc_post


def _inputs(dtype: torch.dtype, hidden: int, tokens: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(41)
    return (
        torch.randn(tokens, 1, hidden, dtype=dtype, device="npu"),
        torch.randn(tokens, 1, 4, hidden, dtype=dtype, device="npu"),
        torch.randn(tokens, 1, 4, dtype=torch.float32, device="npu"),
        torch.randn(tokens, 1, 4, 4, dtype=torch.float32, device="npu"),
    )


def _reference(x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor) -> torch.Tensor:
    mixed = torch.einsum("bskh,bskj->bsjh", residual.float(), comb.float())
    return (x.float().unsqueeze(-2) * post.float().unsqueeze(-1) + mixed).to(x.dtype)


def _check(actual: torch.Tensor, expected: torch.Tensor) -> None:
    tolerance = {torch.float32: 2e-5, torch.float16: 3e-3, torch.bfloat16: 3e-2}[expected.dtype]
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [64, 128, 256, 2048, 5120, 7168])
@pytest.mark.parametrize("tokens", [1, 3])
@torch.inference_mode()
def test_hc_post_aligned_columns(native_hc_post: HcPost, dtype: torch.dtype, hidden: int, tokens: int) -> None:
    args = _inputs(dtype, hidden, tokens)
    _check(native_hc_post(*args), _reference(*args))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden", [128, 5120, 7168])
@torch.inference_mode()
def test_hc_post_graph_replay(native_hc_post: HcPost, dtype: torch.dtype, hidden: int) -> None:
    args = _inputs(dtype, hidden, 1)
    stream = torch.npu.Stream()
    stream.wait_stream(torch.npu.current_stream())
    with torch.npu.stream(stream):
        for _ in range(3):
            native_hc_post(*args)
    torch.npu.current_stream().wait_stream(stream)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        actual = native_hc_post(*args)
    torch.npu.current_stream().wait_stream(stream)
    for _ in range(3):
        args[0].add_(0.125)
        args[1].mul_(0.75)
        torch.npu.synchronize()
        graph.replay()
        torch.npu.synchronize()
        _check(actual, _reference(*args))
