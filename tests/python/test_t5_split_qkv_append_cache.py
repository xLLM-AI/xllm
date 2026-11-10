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

from typing import Callable

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")


def _require_operator() -> Callable[..., tuple[torch.Tensor, ...]]:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("requires an available Ascend NPU")
    from xllm import xllm_export  # noqa: F401

    if not torch.ops.xllm_ops.has_t5_split_qkv_append_cache():
        pytest.skip("requires the A3 T5SplitQkvAppendCache vendor operator")
    return torch.ops.xllm_ops.t5_split_qkv_append_cache


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("length", [0, 128])
@pytest.mark.parametrize("dim", [17, 64])
def test_t5_split_qkv_append_cache(dtype: torch.dtype, length: int, dim: int) -> None:
    op = _require_operator()
    batch, heads = 2, 8
    qkv = torch.randn((batch, 1, 3 * heads * dim), dtype=dtype, device="npu:0")
    past_key = torch.randn((batch, heads, length, dim), dtype=dtype, device="npu:0")
    past_value = torch.randn_like(past_key)

    inputs = (qkv, past_key, past_value)
    originals = tuple(tensor.clone() for tensor in inputs)
    query, key, value = op(*inputs, heads, dim)
    torch.npu.synchronize()

    parts = qkv.reshape(batch, 3, heads, dim)
    expected_query = parts[:, 0].unsqueeze(2).contiguous()
    expected_key = torch.cat((past_key, parts[:, 1].unsqueeze(2)), dim=2)
    expected_value = torch.cat((past_value, parts[:, 2].unsqueeze(2)), dim=2)
    assert query.is_contiguous() and key.is_contiguous() and value.is_contiguous()
    assert len({query.data_ptr(), key.data_ptr(), value.data_ptr()}) == 3
    assert not ({query.data_ptr(), key.data_ptr(), value.data_ptr()} & {tensor.data_ptr() for tensor in inputs})
    torch.testing.assert_close(query, expected_query, rtol=0, atol=0)
    torch.testing.assert_close(key, expected_key, rtol=0, atol=0)
    torch.testing.assert_close(value, expected_value, rtol=0, atol=0)
    for tensor, original in zip(inputs, originals):
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)


def test_t5_split_qkv_append_cache_keeps_previous_outputs() -> None:
    op = _require_operator()
    qkv = torch.randn((2, 1, 3 * 8 * 64), dtype=torch.float16, device="npu:0")
    past_key = torch.randn((2, 8, 128, 64), dtype=qkv.dtype, device=qkv.device)
    past_value = torch.randn_like(past_key)
    inputs = (qkv, past_key, past_value)
    first = op(*inputs, 8, 64)
    saved = tuple(tensor.clone() for tensor in first)
    for tensor in inputs:
        tensor.fill_(3)
    second = op(*inputs, 8, 64)
    torch.npu.synchronize()

    assert not ({tensor.data_ptr() for tensor in first} & {tensor.data_ptr() for tensor in second})
    for output, reference in zip(first, saved):
        torch.testing.assert_close(output, reference, rtol=0, atol=0)
    for output in second:
        assert bool((output == 3).all().item())
