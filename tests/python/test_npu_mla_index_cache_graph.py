# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real NPU scatter/cache regressions; conftest owns native initialization.

Migrated native-scatter/context cases keep XLLM_TEST_NPU_DEVICE as a resource
opt-in. The former native-library bootstrap is replaced by the normal runner.
Static-helper replay still executes without that opt-in.
"""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import partial
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("torch_npu", reason="MLA index-cache graph tests require torch_npu")

from xllm.python import kernels
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.model_executor.forward_context import ForwardContext, forward_context


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    assert torch.npu.is_available(), "MLA index-cache tests require an available NPU"
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if device_index is not None:
        torch.npu.set_device(int(device_index))
    return torch.device("npu", torch.npu.current_device())


def _apply_expected(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    for slot, value in zip(slots.tolist(), values, strict=True):
        if slot >= 0:
            cache[slot].copy_(value)


@pytest.mark.parametrize("dtype", (torch.int8, torch.bfloat16, torch.float16))
@pytest.mark.parametrize("width", (1, 128))
def test_native_index_scatter_skips_negative_slots(npu_device: torch.device, dtype: torch.dtype, width: int) -> None:
    if os.environ.get("XLLM_TEST_NPU_DEVICE") is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE to execute migrated native-scatter tests")
    expected = torch.full((128, width), -77, dtype=dtype)
    actual = expected.to(npu_device)
    slots = torch.tensor([0, -1, 127, -1], dtype=torch.int64)
    values = torch.arange(4 * width).reshape(4, width).remainder(61).to(dtype)
    kernels.scatter_nd_update(actual, slots.to(npu_device).view(-1, 1), values.to(npu_device))
    _apply_expected(expected, slots, values)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "owner,dtype,width,with_scales",
    (
        ("static-helper", torch.float16, 2, False),
        ("static-helper", torch.int8, 2, True),
        ("context", torch.int8, 128, True),
        ("context", torch.bfloat16, 128, True),
        ("context", torch.float16, 128, True),
    ),
)
def test_mla_index_cache_update_replays_dynamic_valid_rows(
    npu_device: torch.device, owner: str, dtype: torch.dtype, width: int, with_scales: bool
) -> None:
    if owner == "context" and os.environ.get("XLLM_TEST_NPU_DEVICE") is None:
        pytest.skip("set XLLM_TEST_NPU_DEVICE to execute migrated context tests")
    blocks, block_size = (2, 4) if owner == "static-helper" else (8, 16)
    initial = torch.full((blocks, block_size, 1, width), -77, dtype=dtype)
    scale_shape = (blocks, block_size, 1) if owner == "static-helper" else (blocks, block_size, 1, 1)
    initial_scale = torch.full(scale_shape, -77, dtype=torch.float16) if with_scales else None
    cache = initial.to(npu_device)
    scale_cache = initial_scale.to(npu_device) if initial_scale is not None else None
    host_slots = torch.tensor([-1] * 4 if owner == "static-helper" else [0, 17, -1, -1], dtype=torch.int64)
    slots = host_slots.to(npu_device)
    values = torch.zeros(4, width, dtype=dtype, device=npu_device)
    scales = torch.ones(4, 1, dtype=torch.float16, device=npu_device) if with_scales else None
    patterns = ([4, 6, 7, -1], [4, 5, 6, 7], [5, -1, 4, 6], [-1] * 4, [0, -1, 7, -1])
    context = None
    update = partial(NpuPagedAttentionBackend._update_mla_index_cache, cache, scale_cache, slots)
    if owner == "context":
        layer_cache = LayerCache(
            key=torch.zeros(blocks, block_size, 1, 128, dtype=torch.bfloat16, device=npu_device),
            value=torch.zeros(blocks, block_size, 1, 128, dtype=torch.bfloat16, device=npu_device),
            index=cache,
            indexer_scale=scale_cache,
        )
        backend = NpuPagedAttentionBackend(1, 1, 128, 1.0, -1, True, npu_device, torch.bfloat16)
        backend.bind_kv_caches([layer_cache])
        metadata = SimpleNamespace(
            slot_mapping=slots,
            block_table=torch.zeros(4, blocks, dtype=torch.int32, device=npu_device),
            kv_seq_lens=torch.ones(4, dtype=torch.int32, device=npu_device),
            kv_seq_lens_host_values=[1] * 4,
            q_cu_seq_lens=None,
            expanded_decode_metadata=None,
            is_prefill=False,
            is_chunked_prefill=False,
            has_kv_shard=False,
        )
        context = ForwardContext(backend, npu_device, metadata, [layer_cache])
        patterns = ([34, -1, 0, 17], [-1] * 4, [127, 126, 125, 0], [-1, 0, -1, 1]) * 3
    # CPU expectations precede all updates and never come from actual cache state.
    expected = initial.clone().view(-1, width)
    expected_scale = initial_scale.clone().view(-1, 1) if initial_scale is not None else None
    _apply_expected(expected, host_slots, torch.zeros(4, width, dtype=dtype))
    if expected_scale is not None:
        _apply_expected(expected_scale, host_slots, torch.ones(4, 1, dtype=torch.float16))
    with forward_context(context) if context is not None else nullcontext():
        if owner == "context":
            backend.prepare(metadata)
            index_context = backend.mla_index_context(SimpleNamespace(layer_id=0))
            assert index_context.index_cache is cache and index_context.index_cache_scale is scale_cache
            assert index_context.slot_mapping is slots
            update = index_context.update_index_cache
        update(values, scales)
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu().view(-1, width), expected, rtol=0, atol=0)
        if scale_cache is not None:
            torch.testing.assert_close(scale_cache.cpu().view(-1, 1), expected_scale, rtol=0, atol=0)
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=stream):
            update(values, scales)
        torch.npu.synchronize()
        torch.testing.assert_close(cache.cpu().view(-1, width), expected, rtol=0, atol=0)
        if scale_cache is not None:
            torch.testing.assert_close(scale_cache.cpu().view(-1, 1), expected_scale, rtol=0, atol=0)
        for step, pattern in enumerate(patterns, 1):
            host_slots = torch.tensor(pattern, dtype=torch.int64)
            host_values = (torch.arange(4 * width).view(4, width).remainder(61) + step).to(dtype)
            host_scales = torch.arange(4, dtype=torch.float16).view(4, 1) + step
            slots.copy_(host_slots)
            values.copy_(host_values)
            if scales is not None:
                scales.copy_(host_scales)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            _apply_expected(expected, host_slots, host_values)
            torch.testing.assert_close(cache.cpu().view(-1, width), expected, rtol=0, atol=0)
            if scale_cache is not None:
                _apply_expected(expected_scale, host_slots, host_scales)
                torch.testing.assert_close(scale_cache.cpu().view(-1, 1), expected_scale, rtol=0, atol=0)
