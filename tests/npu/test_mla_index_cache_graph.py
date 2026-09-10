# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Real NPU cache/capture regressions; run separately from tests/python stubs.

Requires XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE. These opt-ins keep
ordinary CPU collection from loading an unselected native build or NPU device.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def npu_device() -> torch.device:
    library = os.environ.get("XLLM_TEST_NATIVE_LIBRARY")
    device_index = os.environ.get("XLLM_TEST_NPU_DEVICE")
    if not library or device_index is None:
        pytest.skip("set XLLM_TEST_NATIVE_LIBRARY and XLLM_TEST_NPU_DEVICE for real NPU tests")
    pytest.importorskip("torch_npu")
    import xllm.python as runtime

    if not hasattr(runtime, "initialize_runtime"):
        pytest.fail("CPU package stubs are active; run tests/npu in a separate pytest process")
    assert Path(library).is_file(), f"native operator library does not exist: {library}"
    torch.ops.load_library(library)
    torch.npu.set_device(int(device_index))
    runtime.initialize_runtime()
    return torch.device(f"npu:{device_index}")


def _apply_expected(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    for slot, value in zip(slots.tolist(), values, strict=True):
        if slot >= 0:
            cache[slot].copy_(value)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float16])
def test_native_index_scatter_skips_negative_slots(npu_device: torch.device, dtype: torch.dtype) -> None:
    from xllm.python import kernels

    for width in (1, 128):
        expected = torch.full((128, width), -77, dtype=dtype)
        actual = expected.to(npu_device)
        slots = torch.tensor([0, -1, 127, -1], dtype=torch.int64)
        values = torch.arange(4 * width).reshape(4, width).remainder(61).to(dtype)
        kernels.scatter_nd_update(actual, slots.to(npu_device).view(-1, 1), values.to(npu_device))
        _apply_expected(expected, slots, values)
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.int8, torch.bfloat16, torch.float16])
def test_index_context_capture_replays_changing_slots_and_scales(npu_device: torch.device, dtype: torch.dtype) -> None:
    from xllm.python.attention.backend import LayerCache
    from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
    from xllm.python.model_executor.forward_context import ForwardContext, forward_context

    backend = NpuPagedAttentionBackend(1, 1, 128, 1.0, -1, True, npu_device, torch.bfloat16)
    cache = LayerCache(
        key=torch.zeros(8, 16, 1, 128, dtype=torch.bfloat16, device=npu_device),
        value=torch.zeros(8, 16, 1, 128, dtype=torch.bfloat16, device=npu_device),
        index=torch.full((8, 16, 1, 128), -77, dtype=dtype, device=npu_device),
        indexer_scale=torch.full((8, 16, 1, 1), -77, dtype=torch.float16, device=npu_device),
    )
    backend.bind_kv_caches([cache])
    slots = torch.tensor([0, 17, -1, -1], dtype=torch.int64, device=npu_device)
    metadata = SimpleNamespace(
        slot_mapping=slots,
        block_table=torch.zeros(4, 8, dtype=torch.int32, device=npu_device),
        kv_seq_lens=torch.ones(4, dtype=torch.int32, device=npu_device),
        kv_seq_lens_host_values=[1, 1, 1, 1],
        q_cu_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
        has_kv_shard=False,
    )
    values = torch.zeros(4, 128, dtype=dtype, device=npu_device)
    scales = torch.ones(4, 1, dtype=torch.float16, device=npu_device)
    context = ForwardContext(backend, npu_device, metadata, [cache])
    with forward_context(context):
        backend.prepare(metadata)
        index_context = backend.mla_index_context(SimpleNamespace(layer_id=0))
        index_context.update_index_cache(values, scales)
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        stream = torch.npu.Stream()
        with torch.npu.graph(graph, stream=stream):
            index_context.update_index_cache(values, scales)
        torch.npu.synchronize()

        expected_key = cache.index.cpu().view(128, 128)
        expected_scale = cache.index_scale.cpu().view(128, 1)
        patterns = ([34, -1, 0, 17], [-1, -1, -1, -1], [127, 126, 125, 0], [-1, 0, -1, 1])
        for step, pattern in enumerate(patterns * 3, 1):
            host_slots = torch.tensor(pattern)
            host_values = (torch.arange(512).reshape(4, 128).remainder(61) + step).to(dtype)
            host_scales = (torch.arange(4).reshape(4, 1) + step).to(torch.float16)
            slots.copy_(host_slots)
            values.copy_(host_values)
            scales.copy_(host_scales)
            torch.npu.synchronize()
            graph.replay()
            torch.npu.synchronize()
            _apply_expected(expected_key, host_slots, host_values)
            _apply_expected(expected_scale, host_slots, host_scales)
            torch.testing.assert_close(cache.index.cpu().view(128, 128), expected_key, rtol=0, atol=0)
            torch.testing.assert_close(cache.index_scale.cpu().view(128, 1), expected_scale, rtol=0, atol=0)
