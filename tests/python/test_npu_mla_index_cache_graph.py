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

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("torch_npu", reason="MLA index-cache graph tests require torch_npu")

if not hasattr(torch, "npu") or not torch.npu.is_available():
    pytest.skip("MLA index-cache graph tests require an available NPU", allow_module_level=True)


def _load_xllm_export() -> None:
    artifact = os.environ.get("XLLM_EXPORT_PATH")
    if artifact is None:
        pytest.skip("XLLM_EXPORT_PATH must identify the native extension", allow_module_level=True)

    artifact_path = Path(artifact).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"xllm_export artifact does not exist: {artifact_path}")
    spec = importlib.util.spec_from_file_location("xllm_export", artifact_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to create import spec for {artifact_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["xllm_export"] = module
    spec.loader.exec_module(module)


_load_xllm_export()
from xllm.python import kernels  # noqa: E402
from xllm.python.kernels_npu.sparse_attention import scatter_nd_update  # noqa: E402

kernels.scatter_nd_update = scatter_nd_update

from xllm.python.attention.npu_paged_attention import (  # noqa: E402
    NpuPagedAttentionBackend,
)


def _apply_reference_update(
    cache: torch.Tensor,
    scale_cache: torch.Tensor | None,
    slots: torch.Tensor,
    values: torch.Tensor,
    scales: torch.Tensor | None,
) -> None:
    cache_view = cache.view(-1, cache.size(-1))
    scale_view = scale_cache.view(-1, scale_cache.size(-1)) if scale_cache is not None else None
    for row, slot in enumerate(slots.tolist()):
        if slot < 0:
            continue
        cache_view[slot].copy_(values[row])
        if scale_view is not None and scales is not None:
            scale_view[slot].copy_(scales[row])


@pytest.mark.parametrize(
    ("cache_dtype", "with_scales"),
    [
        pytest.param(torch.float16, False, id="float"),
        pytest.param(torch.int8, True, id="w8a8"),
    ],
)
def test_mla_index_cache_update_replays_dynamic_valid_rows(
    cache_dtype: torch.dtype,
    with_scales: bool,
) -> None:
    device = torch.device("npu", torch.npu.current_device())
    initial_cache = torch.arange(16, dtype=cache_dtype).view(2, 4, 1, 2)
    initial_scale_cache = torch.arange(8, dtype=torch.float16).view(2, 4, 1) if with_scales else None
    cache = initial_cache.to(device)
    scale_cache = initial_scale_cache.to(device) if initial_scale_cache is not None else None
    static_slots = torch.full((4,), -1, dtype=torch.int64, device=device)
    static_values = torch.zeros((4, 2), dtype=cache_dtype, device=device)
    static_scales = torch.zeros((4, 1), dtype=torch.float16, device=device) if with_scales else None

    stream = torch.npu.Stream()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph, stream=stream):
        NpuPagedAttentionBackend._update_mla_index_cache(
            cache,
            scale_cache,
            static_slots,
            static_values,
            static_scales,
        )
    torch.npu.synchronize()

    block_size = initial_cache.size(1)
    expected_cache = initial_cache.clone()
    expected_scale_cache = initial_scale_cache.clone() if initial_scale_cache is not None else None
    replay_slots = (
        [4, 6, 7, -1],
        [4, 5, 6, 7],
        [5, -1, 4, 6],
        [-1, -1, -1, -1],
    )
    for replay_id, slots in enumerate(replay_slots):
        value_base = 20 * (replay_id + 1)
        host_slots = torch.tensor(slots, dtype=torch.int64)
        host_values = torch.arange(value_base, value_base + 8, dtype=cache_dtype).view(4, 2)
        host_scales = torch.arange(value_base, value_base + 4, dtype=torch.float16).view(4, 1) if with_scales else None
        static_slots.copy_(host_slots)
        static_values.copy_(host_values)
        if static_scales is not None and host_scales is not None:
            static_scales.copy_(host_scales)

        graph.replay()
        torch.npu.synchronize()

        _apply_reference_update(
            expected_cache,
            expected_scale_cache,
            host_slots,
            host_values,
            host_scales,
        )
        torch.testing.assert_close(
            cache.cpu().view(-1, cache.size(-1))[block_size:],
            expected_cache.view(-1, expected_cache.size(-1))[block_size:],
        )
        if scale_cache is not None and expected_scale_cache is not None:
            torch.testing.assert_close(
                scale_cache.cpu().view(-1, scale_cache.size(-1))[block_size:],
                expected_scale_cache.view(-1, expected_scale_cache.size(-1))[block_size:],
            )
