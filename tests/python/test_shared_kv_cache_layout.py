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

"""Contracts for Python shared Full/Linear KV-cache layout translation."""

import torch

from xllm.python.attention.backend import (
    LayerCache,
    SharedKVCacheLayout,
    detect_shared_kv_cache_layout,
)


def _make_shared_caches() -> list[LayerCache]:
    page_bytes = 1024
    arena = torch.empty(8 * page_bytes, dtype=torch.uint8)

    def view(offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        row_bytes = torch.empty((), dtype=dtype).element_size()
        for width in shape[1:]:
            row_bytes *= width
        return arena.narrow(0, offset, shape[0] * row_bytes).view(dtype).view(shape)

    key = view(0, (57, 1, 1, 64), torch.bfloat16)
    value = view(128, (57, 1, 1, 64), torch.bfloat16)
    conv = view(0, (29, 1, 1, 128), torch.bfloat16)
    ssm = view(256, (58, 1, 1, 32), torch.float32)
    return [
        LayerCache(key=key, value=value),
        LayerCache(key=None, value=None, conv=conv, ssm=ssm),
    ]


def test_detect_shared_layout_and_translate_full_indices() -> None:
    layout = detect_shared_kv_cache_layout(_make_shared_caches())

    assert layout is not None
    assert layout.page_size_tokens == 1
    assert layout.full_row_stride == 8
    assert layout.conv_row_stride == 4
    assert layout.ssm_row_stride == 8
    assert layout.checkpoint_stride == 2
    assert layout.scratch_full_rows == 1
    assert layout.scratch_conv_rows == 1
    assert layout.scratch_ssm_rows == 2

    block_table = torch.tensor([[3, -1]], dtype=torch.int32)
    assert torch.equal(layout.full_block_table(block_table), torch.tensor([[12, -1]], dtype=torch.int32))
    slots = torch.tensor([3, -1], dtype=torch.int32)
    assert torch.equal(layout.full_slot_mapping(slots), torch.tensor([12, -1], dtype=torch.int32))


def test_independent_role_storage_keeps_legacy_layout() -> None:
    key = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    conv = torch.empty(8, 1, 1, 128, dtype=torch.bfloat16)
    ssm = torch.empty(16, 1, 1, 32, dtype=torch.float32)

    assert detect_shared_kv_cache_layout(
        [LayerCache(key=key, value=value), LayerCache(key=None, value=None, conv=conv, ssm=ssm)]
    ) is None


def test_independent_index_cache_keeps_legacy_layout() -> None:
    key = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    conv = torch.empty(8, 1, 1, 128, dtype=torch.bfloat16)
    ssm = torch.empty(16, 1, 1, 32, dtype=torch.float32)
    index = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)

    assert detect_shared_kv_cache_layout(
        [
            LayerCache(key=key, value=value, index=index),
            LayerCache(key=None, value=None, conv=conv, ssm=ssm),
        ]
    ) is None


def test_layout_preserves_token_offset_and_negative_state_sentinel() -> None:
    layout = SharedKVCacheLayout(
        page_size_tokens=128,
        full_row_stride=4,
        conv_row_stride=3,
        ssm_row_stride=9,
        checkpoint_stride=2,
        logical_page_count=8,
        page_bytes=1024,
    )
    slots = torch.tensor([255, -1], dtype=torch.int32)
    assert torch.equal(layout.full_slot_mapping(slots), torch.tensor([639, -1], dtype=torch.int32))
    state = torch.tensor([2, -1], dtype=torch.int32)
    assert torch.equal(
        layout.linear_state_indices(state, role="ssm"),
        torch.tensor([18, -1], dtype=torch.int32),
    )
