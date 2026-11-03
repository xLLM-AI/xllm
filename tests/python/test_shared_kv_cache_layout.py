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

from dataclasses import replace

import pytest
import torch

from xllm.python.attention.backend import (
    LayerCache,
    SharedKVCacheLayout,
    detect_shared_kv_cache_layout,
)


def _arena_view(arena: torch.Tensor, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    row_bytes = torch.empty((), dtype=dtype).element_size()
    for width in shape[1:]:
        row_bytes *= width
    return arena.narrow(0, offset, shape[0] * row_bytes).view(dtype).view(shape)


def _make_shared_caches(page_bytes: int = 1024) -> list[LayerCache]:
    arena = torch.empty(8 * page_bytes, dtype=torch.uint8)
    full_rows, conv_rows, ssm_rows = {512: (29, 15, 30), 1024: (57, 29, 58), 2048: (113, 57, 114)}[page_bytes]
    key = _arena_view(arena, 0, (full_rows, 1, 1, 64), torch.bfloat16)
    value = _arena_view(arena, 128, (full_rows, 1, 1, 64), torch.bfloat16)
    conv = _arena_view(arena, 0, (conv_rows, 1, 1, 128), torch.bfloat16)
    ssm = _arena_view(arena, 256, (ssm_rows, 1, 1, 32), torch.float32)
    return [
        LayerCache(key=key, value=value),
        LayerCache(key=None, value=None, conv=conv, ssm=ssm),
    ]


def _make_multigroup_caches() -> list[LayerCache]:
    # Eight physical pages, two Full groups, and two Linear layers.  These
    # explicit row counts include the last legal extent in each role view.
    arena = torch.empty(8192, dtype=torch.uint8)
    full_caches = [
        LayerCache(
            key=_arena_view(arena, offset, (49, 1, 1, 64), torch.bfloat16),
            value=_arena_view(arena, offset + 128, (49, 1, 1, 64), torch.bfloat16),
        )
        for offset in (0, 256, 512, 768, 1024)
    ]
    linear_caches = [
        LayerCache(
            key=None,
            value=None,
            conv=_arena_view(arena, offset, (25, 1, 1, 128), torch.bfloat16),
            ssm=_arena_view(arena, offset + 256, (50, 1, 1, 32), torch.float32),
        )
        for offset in (0, 1024)
    ]
    return [full_caches[0], linear_caches[0], *full_caches[1:], linear_caches[1]]


def test_detect_shared_layout_and_translate_full_indices() -> None:
    layout = detect_shared_kv_cache_layout(_make_shared_caches())

    assert layout is not None
    assert layout.page_size_tokens == 1
    assert layout.page_bytes == 1024
    assert layout.logical_page_count == 8
    assert layout.full_row_stride == 8
    assert layout.conv_row_stride == 4
    assert layout.ssm_row_stride == 8
    assert layout.checkpoint_stride == 2
    assert layout.scratch_full_rows == 1
    assert layout.scratch_conv_rows == 1
    assert layout.scratch_ssm_rows == 2

    block_table = torch.tensor([[3, -1]], dtype=torch.int32)
    torch.testing.assert_close(layout.full_block_table(block_table), torch.tensor([[24, -1]], dtype=torch.int32))
    slots = torch.tensor([3, -1], dtype=torch.int32)
    torch.testing.assert_close(layout.full_slot_mapping(slots), torch.tensor([24, -1], dtype=torch.int32))


def test_multigroup_layout_maps_last_full_and_linear_extents() -> None:
    caches = _make_multigroup_caches()
    layout = detect_shared_kv_cache_layout(caches)

    assert layout is not None
    torch.testing.assert_close(
        layout.full_block_table(torch.tensor([[6]], dtype=torch.int32)),
        torch.tensor([[48]], dtype=torch.int32),
    )
    torch.testing.assert_close(layout.linear_state_indices(torch.tensor([6]), role="conv"), torch.tensor([24]))
    torch.testing.assert_close(
        layout.linear_state_indices(torch.tensor([6]), role="ssm", checkpoint_offsets=torch.tensor([1])),
        torch.tensor([49]),
    )
    last_full = caches[-2]
    last_linear = caches[-1]
    assert last_full.key is not None and last_full.value is not None
    assert last_linear.conv is not None and last_linear.ssm is not None
    last_full.key[48].fill_(11)
    last_full.value[48].fill_(12)
    torch.testing.assert_close(last_full.key[48], torch.full_like(last_full.key[48], 11))
    torch.testing.assert_close(last_full.value[48], torch.full_like(last_full.value[48], 12))
    # Once the Full extent is released, the same bytes can serve a Linear
    # extent.  The second checkpoint also fits within the final physical page.
    last_linear.conv[24].fill_(21)
    last_linear.ssm[49].fill_(22)
    assert last_full.key[48].data_ptr() == last_linear.conv[24].data_ptr()
    torch.testing.assert_close(last_full.key[48], torch.full_like(last_full.key[48], 21))
    torch.testing.assert_close(last_linear.ssm[49], torch.full_like(last_linear.ssm[49], 22))


def test_independent_role_storage_keeps_legacy_layout() -> None:
    key = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    conv = torch.empty(8, 1, 1, 128, dtype=torch.bfloat16)
    ssm = torch.empty(16, 1, 1, 32, dtype=torch.float32)

    assert (
        detect_shared_kv_cache_layout(
            [LayerCache(key=key, value=value), LayerCache(key=None, value=None, conv=conv, ssm=ssm)]
        )
        is None
    )


def test_independent_index_cache_keeps_legacy_layout() -> None:
    key = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    conv = torch.empty(8, 1, 1, 128, dtype=torch.bfloat16)
    ssm = torch.empty(16, 1, 1, 32, dtype=torch.float32)
    index = torch.empty(8, 1, 1, 64, dtype=torch.bfloat16)

    assert (
        detect_shared_kv_cache_layout(
            [
                LayerCache(key=key, value=value, index=index),
                LayerCache(key=None, value=None, conv=conv, ssm=ssm),
            ]
        )
        is None
    )


def test_full_only_aliases_do_not_enable_shared_linear_layout() -> None:
    caches = _make_shared_caches()
    linear = caches[1]
    assert linear.conv is not None and linear.ssm is not None
    caches[1] = replace(linear, conv=linear.conv.clone(), ssm=linear.ssm.clone())

    assert detect_shared_kv_cache_layout(caches) is None


@pytest.mark.parametrize("role", ["key", "value", "conv", "ssm"])
def test_partial_shared_arena_is_rejected(role: str) -> None:
    caches = _make_shared_caches()
    layer_index = 0 if role in ("key", "value") else 1
    tensor = getattr(caches[layer_index], role)
    caches[layer_index] = replace(caches[layer_index], **{role: tensor.clone()})

    with pytest.raises(ValueError, match="one complete arena"):
        detect_shared_kv_cache_layout(caches)


@pytest.mark.parametrize("role", ["key", "value", "conv", "ssm"])
def test_incomplete_shared_layer_is_rejected(role: str) -> None:
    caches = _make_shared_caches()
    layer_index = 0 if role in ("key", "value") else 1
    caches[layer_index] = replace(caches[layer_index], **{role: None})

    with pytest.raises(ValueError, match="paired"):
        detect_shared_kv_cache_layout(caches)


@pytest.mark.parametrize("page_bytes", [512, 2048])
def test_noncanonical_physical_page_size_is_rejected(page_bytes: int) -> None:
    with pytest.raises(ValueError, match="static allocator layout"):
        detect_shared_kv_cache_layout(_make_shared_caches(page_bytes))


@pytest.mark.parametrize("role", ["key", "value", "conv", "ssm"])
def test_truncated_role_view_is_rejected(role: str) -> None:
    caches = _make_multigroup_caches()
    layer_index = -2 if role in ("key", "value") else -1
    tensor = getattr(caches[layer_index], role)
    caches[layer_index] = replace(caches[layer_index], **{role: tensor[:-1]})

    with pytest.raises(ValueError, match="inconsistent.*geometry"):
        detect_shared_kv_cache_layout(caches)


def test_shared_role_offsets_are_validated() -> None:
    caches = _make_shared_caches()
    full = caches[0]
    assert full.key is not None
    caches[0] = replace(full, value=full.key)

    with pytest.raises(ValueError, match="static page role offsets"):
        detect_shared_kv_cache_layout(caches)


def test_shared_index_role_is_rejected() -> None:
    caches = _make_shared_caches()
    caches[0] = replace(caches[0], index=torch.empty(1))

    with pytest.raises(ValueError, match="supports only Full key/value and Linear Conv/SSM"):
        detect_shared_kv_cache_layout(caches)


def test_same_byte_width_different_dtype_is_rejected() -> None:
    caches = _make_shared_caches()
    full = caches[0]
    assert full.value is not None
    caches[0] = replace(full, value=full.value.view(torch.float16))

    with pytest.raises(ValueError, match="inconsistent page geometry"):
        detect_shared_kv_cache_layout(caches)


def test_noncontiguous_shared_role_is_rejected() -> None:
    caches = _make_shared_caches()
    full = caches[0]
    assert full.value is not None
    caches[0] = replace(full, value=full.value.view(57, 1, 2, 32).transpose(2, 3))

    with pytest.raises(ValueError, match="contiguous row views"):
        detect_shared_kv_cache_layout(caches)


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
    slots = torch.tensor([0, 127, 128, 255, -1, -2], dtype=torch.int32)
    torch.testing.assert_close(
        layout.full_slot_mapping(slots),
        torch.tensor([0, 127, 512, 639, -1, -2], dtype=torch.int32),
    )
    state = torch.tensor([2, 0, -1, -2], dtype=torch.int32)
    torch.testing.assert_close(
        layout.linear_state_indices(state, role="conv"), torch.tensor([6, 0, -1, -2], dtype=torch.int32)
    )
    torch.testing.assert_close(
        layout.linear_state_indices(state, role="ssm", checkpoint_offsets=torch.tensor([1, 0, 1, 1])),
        torch.tensor([19, 0, -1, -2]),
    )
