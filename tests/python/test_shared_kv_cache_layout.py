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
    CacheRowTable,
    LayerCache,
    SharedKVCacheLayout,
    detect_shared_kv_cache_layout,
    normalize_layer_caches,
)


def _arena_view(arena: torch.Tensor, offset: int, shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    row_bytes = torch.empty((), dtype=dtype).element_size()
    for width in shape[1:]:
        row_bytes *= width
    return arena.narrow(0, offset, shape[0] * row_bytes).view(dtype).view(shape)


def _identity_slots(num_slots: int) -> torch.Tensor:
    return torch.arange(num_slots, dtype=torch.int32)


def _make_slab_caches() -> list[LayerCache]:
    # Four 240-byte slabs contain dense 8-byte Full rows, 20-byte Conv
    # rows, and 16-byte SSM rows. Logical capacity is independent of slabs.
    arena = torch.zeros(960, dtype=torch.uint8)
    caches = []
    for layer in range(2):
        full_rows = torch.zeros(32, dtype=torch.int32)
        full_rows[1:3] = torch.tensor([30 + 4 * layer, 32 + 4 * layer])
        linear_rows = torch.zeros(2, 32, dtype=torch.int32)
        linear_rows[0, 1:3] = torch.tensor([24 + 2 * layer, 25 + 2 * layer])
        linear_rows[1, 1:3] = torch.tensor([45 + 6 * layer, 48 + 6 * layer])
        caches.append(
            LayerCache(
                key=_arena_view(arena, 0, (120, 2, 1, 2), torch.bfloat16),
                value=_arena_view(arena, 8, (119, 2, 1, 2), torch.bfloat16),
                full_block_rows=full_rows,
                shared_checkpoint_stride=3,
            )
        )
        caches.append(
            LayerCache(
                key=None,
                value=None,
                conv=_arena_view(arena, 0, (48, 5, 2), torch.bfloat16),
                ssm=_arena_view(arena, 0, (60, 1, 2, 2), torch.float32),
                linear_state_slots=linear_rows,
                shared_checkpoint_stride=3,
            )
        )
    return caches


def _make_mirrored_caches(*, slice_bytes: int = 96) -> list[LayerCache]:
    bank_bytes = 4 * slice_bytes
    arena = torch.zeros(2 * bank_bytes, dtype=torch.uint8)
    descriptor = {
        "shared_bank_bytes": bank_bytes,
        "shared_slice_bytes": slice_bytes,
        "shared_checkpoint_stride": 3,
        "shared_conv_scratch_row": 0,
        "shared_ssm_scratch_row": 0,
        "shared_full_read_padding_page": 6,
    }
    full_rows = bank_bytes // 8 - 3
    caches = [
        LayerCache(
            key=_arena_view(arena, bank * bank_bytes + lane * 16, (full_rows, 2, 1, 2), torch.bfloat16),
            value=_arena_view(arena, bank * bank_bytes + lane * 16 + 8, (full_rows, 2, 1, 2), torch.bfloat16),
            **descriptor,
        )
        for bank in range(2)
        for lane in range(2)
    ]
    for group in range(2):
        rows = torch.zeros(2, 32, dtype=torch.int32)
        rows[:, 1] = torch.tensor([10 + 2 * group, 12 + 3 * group])
        caches.append(
            LayerCache(
                key=None,
                value=None,
                conv=_arena_view(arena, 0, (arena.numel() // 20, 5, 2), torch.bfloat16),
                ssm=_arena_view(arena, 0, (arena.numel() // 16, 1, 2, 2), torch.float32),
                linear_state_slots=rows,
                **descriptor,
            )
        )
    return caches


@pytest.mark.parametrize("slice_bytes", [96, 192])
def test_mirrored_layout_exposes_fixed_bank_lanes_and_dense_linear_views(slice_bytes: int) -> None:
    caches = _make_mirrored_caches(slice_bytes=slice_bytes)
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    assert layout.bank_bytes == 4 * slice_bytes
    assert layout.slot_bytes == slice_bytes
    assert layout.full_read_padding_row == 6
    assert layout.scratch_conv_offset == layout.scratch_ssm_offset == 0
    assert layout.scratch_conv_rows == 1
    assert layout.scratch_ssm_rows == 3
    common_rows = layout.bank_bytes // 8 - 3
    for index, cache in enumerate(caches[:4]):
        bank, lane = divmod(index, 2)
        assert cache.key.shape[0] == cache.value.shape[0] == common_rows
        assert cache.key.storage_offset() * cache.key.element_size() == bank * layout.bank_bytes + 16 * lane
        assert cache.value.data_ptr() - cache.key.data_ptr() == 8
    state = torch.tensor([-1, 0, 1], dtype=torch.int32)
    for group, cache in enumerate(caches[4:]):
        assert cache.conv.storage_offset() == cache.ssm.storage_offset() == 0
        assert layout.linear_state_indices(state, role="conv", slots=cache.linear_state_slots).tolist() == [
            -1,
            0,
            10 + 2 * group,
        ]
        assert layout.linear_state_indices(state, role="ssm", slots=cache.linear_state_slots).tolist() == [
            -1,
            0,
            12 + 3 * group,
        ]


def test_mirrored_padding_stays_finite_after_linear_scratch_is_poisoned() -> None:
    caches = _make_mirrored_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    for cache in caches[4:]:
        cache.conv[0].fill_(torch.nan)
        cache.ssm[:3].fill_(torch.nan)
    for cache in caches[:4]:
        assert torch.isfinite(cache.key[layout.full_read_padding_row]).all()
        assert torch.isfinite(cache.value[layout.full_read_padding_row]).all()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("shared_slice_bytes", 192, "one native geometry"),
        ("shared_full_read_padding_page", 45, "in-range"),
    ],
)
def test_mirrored_layout_rejects_inconsistent_native_geometry(field: str, value: int, error: str) -> None:
    caches = _make_mirrored_caches()
    caches[0] = replace(caches[0], **{field: value})
    with pytest.raises(ValueError, match=error):
        detect_shared_kv_cache_layout(caches)


def test_normalize_mirrored_tuple_preserves_zero_scratch_descriptors() -> None:
    cache = _make_mirrored_caches()[0]
    raw = (cache.key, cache.value, *(None for _ in range(12)), 3, 384, 96, 0, 0, 6)
    normalized = normalize_layer_caches([raw])[0]
    assert normalized.shared_bank_bytes == 384
    assert normalized.shared_slice_bytes == 96
    assert normalized.shared_conv_scratch_row == normalized.shared_ssm_scratch_row == 0
    assert normalized.shared_full_read_padding_page == 6


def test_slab_full_mapping_is_per_layer_and_preserves_token_offsets() -> None:
    caches = _make_slab_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    assert layout.num_slots == 32
    assert layout.num_slabs == 4
    blocks = torch.tensor([[1, 2, 0, -1]], dtype=torch.int32)
    torch.testing.assert_close(
        layout.full_block_table(blocks, caches[0].full_block_rows),
        torch.tensor([[30, 32, 6, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        layout.full_block_table(blocks, caches[2].full_block_rows),
        torch.tensor([[34, 36, 6, -1]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        layout.full_slot_mapping(torch.tensor([2, 3, 4, -1]), caches[0].full_block_rows),
        torch.tensor([60, 61, 64, -1]),
    )
    # Both operands use the K-row index; V's one-row view offset selects
    # the adjacent allocation without changing FIA's common block table.
    assert caches[0].value[30].data_ptr() == caches[0].key[31].data_ptr()


def test_slab_read_padding_stays_finite_after_writable_scratch_is_poisoned() -> None:
    caches = _make_slab_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    full, linear = caches[:2]
    assert full.key is not None and full.value is not None
    assert linear.conv is not None and linear.ssm is not None
    full.key[0].fill_(float("nan"))
    full.value[0].fill_(float("nan"))
    linear.conv[0].fill_(float("nan"))
    linear.ssm[: layout.checkpoint_stride].fill_(float("nan"))

    physical = layout.full_block_table(torch.tensor([0], dtype=torch.int32), full.full_block_rows)
    assert physical.item() == layout.full_read_padding_row == 6
    torch.testing.assert_close(full.key[physical], torch.zeros_like(full.key[physical]))
    torch.testing.assert_close(full.value[physical], torch.zeros_like(full.value[physical]))
    torch.testing.assert_close(
        layout.full_slot_mapping(torch.tensor([0, 1, -1]), full.full_block_rows),
        torch.tensor([0, 1, -1]),
    )


def test_manual_slab_conv_stride_keeps_read_padding_outside_poisoned_scratch() -> None:
    layout = SharedKVCacheLayout(
        page_size_tokens=128,
        full_row_stride=8,
        conv_row_stride=1,
        ssm_row_stride=8,
        checkpoint_stride=2,
        num_slots=32,
        slot_bytes=262144,
        scratch_full_rows=1,
        scratch_conv_rows=1,
        scratch_ssm_rows=2,
        num_slabs=4,
    )
    # The optional Conv byte count is omitted, but one Conv row spans an
    # entire 256 KiB slab. The immutable Full pair must use the next slab.
    arena = torch.zeros(layout.num_slabs * layout.slot_bytes, dtype=torch.uint8)
    full_row_bytes = layout.slot_bytes // layout.full_row_stride
    rows = torch.zeros(layout.num_slots, dtype=torch.int32)
    rows[1] = 16
    full = LayerCache(
        key=arena.view(torch.bfloat16).view(32, 128, 1, 128),
        value=arena[full_row_bytes:].view(torch.bfloat16).view(31, 128, 1, 128),
        full_block_rows=rows,
    )
    linear = LayerCache(
        key=None,
        value=None,
        conv=arena.view(torch.bfloat16).view(4, 512, 256),
        ssm=arena.view(torch.float32).view(32, 1, 64, 128),
    )
    assert full.key is not None and full.value is not None
    assert linear.conv is not None and linear.ssm is not None
    full.value[16].fill_(3.25)
    protected_live = full.value[16].clone()
    full.key[0].fill_(float("nan"))
    full.value[0].fill_(float("nan"))
    linear.conv[0].fill_(float("nan"))
    linear.ssm[: layout.checkpoint_stride].fill_(float("nan"))
    # Ignoring conv_row_stride would select row 2 inside the poisoned Conv.
    assert torch.isnan(full.value[2]).all()
    physical = layout.full_block_table(torch.tensor([0, 1, -1], dtype=torch.int32), rows)
    torch.testing.assert_close(physical, torch.tensor([8, 16, -1], dtype=torch.int32))
    padding_row = int(physical[0].item())
    torch.testing.assert_close(full.key[padding_row], torch.zeros_like(full.key[padding_row]))
    torch.testing.assert_close(full.value[padding_row], torch.zeros_like(full.value[padding_row]))

    write_slots = layout.full_slot_mapping(torch.tensor([0, 127, -1], dtype=torch.int32), rows)
    torch.testing.assert_close(write_slots, torch.tensor([0, 127, -1], dtype=torch.int32))
    full.value.view(-1, 1, 128).index_fill_(0, write_slots[:2].to(torch.int64), 5)
    torch.testing.assert_close(full.value[padding_row], torch.zeros_like(full.value[padding_row]))
    torch.testing.assert_close(full.value[16], protected_live)


def test_slab_linear_mapping_uses_independent_role_and_layer_rows() -> None:
    caches = _make_slab_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    indices = torch.tensor([2, 1, 0, -1], dtype=torch.int32)
    torch.testing.assert_close(
        layout.linear_state_indices(indices, role="conv", slots=caches[1].linear_state_slots),
        torch.tensor([25, 24, 0, -1], dtype=torch.int32),
    )
    torch.testing.assert_close(
        layout.linear_state_indices(
            indices, role="ssm", slots=caches[3].linear_state_slots, checkpoint_offsets=torch.tensor([2, 1, 0, 2])
        ),
        torch.tensor([56, 52, 0, -1], dtype=torch.int32),
    )


def test_slab_map_reuse_keeps_logical_ids_and_storage_addresses() -> None:
    caches = _make_slab_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    rows = caches[0].full_block_rows
    assert rows is not None
    address = rows.data_ptr()
    rows[1] = 72
    torch.testing.assert_close(layout.full_block_table(torch.tensor([1]), rows), torch.tensor([72]))
    assert rows.data_ptr() == address


def test_batched_native_cache_rows_keep_live_storage_including_layer_subsets() -> None:
    native_rows = torch.arange(6 * 8, dtype=torch.int32).view(6, 8)
    table = CacheRowTable([native_rows[index] for index in (1, 3, 5)])
    batched = table.rows()
    assert batched.data_ptr() == native_rows[1].data_ptr()
    torch.testing.assert_close(batched, native_rows[1::2])
    native_rows[3, 2] = 1024
    assert batched[1, 2].item() == 1024
    assert table.rows().data_ptr() == batched.data_ptr()


def test_batched_independent_cache_rows_refresh_updated_mappings() -> None:
    rows = [torch.tensor([0, 16, 18], dtype=torch.int32), torch.tensor([0, 48, 50], dtype=torch.int32)]
    table = CacheRowTable(rows)
    torch.testing.assert_close(table.rows(), torch.tensor([[0, 16, 18], [0, 48, 50]], dtype=torch.int32))
    rows[0][1] = 72
    torch.testing.assert_close(table.rows(), torch.tensor([[0, 72, 18], [0, 48, 50]], dtype=torch.int32))


def test_batched_full_translation_preserves_read_padding_and_writable_scratch() -> None:
    caches = _make_slab_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    rows = CacheRowTable([caches[0].full_block_rows, caches[2].full_block_rows]).rows()
    blocks = torch.tensor([[1, 2, 0, -1]], dtype=torch.int32)
    tables = torch.empty(2, 1, 4, dtype=torch.int32)
    pointer = tables.data_ptr()
    assert layout.full_block_table(blocks, rows, out=tables) is tables
    assert tables.data_ptr() == pointer
    torch.testing.assert_close(tables, torch.tensor([[[30, 32, 6, -1]], [[34, 36, 6, -1]]], dtype=torch.int32))
    slots = torch.tensor([2, 3, 4, 0, -1], dtype=torch.int64)
    mappings = torch.empty(2, 5, dtype=torch.int64)
    assert layout.full_slot_mapping(slots, rows, out=mappings) is mappings
    torch.testing.assert_close(mappings, torch.tensor([[60, 61, 64, 0, -1], [68, 69, 72, 0, -1]]))


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("full_block_rows", torch.zeros(2, 32, dtype=torch.int32), "logical block row map"),
        ("shared_checkpoint_stride", None, "one checkpoint stride"),
    ],
)
def test_slab_layout_rejects_incomplete_full_metadata(field: str, value: object, error: str) -> None:
    caches = _make_slab_caches()
    caches[0] = replace(caches[0], **{field: value})
    with pytest.raises(ValueError, match=error):
        detect_shared_kv_cache_layout(caches)


def test_normalize_cache_tuple_retains_slab_rows_and_checkpoint_stride() -> None:
    cache = _make_slab_caches()[0]
    raw = (cache.key, cache.value, *(None for _ in range(11)), cache.full_block_rows, 3)
    normalized = normalize_layer_caches([raw])[0]
    assert normalized.full_block_rows is cache.full_block_rows
    assert normalized.shared_checkpoint_stride == 3


def _make_shared_caches(page_bytes: int = 1024, group_slots: tuple[torch.Tensor, ...] = ()) -> list[LayerCache]:
    """One Full layer, so one page per slot; each extra row adds a Linear group."""
    arena = torch.empty(8 * page_bytes, dtype=torch.uint8)
    full_rows, conv_rows, ssm_rows = {512: (29, 15, 30), 1024: (57, 29, 58), 2048: (113, 57, 114)}[page_bytes]
    key = _arena_view(arena, 0, (full_rows, 1, 1, 64), torch.bfloat16)
    value = _arena_view(arena, 128, (full_rows, 1, 1, 64), torch.bfloat16)
    conv = _arena_view(arena, 0, (conv_rows, 1, 1, 128), torch.bfloat16)
    ssm = _arena_view(arena, 256, (ssm_rows, 1, 1, 32), torch.float32)
    return [
        LayerCache(key=key, value=value),
        *(
            LayerCache(key=None, value=None, conv=conv, ssm=ssm, linear_state_slots=slots)
            for slots in (_identity_slots(8), *group_slots)
        ),
    ]


def _make_multigroup_caches() -> list[LayerCache]:
    # Four slots of two pages each: five Full layers fill two lanes and two
    # Linear layers share Linear group 0.  These explicit row counts include
    # the last slot in each role view.
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
            linear_state_slots=_identity_slots(4),
        )
        for offset in (0, 1024)
    ]
    return [full_caches[0], linear_caches[0], *full_caches[1:], linear_caches[1]]


def _make_ceiling_mapped_caches(group_slots: tuple[torch.Tensor, ...] = ()) -> list[LayerCache]:
    # Scale Qwen3.8 TP2 row bytes by 512 while retaining its 4608/35 Conv ratio.
    # One Full layer gives one page per slot, so every Linear layer is a group.
    arena = torch.empty(40 * 18432, dtype=torch.uint8)
    conv_rows = (39 * 18432 + 139) // 140 + 1
    ssm_rows = 39 * 6 + 5
    conv = _arena_view(arena, 0, (conv_rows, 7, 5), torch.float32)
    ssm = _arena_view(arena, 3072, (ssm_rows, 3, 16, 16), torch.float32)
    return [
        LayerCache(
            key=_arena_view(arena, 0, (2809, 8, 8), torch.float32),
            value=_arena_view(arena, 256, (2809, 8, 8), torch.float32),
        ),
        *(
            LayerCache(key=None, value=None, conv=conv, ssm=ssm, linear_state_slots=slots)
            for slots in (_identity_slots(40), *group_slots)
        ),
    ]


def test_detect_shared_layout_and_translate_full_indices() -> None:
    layout = detect_shared_kv_cache_layout(_make_shared_caches())

    assert layout is not None
    assert layout.page_size_tokens == 1
    assert layout.slot_bytes == 1024
    assert layout.num_slots == 8
    assert layout.pages_per_slot == 1
    assert layout.num_linear_groups == 1
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
    assert layout.slot_bytes == 2048
    assert layout.pages_per_slot == 2
    assert layout.num_slots == 4
    assert layout.num_linear_groups == 1
    torch.testing.assert_close(
        layout.full_block_table(torch.tensor([[3]], dtype=torch.int32)),
        torch.tensor([[48]], dtype=torch.int32),
    )
    torch.testing.assert_close(layout.linear_state_indices(torch.tensor([3]), role="conv"), torch.tensor([24]))
    torch.testing.assert_close(
        layout.linear_state_indices(torch.tensor([3]), role="ssm", checkpoint_offsets=torch.tensor([1])),
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
    # Once the Full slot is released, the same bytes can serve a Linear
    # state.  The second checkpoint also fits within the final physical page.
    last_linear.conv[24].fill_(21)
    last_linear.ssm[49].fill_(22)
    assert last_full.key[48].data_ptr() == last_linear.conv[24].data_ptr()
    torch.testing.assert_close(last_full.key[48], torch.full_like(last_full.key[48], 21))
    torch.testing.assert_close(last_linear.ssm[49], torch.full_like(last_linear.ssm[49], 22))


def test_ceiling_conv_mapping_preserves_sentinels_and_int64_intermediates() -> None:
    layout = detect_shared_kv_cache_layout(_make_ceiling_mapped_caches())

    assert layout is not None
    assert layout.slot_bytes == 18432
    assert layout.full_row_stride == 72
    assert layout.conv_row_stride is None
    assert layout.conv_row_bytes == 140
    assert layout.ssm_row_stride == 6
    assert layout.checkpoint_stride == 5
    logical = torch.tensor([0, 1, 2, 34, 35, 500000, -1, -2], dtype=torch.int32)
    torch.testing.assert_close(
        layout.linear_state_indices(logical, role="conv"),
        torch.tensor([0, 132, 264, 4477, 4608, 65828572, -1, -2], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="only for SSM"):
        layout.linear_state_indices(logical, role="conv", checkpoint_offsets=torch.zeros_like(logical))


def test_ceiling_conv_view_reuses_only_its_page_and_fits_final_extent() -> None:
    caches = _make_ceiling_mapped_caches()
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    full, linear = caches
    assert full.key is not None and linear.conv is not None and linear.ssm is not None
    full.key[72].fill_(17)
    full.key[144].fill_(53)
    linear.conv[132].fill_(29)
    torch.testing.assert_close(full.key[72].flatten()[:12], torch.full((12,), 17.0))
    torch.testing.assert_close(full.key[72].flatten()[12:47], torch.full((35,), 29.0))
    torch.testing.assert_close(full.key[72].flatten()[47:], torch.full((17,), 17.0))
    torch.testing.assert_close(full.key[144], torch.full_like(full.key[144], 53))
    logical = torch.tensor([39])
    assert layout.linear_state_indices(logical, role="conv").item() == 5135
    assert layout.linear_state_indices(logical, role="ssm", checkpoint_offsets=torch.tensor([4])).item() == 238
    linear.conv[5135].fill_(31)
    linear.ssm[238].fill_(37)
    torch.testing.assert_close(linear.conv[5135], torch.full_like(linear.conv[5135], 31))
    torch.testing.assert_close(linear.ssm[238], torch.full_like(linear.ssm[238], 37))


def test_ceiling_conv_groups_resolve_one_state_to_separate_slots() -> None:
    second_slots = _identity_slots(40)
    second_slots[38] = 39
    second_slots[39] = 38
    caches = _make_ceiling_mapped_caches(group_slots=(second_slots,))
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    assert layout.num_linear_groups == 2
    first, second = caches[1:]
    assert first.conv is not None and first.ssm is not None and second.conv is not None and second.ssm is not None
    # Both groups read the same lane; their slots keep the bytes apart.
    assert second.conv.data_ptr() == first.conv.data_ptr()
    assert 18432 % layout.conv_row_bytes != 0
    state = torch.tensor([38])
    rows = []
    for cache in (first, second):
        conv_row = layout.linear_state_indices(state, role="conv", slots=cache.linear_state_slots).item()
        ssm_row = layout.linear_state_indices(
            state, role="ssm", slots=cache.linear_state_slots, checkpoint_offsets=torch.tensor([4])
        ).item()
        rows.append((conv_row, ssm_row))
    assert rows == [(5003, 232), (5135, 238)]
    first.conv[5003].fill_(31)
    first.ssm[232].fill_(37)
    second.conv[5135].fill_(43)
    second.ssm[238].fill_(47)
    torch.testing.assert_close(first.conv[5003], torch.full_like(first.conv[5003], 31))
    torch.testing.assert_close(first.ssm[232], torch.full_like(first.ssm[232], 37))
    torch.testing.assert_close(second.conv[5135], torch.full_like(second.conv[5135], 43))
    torch.testing.assert_close(second.ssm[238], torch.full_like(second.ssm[238], 47))


def test_linear_groups_require_the_reading_layer_slot_row() -> None:
    second_slots = torch.tensor([0, 5, 6, 7, 1, 2, 3, 4], dtype=torch.int32)
    caches = _make_shared_caches(group_slots=(second_slots,))
    layout = detect_shared_kv_cache_layout(caches)
    assert layout is not None
    assert layout.num_linear_groups == 2
    state = torch.tensor([1, 0, -1], dtype=torch.int32)
    torch.testing.assert_close(
        layout.linear_state_indices(state, role="conv", slots=caches[2].linear_state_slots),
        torch.tensor([20, 0, -1], dtype=torch.int32),
    )
    with pytest.raises(ValueError, match="reading layer's state slot row"):
        layout.linear_state_indices(state, role="conv")


def test_shared_linear_layer_without_slot_row_is_rejected() -> None:
    caches = _make_shared_caches()
    caches[1] = replace(caches[1], linear_state_slots=None)

    with pytest.raises(ValueError, match="state slot row per slot"):
        detect_shared_kv_cache_layout(caches)


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
        num_slots=8,
        slot_bytes=1024,
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
