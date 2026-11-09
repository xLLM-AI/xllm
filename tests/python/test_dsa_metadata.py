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

"""Unit tests for the Python DSA metadata builder.

Validates the faithful port of ``DSAMetadataBuilder`` (core/layers/common/
dsa_metadata_builder.cpp) against the cache-spec and slot-expansion rules the
C++ implementation enforces. These tests are pure Python: they do not load the
compiled NPU operators, so they run anywhere.
"""

from __future__ import annotations

import pytest
import torch

from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RIGHT_ALIGNED,
    READ_TABLE_LAYOUT_RING,
    DsaMetadataBuilder,
    build_cache_specs,
    last_ring_writes,
)


def test_build_cache_specs_groups() -> None:
    """Group 0 is always SWA; TOKEN groups register in first-seen order."""
    compress_ratios = [0, 0, 4, 128, 4, 128, 4, 0]
    caches_info, group_infos = build_cache_specs(compress_ratios, window_size=128, n_layers=8)

    # Three groups: SWA(1,128), TOKEN(4,128), TOKEN(128,128).
    assert len(group_infos) == 3
    assert group_infos[0].cache_type == DSA_CACHE_SLIDING_WINDOW
    assert group_infos[0].ratio == 1
    assert group_infos[1].cache_type == DSA_CACHE_TOKEN
    assert group_infos[1].ratio == 4
    assert group_infos[2].cache_type == DSA_CACHE_TOKEN
    assert group_infos[2].ratio == 128


def test_build_cache_specs_per_layer_cache_counts() -> None:
    """C1 -> 1 cache, C4 -> 8 caches, C128 -> 4 caches."""
    compress_ratios = [0, 4, 128]
    caches_info, _ = build_cache_specs(compress_ratios, 128, 3)

    assert len(caches_info[0]) == 1  # cr=0 -> normalized to 1
    assert len(caches_info[1]) == 8  # cr=4
    assert len(caches_info[2]) == 4  # cr=128


def test_build_cache_specs_does_not_silently_accept_unknown_ratio() -> None:
    caches_info, group_infos = build_cache_specs([2], 128, 1)

    assert len(group_infos) == 1
    assert caches_info == [[]]


def test_build_cache_specs_real_dsv4_config() -> None:
    compress_ratios = [0, 0] + [4, 128] * 20 + [4, 0]
    assert len(compress_ratios) == 44
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, 43)

    assert len(group_infos) == 3
    c1 = sum(1 for layer in caches_info if len(layer) == 1)
    c4 = sum(1 for layer in caches_info if len(layer) == 8)
    c128 = sum(1 for layer in caches_info if len(layer) == 4)
    assert c1 == 2
    assert c4 == 21
    assert c128 == 20
    assert c1 + c4 + c128 == 43


def _make_builder(
    n_layers: int = 4,
    read_table_layout: str = READ_TABLE_LAYOUT_RIGHT_ALIGNED,
) -> tuple[DsaMetadataBuilder, list, list]:
    compress_ratios = [0, 4, 128, 4]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, n_layers)
    return (
        DsaMetadataBuilder(caches_info, group_infos, read_table_layout=read_table_layout),
        caches_info,
        group_infos,
    )


def test_build_seq_lengths_and_start_pos() -> None:
    """start_pos = kv_len - q_len per sequence."""
    builder, _, _ = _make_builder()
    # batch=2, decode (q_len=1 each).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([5, 7], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    assert dsa.seq_lens.tolist() == [6, 8]
    assert dsa.seq_lens_q.tolist() == [1, 1]
    assert dsa.start_pos.tolist() == [5, 7]
    # actual_seq_lengths_query is cumsum(q_lens) with a leading zero.
    assert dsa.actual_seq_lengths_query.tolist() == [0, 1, 2]
    assert dsa.kv_cu_seq_lens.tolist() == [0, 6, 14]
    assert dsa.max_seqlen_q.ndim == 0
    assert dsa.max_seqlen_kv.ndim == 0


def test_build_max_lengths_include_attention_metadata_capacity() -> None:
    """C++ takes max(params.meta.max_*, max(host sequence lengths))."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 2],
        positions=torch.tensor([5, 6, 7], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=16,
        max_seq_len=32,
    )

    assert dsa.max_query_len == 16
    assert dsa.max_seq_len == 32


def test_build_token_group_slot_committed_rows() -> None:
    """Decode writes the current SWA token and one shared TOKEN4 row."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7]
    assert dsa.slot_mappings[1][0].tolist() == [20 * 128 + 1]
    # TOKEN4 caches 0/1/7 and SWA caches 2/3 share their group storage.
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][1].data_ptr()
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][7].data_ptr()
    assert dsa.slot_mappings[1][2].data_ptr() == dsa.slot_mappings[1][3].data_ptr()


def test_build_token_group_slot_empty_between_boundaries() -> None:
    """Eager decode uses an actual empty tensor when no row is committed."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[129],
        q_seq_lens=[1],
        positions=torch.tensor([128], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].numel() == 0


def test_build_token_group_slot_commits_at_later_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[132],
        q_seq_lens=[1],
        positions=torch.tensor([131], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 128 + 32]


def test_build_swa_group_uses_scheduler_resolved_slots_for_expanded_decode() -> None:
    """Expanded MTP decode preserves the scheduler's ring-buffer slot order."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10], [11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20], [21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8, 8],
        q_seq_lens=[2, 2],
        positions=torch.tensor([6, 7, 6, 7], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=[301, 302, 401, 402],
    )

    assert dsa.slot_mappings[0][0].tolist() == [301, 302, 401, 402]


@pytest.mark.parametrize("enable_graph", [False, True])
def test_build_c4_pad_positions(enable_graph: bool) -> None:
    """Nonzero compressed positions precede the exact eager/graph padding tail."""
    builder, _, _ = _make_builder()
    # Positions 249..256 cross C4 boundaries at 251/255 and C128 at 255.
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[257],
        q_seq_lens=[8],
        positions=torch.arange(249, 257, dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
        enable_graph=enable_graph,
    )

    assert dsa.c4_pad_positions.tolist() == [248, 252] + [0] * (6 if enable_graph else 1)
    assert dsa.c128_pad_positions.tolist() == [128] + [0] * (7 if enable_graph else 0)


def _swa_read_table(
    raw_bt: torch.Tensor,
    ctx_lens: list[int],
    *,
    block_size: int = 128,
    graph_cols: int = 0,
    layout: str = READ_TABLE_LAYOUT_RING,
) -> torch.Tensor:
    builder, _, _ = _make_builder(read_table_layout=layout)
    read_bt, _ = builder._process_swa_group(
        raw_bt=raw_bt,
        block_size=block_size,
        ctx_lens=list(ctx_lens),
        q_lens=[1] * len(ctx_lens),
        batch_size=len(ctx_lens),
        graph_slot_capacity=len(ctx_lens) if graph_cols > 0 else 0,
        graph_block_table_capacity_cols=graph_cols,
        new_cache_slots=None,
    )
    return read_bt


@pytest.mark.parametrize("ctx_len", [129, 257, 513])
def test_swa_read_table_rings_every_addressed_column(ctx_len: int) -> None:
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [ctx_len])
    addressed = (ctx_len + 127) // 128
    assert read_bt.shape == (1, addressed)
    assert read_bt[0, :addressed].tolist() == [8] * addressed


def test_swa_read_table_resolves_each_sequence_ring_block() -> None:
    rings = [8, 9, 10, 11]
    raw_bt = torch.tensor([[ring] for ring in rings], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257] * len(rings))
    addressed = (257 + 127) // 128
    assert read_bt.shape == (len(rings), addressed)
    for seq, ring in enumerate(rings):
        assert read_bt[seq].tolist() == [ring] * addressed


def test_swa_read_table_modulos_by_semantic_cols() -> None:
    raw_bt = torch.tensor([[8, 9]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [385])
    addressed = (385 + 127) // 128
    assert read_bt.shape == (1, addressed)
    assert read_bt[0].tolist() == [8, 9, 8, 9]


def test_swa_read_table_rings_the_full_graph_storage_width() -> None:
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], graph_cols=8192)
    assert read_bt.shape == (1, 8192)
    assert bool((read_bt[0] == 8).all())


def test_swa_read_table_matches_the_write_path_ring_resolution() -> None:
    block_size = 128
    ctx_len = 513
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [ctx_len], block_size=block_size)
    read_cols = int(read_bt.size(1))
    semantic_cols = int(raw_bt.size(1))
    for pos in range(ctx_len):
        write_block = int(raw_bt[0, (pos // block_size) % semantic_cols].item())
        eager_block = int(read_bt[0, (pos // block_size) % read_cols].clamp_min(0).item())
        assert eager_block == write_block, f"pos={pos}"


def test_default_swa_read_table_stays_right_aligned() -> None:
    raw_bt = torch.tensor([[8]], dtype=torch.int32)
    read_bt = _swa_read_table(raw_bt, [257], layout=READ_TABLE_LAYOUT_RIGHT_ALIGNED)
    assert read_bt.shape == (1, 3)
    assert read_bt[0].tolist() == [-1, -1, 8]


def test_empty_batch_preserves_cpp_zero_length_buffers() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[],
        q_seq_lens=[],
        positions=torch.empty(0, dtype=torch.int64),
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.actual_seq_lengths_query.tolist() == [0]
    assert dsa.kv_cu_seq_lens.tolist() == [0]
    assert dsa.max_seqlen_q.shape == (1,)
    assert dsa.max_seqlen_kv.shape == (1,)
    assert dsa.max_query_len == 0
    assert dsa.max_seq_len == 0
    assert dsa.c4_pad_positions.numel() == 0
    assert dsa.c128_pad_positions.numel() == 0


def test_build_c128_slot_at_compression_boundary() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    token128_bt = torch.tensor([[30, 31]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt, token128_bt],
        kv_seq_lens=[128],
        q_seq_lens=[1],
        positions=torch.tensor([127], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    # Layer 2 uses TOKEN(128) for cache 0. The first compressed row is offset 0.
    assert dsa.slot_mappings[2][0].tolist() == [30 * 128]
    assert dsa.c128_pad_positions.tolist() == [0]


def test_multi_batch_slots_are_concatenated_by_sequence() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11], [12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21], [22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[4, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([3, 7], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 3, 12 * 128 + 7]
    assert dsa.slot_mappings[1][0].tolist() == [20 * 128, 22 * 128 + 1]


def test_packed_manager_block_table_is_unpacked() -> None:
    builder, _, _ = _make_builder()
    packed = torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[packed],
        kv_seq_lens=[128],
        q_seq_lens=[1],
        positions=torch.tensor([127], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.block_tables[0][0].tolist() == [[10, -1]]
    assert dsa.block_tables[1][0].tolist() == [[20, 21]]
    assert dsa.block_tables[2][0].tolist() == [[30, 31]]


def test_graph_slots_and_block_tables_use_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7, 0, 0, 0], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 128 + 1, -1, -1, -1]
    assert dsa.block_tables[1][0].shape == (1, 4)
    assert dsa.block_tables[1][0].tolist() == [[20, 21, -1, -1]]
    assert dsa.slot_mappings[0][0].tolist() == [10 * 128 + 7, -1, -1, -1]
    assert dsa.block_tables[0][0].shape == (1, 4)


def test_compressed_positions_preserve_position_dtype() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int32),
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert dsa.c4_pad_positions.dtype == torch.int32
    assert dsa.c128_pad_positions.dtype == torch.int32


def test_graph_rejects_block_table_larger_than_bucket_capacity() -> None:
    builder, _, _ = _make_builder()
    block_table = torch.tensor([[10, 11, 12]], dtype=torch.int32)

    with pytest.raises(ValueError, match="exceeds bucket capacity"):
        builder.build(
            multi_block_tables=[block_table],
            kv_seq_lens=[8],
            q_seq_lens=[1],
            positions=torch.tensor([7], dtype=torch.int64),
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=2,
        )


@pytest.mark.parametrize(
    "slots, keep_rows",
    [
        ([4, 1, 4, -1, 1, 2], [4, 5, 2]),
        ([-1, -1], []),
        ([3], [0]),
        ([-1], []),
        ([], []),
        ([3, 1, 2], [1, 2, 0]),
        ([16777214, 16777215, 16777214, -1], [2, 1]),
        ([16777216, 16777217, 16777216, -1], [2, 1]),
    ],
)
@pytest.mark.parametrize("slot_dtype", [torch.int32, torch.int64])
def test_last_ring_writes_keeps_latest_valid_rows(
    slots: list[int], keep_rows: list[int], slot_dtype: torch.dtype
) -> None:
    physical_slots = torch.tensor(slots, dtype=slot_dtype)
    values = torch.arange(len(slots) * 2).reshape(-1, 2)
    actual_slots, actual_values = last_ring_writes(physical_slots, values, slot_capacity=max(slots, default=-1) + 1)
    expected_rows = torch.tensor(keep_rows, dtype=torch.int64)
    assert actual_slots.dtype == slot_dtype
    assert torch.equal(actual_slots, physical_slots[expected_rows])
    assert torch.equal(actual_values, values[expected_rows])


@pytest.mark.parametrize("preserve_padding", [False, True])
@pytest.mark.parametrize("ctx_lens,live_slots", [([0, 1], [256]), ([1, 0], [0])])
@pytest.mark.parametrize("graph_rows", [0, 4])
@pytest.mark.parametrize("supplied_slots", [None, "compact", "padded"])
def test_swa_query_padding_keeps_input_row_alignment(
    preserve_padding: bool,
    ctx_lens: list[int],
    live_slots: list[int],
    graph_rows: int,
    supplied_slots: str | None,
) -> None:
    caches_info, group_infos = build_cache_specs([0], 128, 1)
    builder = DsaMetadataBuilder(caches_info, group_infos, preserve_query_padding=preserve_padding)
    expected = [(-1 if length == 0 else slot) for length, slot in zip(ctx_lens, [0, 256])]
    provided = None
    if supplied_slots == "compact":
        provided = live_slots
    elif supplied_slots == "padded":
        provided = expected
    _, slots = builder._process_swa_group(
        raw_bt=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        block_size=128,
        ctx_lens=ctx_lens,
        q_lens=[1, 1],
        batch_size=2,
        graph_slot_capacity=graph_rows,
        graph_block_table_capacity_cols=0,
        new_cache_slots=provided,
    )
    if not preserve_padding:
        expected = list(live_slots)
    expected += [-1] * max(0, graph_rows - len(expected))
    assert slots.tolist() == expected
