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
    DsaMetadataBuilder,
    build_cache_specs,
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
    """The shipped DeepSeek-V4-Flash config produces 2 C1 + 21 C4 + 20 C128.

    config.json has ``compress_ratios`` of length 44 (3 zeros, 21 fours, 20
    one-twenty-eights) but ``num_hidden_layers=43``; only layers 0..42 are
    built, so the trailing zero (index 43) is ignored and two C1 layers
    remain (indices 0 and 1).
    """
    compress_ratios = (
        [0, 0] + [4, 128] * 20 + [4, 0]  # layer 42 is C4; index 43 (zero) is ignored.
    )
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


def _make_builder(n_layers: int = 4) -> tuple[DsaMetadataBuilder, list, list]:
    compress_ratios = [0, 4, 128, 4]
    caches_info, group_infos = build_cache_specs(compress_ratios, 128, n_layers)
    return DsaMetadataBuilder(caches_info, group_infos), caches_info, group_infos


def test_build_seq_lengths_and_start_pos() -> None:
    """start_pos = kv_len - q_len per sequence."""
    builder, _, _ = _make_builder()
    # batch=2, decode (q_len=1 each).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[6, 8],
        q_seq_lens=[1, 1],
        positions=torch.tensor([5, 7], dtype=torch.int64),
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
        max_query_len=16,
        max_seq_len=32,
    )

    assert dsa.max_query_len == 16
    assert dsa.max_seq_len == 32


def test_build_token_group_slot_committed_rows() -> None:
    """A TOKEN cache commits one row per ratio boundary crossed this step."""
    builder, caches_info, group_infos = _make_builder()
    # group 0 = SWA, group 1 = TOKEN(4). Give each a [batch=1, cols=4] table.
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    # kv_len=8, q_len=1 (decode): prev_ctx_len=7, committed = 8//4 - 7//4 = 2 - 1 = 1.
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): cmp cache is caches_info[1][0] -> group 1 (TOKEN4).
    cmp_slot = dsa.slot_mappings[1][0]
    # One committed row: compressed_idx = prev_committed = 7//4 = 1.
    # block_idx = 1 // 128 = 0, block_id = token4_bt[0,0] = 20.
    # slot = 20 * 128 + 1 = 2561.
    assert cmp_slot.numel() >= 1
    assert cmp_slot[0].item() == 20 * 128 + 1


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
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    assert dsa.slot_mappings[1][0].tolist() == [20 * 128 + 32]


def test_build_swa_group_slot_query_tokens_only() -> None:
    """A SWA cache writes only the current forward's query token."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    # kv_len=8, q_len=1, q_start=7, pos=7, block_idx = 7//128 % 4 = 0,
    # block_id = swa_bt[0,0] = 10, offset = 7 % 128 = 7 -> slot = 10*128+7 = 1287.
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 0 (cr=1): the single SWA cache -> group 0.
    swa_slot = dsa.slot_mappings[0][0]
    assert swa_slot[0].item() == 10 * 128 + 7


def test_build_block_tables_shared_within_group() -> None:
    """Caches in the same group share the same underlying tensor."""
    builder, _, _ = _make_builder()
    swa_bt = torch.tensor([[10, 11, 12, 13]], dtype=torch.int32)
    token4_bt = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)
    dsa = builder.build(
        multi_block_tables=[swa_bt, token4_bt],
        kv_seq_lens=[8],
        q_seq_lens=[1],
        positions=torch.tensor([7], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )
    # Layer 1 (cr=4): caches 0,1,7 are TOKEN4 (group 1) -> same slot tensor.
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][1].data_ptr()
    assert dsa.slot_mappings[1][0].data_ptr() == dsa.slot_mappings[1][7].data_ptr()
    # Caches 2-6 are SWA (group 0) -> same slot tensor.
    assert dsa.slot_mappings[1][2].data_ptr() == dsa.slot_mappings[1][3].data_ptr()


def test_build_c4_pad_positions() -> None:
    """c4_pad_positions records next_pos-4 when (pos+1) % 4 == 0."""
    builder, _, _ = _make_builder()
    # q_len=4, q_start=3 -> positions 3,4,5,6. (pos+1)%4==0 at pos=3 (next=4).
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[7],
        q_seq_lens=[4],
        positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
    )
    # pos=3 -> next_pos=4 -> 4%4==0 -> record 4-4=0.
    assert 0 in dsa.c4_pad_positions.tolist()


def test_graph_compressed_positions_use_zero_padding() -> None:
    """ACL graph position buffers match C++ vector::resize zero fill."""
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[7],
        q_seq_lens=[4],
        positions=torch.tensor([3, 4, 5, 6], dtype=torch.int64),
        dsa_cos_sin=None,
        is_prefill=True,
        is_chunked_prefill=False,
        enable_graph=True,
    )

    assert dsa.c4_pad_positions.tolist() == [0, 0, 0, 0]
    assert dsa.c128_pad_positions.tolist() == [0, 0, 0, 0]


def test_empty_batch_preserves_cpp_zero_length_buffers() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[],
        q_seq_lens=[],
        positions=torch.empty(0, dtype=torch.int64),
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
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
        dsa_cos_sin=None,
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


def test_rope_cache_is_split_into_contiguous_cos_and_sin_tables() -> None:
    builder, _, _ = _make_builder()
    cos_sin = torch.arange(24, dtype=torch.float32).view(3, 8)
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[3],
        q_seq_lens=[3],
        positions=torch.arange(3, dtype=torch.int64),
        dsa_cos_sin=cos_sin,
        is_prefill=True,
        is_chunked_prefill=False,
    )

    assert torch.equal(dsa.cos_table, cos_sin[:, :4])
    assert torch.equal(dsa.sin_table, cos_sin[:, 4:])
    assert dsa.cos_table.is_contiguous()
    assert dsa.sin_table.is_contiguous()


def test_compressed_positions_preserve_position_dtype() -> None:
    builder, _, _ = _make_builder()
    dsa = builder.build(
        multi_block_tables=[],
        kv_seq_lens=[4],
        q_seq_lens=[4],
        positions=torch.arange(4, dtype=torch.int32),
        dsa_cos_sin=None,
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
            dsa_cos_sin=None,
            is_prefill=False,
            is_chunked_prefill=False,
            enable_graph=True,
            graph_block_table_capacity_cols=2,
        )
