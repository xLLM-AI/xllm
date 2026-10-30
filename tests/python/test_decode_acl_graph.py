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

"""Tests for the NPU ACL decode-graph runner."""

import os
from contextlib import ExitStack, nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import DsaAttentionBackend
from xllm.python.attention.dsa_metadata import DsaMetadataBuilder, build_cache_specs
from xllm.python.attention.paged_kv_metadata import build_row_aligned_paged_kv_metadata
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.block_draft_acl_graph import (
    BlockDraftAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner(
    *,
    max_batch: int = 8,
    max_model_len: int = 8,
    dp_size: int = 1,
    num_decoding_tokens: int = 1,
    spec_verify_width: int = 1,
    enable_graph_mode_decode_no_padding: bool = False,
    enable_mega_moe_token_mask: bool = False,
) -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False, create_graph_block_tables=lambda *_: ())
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=max_batch,
        max_model_len=max_model_len,
        dp_size=dp_size,
        num_decoding_tokens=num_decoding_tokens,
        spec_verify_width=spec_verify_width,
        enable_graph_mode_decode_no_padding=enable_graph_mode_decode_no_padding,
        enable_mega_moe_token_mask=enable_mega_moe_token_mask,
    )


def _metadata(linear_state_indices: torch.Tensor, rows: int | None = None) -> SimpleNamespace:
    rows = rows or linear_state_indices.numel()
    lengths = torch.arange(1, rows + 1, dtype=torch.int32)
    pages = lengths * 10
    return SimpleNamespace(
        slot_mapping=torch.arange(rows, dtype=torch.int32),
        paged_kv_indptr=torch.arange(rows + 1, dtype=torch.int32),
        paged_kv_indices=pages,
        paged_kv_last_page_len=lengths,
        block_table=torch.stack((pages, torch.zeros_like(pages)), dim=1),
        kv_seq_lens=lengths,
        kv_seq_lens_host_values=lengths.tolist(),
        kv_cu_seq_lens=torch.cat((torch.zeros(1, dtype=torch.int32), lengths.cumsum(0, dtype=torch.int32))),
        q_cu_seq_lens=None,
        linear_state_indices=linear_state_indices,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
    )


class _FakeBlockDraftAttentionBackend:
    page_size = 4
    logical_page_size = 4
    is_mla = False

    @staticmethod
    def prepare_metadata(metadata: SimpleNamespace, *, device_kv_lengths: bool) -> SimpleNamespace:
        assert device_kv_lengths
        return SimpleNamespace(
            actual_seq_q=list(metadata.q_cu_seq_lens_host_values),
            actual_seq_kv=list(metadata.kv_seq_lens_host_values),
            query_ends=list(metadata.q_cu_seq_lens_host_values),
        )


def _block_draft_runner(max_batch: int = 1, **kwargs: object) -> BlockDraftAclGraphRunner:
    attention_backend = _FakeBlockDraftAttentionBackend()
    return BlockDraftAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=max_batch,
        max_model_len=8,
        **kwargs,
    )


def _block_draft_metadata(block_table: torch.Tensor) -> SimpleNamespace:
    sequence_count, block_cols = block_table.shape
    query_width = 2
    token_count = sequence_count * query_width
    query_ends = [query_width * (index + 1) for index in range(sequence_count)]
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
        q_seq_lens=torch.full((sequence_count,), query_width, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, *query_ends], dtype=torch.int32),
        q_cu_seq_lens_host_values=query_ends,
        q_seq_lens_host=None,
        block_table=block_table,
        slot_mapping=torch.arange(token_count, dtype=torch.int32),
        kv_seq_lens=torch.full((sequence_count,), block_cols * 4, dtype=torch.int32),
        kv_cu_seq_lens=None,
        kv_seq_lens_host_values=[block_cols * 4] * sequence_count,
        paged_kv_indptr=torch.arange(sequence_count + 1, dtype=torch.int32) * block_cols,
        paged_kv_indices=torch.arange(sequence_count * block_cols, dtype=torch.int32),
        paged_kv_last_page_len=torch.full((sequence_count,), 4, dtype=torch.int32),
    )


@pytest.mark.parametrize("dp_rank,sequence_count", [(0, 1), (1, 3)])
def test_dspark_dp_graph_uses_group_wide_complete_blocks(dp_rank: int, sequence_count: int) -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2, dp_rank=dp_rank)
    metadata = _block_draft_metadata(torch.zeros(sequence_count, 3, dtype=torch.int32))
    metadata.dp_execution_token_counts = (2, 6)
    metadata.dp_is_decode = (1, 1)
    input_ids = torch.arange(sequence_count * 2, dtype=torch.int32)

    assert runner.can_execute(input_ids, metadata)
    assert runner._padded_sequence_count(metadata, sequence_count, 2) == 4
    entry = runner._allocate_entry(input_ids, input_ids, metadata, sequence_count=4, page_table_capacity=4)
    runner._fill_entry(entry, input_ids, input_ids, metadata)
    static = entry.static_metadata
    assert entry.static_input_ids.numel() == 8
    assert static.q_cu_seq_lens.tolist() == [0, 2, 4, 6, 8]
    assert static.q_seq_lens.tolist() == [2] * 4
    assert static.dp_execution_token_counts == (8, 8)
    assert static.dp_is_decode == (1, 1)
    assert static.slot_mapping[sequence_count * 2 :].tolist() == [-1] * (8 - sequence_count * 2)
    assert static.kv_seq_lens.tolist() == [12] * sequence_count + [1] * (4 - sequence_count)
    assert static.prepared_attention_state.actual_seq_q == [2, 4, 6, 8]
    assert static.prepared_attention_state.actual_seq_kv == [16] * 4


def test_dspark_dp_graph_admits_an_idle_rank_full_block() -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2, dp_rank=0)
    metadata = _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32))
    metadata.dp_execution_token_counts = (2, 6)
    metadata.dp_global_sequence_nums = (0, 3)
    metadata.dp_is_decode = (1, 1)
    metadata.slot_mapping.fill_(-1)
    metadata.kv_seq_lens.fill_(1)
    metadata.kv_seq_lens_host_values = [1]
    metadata.paged_kv_last_page_len.fill_(1)
    input_ids = torch.zeros(2, dtype=torch.int32)

    assert runner.can_execute(input_ids, metadata)
    entry = runner._allocate_entry(input_ids, input_ids, metadata, sequence_count=4, page_table_capacity=4)
    runner._fill_entry(entry, input_ids, input_ids, metadata)
    assert entry.static_metadata.slot_mapping.tolist() == [-1] * 8
    assert entry.static_metadata.kv_seq_lens.tolist() == [1] * 4


@pytest.mark.parametrize(
    "counts,phases,error",
    [
        ((2,), (1, 1), "every DP rank"),
        ((2, 3), (1, 1), "complete query blocks"),
        ((4, 6), (1, 1), "local execution count"),
        ((2, 0), (1, 1), "complete query blocks"),
    ],
)
def test_dspark_dp_graph_rejects_invalid_counts(counts: tuple[int, ...], phases: tuple[int, ...], error: str) -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2)
    metadata = _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32))
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    with pytest.raises(ValueError, match=error):
        runner.can_execute(torch.zeros(2, dtype=torch.int32), metadata)


@pytest.mark.parametrize("counts,phases", [((2, 10), (1, 1)), ((2, 6), (1, 0))])
def test_dspark_dp_graph_falls_back_group_wide(counts: tuple[int, ...], phases: tuple[int, ...]) -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2)
    metadata = _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32))
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    assert not runner.can_execute(torch.zeros(2, dtype=torch.int32), metadata)


def test_dspark_dp_replay_clears_rows_from_a_larger_local_batch() -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2)
    metadata = _block_draft_metadata(torch.full((3, 2), 10, dtype=torch.int32))
    input_ids = torch.arange(6, dtype=torch.int32) + 10
    metadata.kv_cu_seq_lens = torch.tensor([0, 8, 16, 24], dtype=torch.int32)
    entry = runner._allocate_entry(input_ids, input_ids, metadata, sequence_count=4, page_table_capacity=4)
    runner._fill_entry(entry, input_ids, input_ids, metadata)
    smaller = _block_draft_metadata(torch.tensor([[20]], dtype=torch.int32))
    runner._fill_entry(entry, input_ids[:2], input_ids[:2], smaller)

    assert entry.static_input_ids.tolist() == [10, 11, 0, 0, 0, 0, 0, 0]
    assert entry.static_positions.tolist() == [10, 11, 0, 0, 0, 0, 0, 0]
    static = entry.static_metadata
    assert static.slot_mapping.tolist() == [0, 1, -1, -1, -1, -1, -1, -1]
    assert static.kv_seq_lens.tolist() == [4, 1, 1, 1]
    assert static.kv_cu_seq_lens.tolist() == [0, 4, 5, 6, 7]
    assert static.paged_kv_indptr.tolist() == [0, 1, 1, 1, 1]
    assert static.paged_kv_last_page_len.tolist() == [4, 1, 1, 1]
    assert static.block_table[1:].count_nonzero().item() == 0


def test_dspark_dp_graph_key_tracks_peer_page_capacity() -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2)
    input_ids = torch.arange(2, dtype=torch.int32)
    local_key = runner._graph_key(input_ids, _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32)), 4, 2)
    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.dist.is_initialized", return_value=True),
        patch(
            "xllm.python.distributed.all_gather",
            side_effect=[torch.tensor([1, 1]), torch.tensor([1, 2]), torch.tensor([1, 1])],
        ),
    ):
        first = runner._synchronize_dp_graph_key(local_key, input_ids)
        changed_peer = runner._synchronize_dp_graph_key(local_key, input_ids)
        restored = runner._synchronize_dp_graph_key(local_key, input_ids)
    assert first[:-1] == local_key
    assert first != changed_peer
    assert first == restored


def test_dspark_dp_warmup_recaptures_when_any_rank_is_missing() -> None:
    runner = _block_draft_runner(max_batch=8, dp_size=2)
    metadata = _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32))
    metadata.dp_execution_token_counts = (2, 6)
    metadata.dp_is_decode = (1, 1)
    input_ids = torch.arange(2, dtype=torch.int32)
    local_key = runner._graph_key(input_ids, metadata, 4, 2)
    graph_key = (*local_key, (1, 1))
    runner._graphs[graph_key] = object()
    expected_eplb = Mock()

    def presence(count: torch.Tensor, *, group_name: str) -> None:
        assert group_name == "dp"
        count.fill_(1)

    def prepare(
        *_args: object,
        graph_key: tuple[object, ...],
        eplb: object | None = None,
        input_batch: object = None,
    ) -> None:
        assert eplb is expected_eplb
        assert input_batch is None
        assert graph_key not in runner._graphs
        runner._graphs[graph_key] = object()

    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.dist.is_initialized", return_value=True),
        patch("xllm.python.distributed.all_gather", return_value=torch.tensor([1, 1])),
        patch("xllm.python.distributed.all_reduce_", side_effect=presence),
        patch.object(runner, "_prepare_graph_entry", side_effect=prepare) as capture,
    ):
        assert runner.warmup(input_ids, input_ids, metadata, eplb=expected_eplb) == graph_key
    capture.assert_called_once()


@pytest.mark.parametrize("dp_rank", [0, 1])
def test_block_draft_dp_prewarm_keeps_collectives_aligned_for_unequal_capacities(dp_rank: int) -> None:
    runner = _block_draft_runner(dp_size=2, dp_rank=dp_rank, block_draft_graph_warmup_max_kv_len=4096)
    runner.max_model_len = 16384
    runner.attention_backend.logical_page_size = 128
    metadata = _block_draft_metadata(torch.zeros(1, 128, dtype=torch.int32))
    length = 8193 if dp_rank == 0 else 2
    metadata.kv_seq_lens.fill_(length)
    metadata.kv_seq_lens_host_values = [length]
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 1)
    tokens = torch.arange(2, dtype=torch.int32)
    captured = []

    def presence(count: torch.Tensor, *, group_name: str) -> None:
        assert group_name == "dp"
        count.mul_(2)

    def prepare(*_args: object, graph_key: tuple[object, ...], **_: object) -> None:
        captured.append(graph_key)
        runner._graphs[graph_key] = object()

    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.dist.is_initialized", return_value=True),
        patch(
            "xllm.python.distributed.all_gather",
            side_effect=[torch.tensor([1, 1]), torch.tensor([2, 1])],
        ) as gather,
        patch("xllm.python.distributed.all_reduce_", side_effect=presence) as reduce,
        patch.object(runner, "_prepare_graph_entry", side_effect=prepare),
    ):
        key = runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)

    # Both ranks visit the bounded tier, then their current local capacity.
    assert gather.call_count == reduce.call_count == 2
    assert [key[2] for key in captured] == ([32, 128] if dp_rank == 0 else [32, 32])
    assert [key[-1] for key in captured] == [(1, 1), (2, 1)]
    assert key == captured[-1]


def test_block_draft_dp_prewarmed_capacity_keys_are_reused_at_runtime() -> None:
    runner = _block_draft_runner(dp_size=2)
    runner.max_model_len = 16384
    runner.attention_backend.logical_page_size = 128
    metadata = _block_draft_metadata(torch.zeros(1, 128, dtype=torch.int32))
    metadata.kv_seq_lens.fill_(2)
    metadata.kv_seq_lens_host_values = [2]
    metadata.dp_execution_token_counts = (2, 2)
    metadata.dp_is_decode = (1, 1)
    tokens = torch.arange(2, dtype=torch.int32)
    captured = []

    def gather(variant: torch.Tensor, *, dim: int, world_size: int, group_name: str) -> torch.Tensor:
        assert dim == 0 and world_size == 2 and group_name == "dp"
        return variant.repeat(world_size)

    def presence(count: torch.Tensor, *, group_name: str) -> None:
        assert group_name == "dp"
        count.mul_(2)

    def prepare(*_args: object, graph_key: tuple[object, ...], **_: object) -> None:
        captured.append(graph_key)
        runner._graphs[graph_key] = object()

    with (
        patch("xllm.python.model_executor.runners.decode_acl_graph.dist.is_initialized", return_value=True),
        patch("xllm.python.distributed.all_gather", side_effect=gather),
        patch("xllm.python.distributed.all_reduce_", side_effect=presence),
        patch.object(runner, "_prepare_graph_entry", side_effect=prepare),
    ):
        current_key = runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)
        assert current_key == captured[-1]
        assert [(key[2], key[-1]) for key in captured] == [(128, (1, 1)), (64, (2, 2)), (32, (3, 3))]
        for length in (2048, 4097, 8193, 2048):
            metadata.kv_seq_lens.fill_(length)
            metadata.kv_seq_lens_host_values = [length]
            assert runner.warmup(tokens, tokens, metadata) in captured
        runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)

    assert len(captured) == 3


def test_dspark_graph_key_buckets_live_page_table_width() -> None:
    runner = _block_draft_runner()
    input_ids = torch.zeros(2, dtype=torch.int32)
    key = runner._graph_key(
        input_ids,
        _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32)),
        sequence_count=1,
        query_width=2,
    )
    wider_key = runner._graph_key(
        input_ids,
        _block_draft_metadata(torch.zeros(1, 4, dtype=torch.int32)),
        sequence_count=1,
        query_width=2,
    )
    assert key == wider_key
    assert key != runner._graph_key(
        input_ids,
        _block_draft_metadata(torch.zeros(1, 5, dtype=torch.int32)),
        sequence_count=1,
        query_width=2,
    )


@pytest.mark.parametrize(
    ("block_cols", "capacity"),
    [(1, 4), (2, 4), (3, 4), (4, 4), (5, 8), (15, 16), (16, 16), (17, 32)],
)
def test_dspark_page_table_capacity_grows_geometrically(block_cols: int, capacity: int) -> None:
    runner = _block_draft_runner()
    metadata = _block_draft_metadata(torch.zeros(1, block_cols, dtype=torch.int32))
    assert runner._page_table_capacity(metadata) == capacity


def test_dspark_graph_capacity_does_not_use_model_maximum() -> None:
    runner = _block_draft_runner()
    runner.max_model_len = 1_048_576
    runner.attention_backend.logical_page_size = 128
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _block_draft_metadata(torch.zeros(1, 128, dtype=torch.int32))
    metadata.kv_seq_lens.fill_(15 * 128)
    metadata.kv_seq_lens_host_values = [15 * 128]

    entry = runner._allocate_entry(input_ids, positions, metadata, sequence_count=1, page_table_capacity=32)

    assert entry.static_metadata.block_table.shape == (1, 32)
    assert entry.static_metadata.paged_kv_indices.numel() == 32
    assert entry.static_metadata.kv_seq_lens_host_values == [4096]


@pytest.mark.parametrize("sparse_mode", [0, 4], ids=["dspark", "dflash2"])
def test_windowed_block_draft_reuses_one_source_capacity_across_contexts(sparse_mode: int) -> None:
    from xllm.python.layers.attention import Attention

    model = nn.ModuleList([Attention(4, 1, 64, 0.125, 2048, 0, fia_sparse_mode=sparse_mode, fia_pre_tokens=2047)])
    backend = _FakeBlockDraftAttentionBackend()
    backend.logical_page_size = 128
    runner = BlockDraftAclGraphRunner(model, backend, torch.device("cpu"), max_batch=1, max_model_len=65537)
    tokens = torch.arange(2, dtype=torch.int32)

    def prepare_entry(
        *args: object, graph_key: tuple[object, ...], eplb: object = None, input_batch: object = None
    ) -> None:
        runner._graphs[graph_key] = object()

    with patch.object(runner, "_prepare_graph_entry", side_effect=prepare_entry) as prepare:
        for length in (16, 4097, 16385, 32769, 65537, 16):
            columns = (length + 127) // 128
            metadata = _block_draft_metadata(torch.zeros(1, columns, dtype=torch.int32))
            metadata.kv_seq_lens_host_values[:] = [length]
            metadata.kv_seq_lens.fill_(length)
            key = runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)
            assert key[2] == 513
    assert prepare.call_count == 1
    entry = runner._allocate_entry(tokens, tokens, metadata, sequence_count=1, page_table_capacity=key[2])
    assert entry.static_metadata.block_table.shape == (1, 513)


@pytest.mark.parametrize(("sequence_count", "padded_count"), [(1, 1), (2, 4)])
def test_block_draft_graph_compacts_reserved_pages_with_live_lengths(sequence_count: int, padded_count: int) -> None:
    runner = _block_draft_runner(max_batch=4)
    runner.max_model_len = 1_048_576
    runner.attention_backend.logical_page_size = 128
    block_table = torch.arange(1, sequence_count * 128 + 1, dtype=torch.int32).view(sequence_count, 128)
    metadata = _block_draft_metadata(block_table)
    lengths = [4095, 129][:sequence_count]
    metadata.kv_seq_lens = torch.tensor(lengths, dtype=torch.int32)
    metadata.kv_seq_lens_host_values = lengths
    input_ids = torch.arange(2 * sequence_count, dtype=torch.int32)
    entry = runner._allocate_entry(input_ids, input_ids, metadata, sequence_count=padded_count, page_table_capacity=32)

    runner._fill_entry(entry, input_ids, input_ids, metadata)

    static = entry.static_metadata
    assert static.block_table.shape == (padded_count, 32)
    counts = [(length + 127) // 128 for length in lengths]
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    assert static.paged_kv_indptr.tolist() == offsets + [offsets[-1]] * (padded_count - sequence_count)
    expected = torch.cat([block_table[row, :count] for row, count in enumerate(counts)])
    torch.testing.assert_close(static.paged_kv_indices[: expected.numel()], expected)
    assert not static.paged_kv_indices[expected.numel() :].count_nonzero().item()
    assert static.paged_kv_last_page_len.tolist() == [127, 1][:sequence_count] + [1] * (padded_count - sequence_count)
    prepared = static.prepared_attention_state
    assert prepared.actual_seq_kv == [4096] * padded_count

    # Reuse the same storage for a shorter request and clear all old pages.
    smaller = _block_draft_metadata(torch.full((1, 128), 500, dtype=torch.int32))
    smaller.kv_seq_lens.fill_(129)
    smaller.kv_seq_lens_host_values = [129]
    runner._fill_entry(entry, input_ids[:2], input_ids[:2], smaller)
    assert static.paged_kv_indptr.tolist() == [0] + [2] * padded_count
    assert static.paged_kv_last_page_len.tolist() == [1] * padded_count
    assert static.paged_kv_indices[:2].tolist() == [500, 500]
    assert not static.paged_kv_indices[2:].count_nonzero().item()
    assert prepared.actual_seq_kv == [4096] * padded_count


def test_dspark_warmup_captures_once_per_page_capacity_bucket() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()

    def prepare_entry(*_args: object, graph_key: tuple[object, ...], **_: object) -> None:
        runner._graphs[graph_key] = object()

    with patch.object(runner, "_prepare_graph_entry", side_effect=prepare_entry) as prepare:
        keys = [
            runner.warmup(
                input_ids, positions, _block_draft_metadata(torch.zeros(1, width, dtype=torch.int32)), eplb=None
            )
            for width in (3, 4, 5, 3)
        ]
    assert prepare.call_count == 2
    assert keys[0] == keys[1] == keys[3]
    assert keys[0] != keys[2]


@pytest.mark.parametrize(
    ("max_model_len", "expected"),
    [(128, [2]), (2048, [32]), (8320, [128, 64, 32]), (16384, [128, 64, 32])],
)
def test_block_draft_startup_prewarms_large_buckets_with_short_metadata(
    max_model_len: int, expected: list[int]
) -> None:
    runner = _block_draft_runner(block_draft_graph_warmup_max_kv_len=0)
    runner.max_model_len = max_model_len
    runner.attention_backend.logical_page_size = 128
    metadata = _block_draft_metadata(torch.zeros(1, 1, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = [2]
    tokens = torch.arange(2, dtype=torch.int32)
    captured = []

    def prepare_entry(*_args: object, graph_key: tuple[object, ...], **_: object) -> None:
        captured.append(graph_key[2])
        runner._graphs[graph_key] = object()

    with patch.object(runner, "_prepare_graph_entry", side_effect=prepare_entry):
        key = runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)
        runner.warmup(tokens, tokens, metadata, pre_capture_kv_buckets=True)
    assert captured == expected
    assert key[2] == expected[-1]
    metadata.block_table = torch.zeros(1, (max_model_len + 127) // 128, dtype=torch.int32)
    metadata.kv_seq_lens_host_values[:] = [max_model_len]
    metadata.kv_seq_lens.fill_(max_model_len)
    runtime_key = runner._graph_key(tokens, metadata, 1, 2)
    assert runtime_key in runner._graphs


def test_dspark_request_reuses_startup_page_bucket() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    startup_metadata = _block_draft_metadata(torch.zeros(1, 4, dtype=torch.int32))
    request_metadata = _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32))

    assert runner._graph_key(input_ids, startup_metadata, 1, 2) == runner._graph_key(
        input_ids,
        request_metadata,
        1,
        2,
    )


def test_dspark_graph_is_cached_only_after_successful_capture() -> None:
    runner = _block_draft_runner()
    runner._stream = object()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32))
    graph_key = runner._graph_key(input_ids, metadata, 1, 2)

    def capture_entry(entry: AclGraphEntry, stream: object) -> None:
        assert graph_key not in runner._graphs
        assert stream is runner._stream
        entry.graph = object()

    with (
        patch.object(runner, "_prepare_attention"),
        patch.object(runner, "_capture", side_effect=capture_entry) as capture,
    ):
        entry = runner._prepare_graph_entry(input_ids, positions, metadata, graph_key=graph_key)
        assert runner._graphs[graph_key] is entry
        assert entry.graph is not None
        assert runner._prepare_graph_entry(input_ids, positions, metadata, graph_key=graph_key) is entry
        capture.assert_called_once()


@pytest.mark.parametrize("failure_stage", ["_fill_entry", "_prepare_attention", "_capture"])
def test_dspark_failed_graph_preparation_is_not_cached(failure_stage: str) -> None:
    runner = _block_draft_runner()
    runner._stream = object()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32))
    graph_key = runner._graph_key(input_ids, metadata, 1, 2)

    with (
        patch.object(runner, "_fill_entry", wraps=runner._fill_entry) as fill,
        patch.object(runner, "_prepare_attention") as prepare,
        patch.object(runner, "_capture") as capture,
    ):
        steps = {"_fill_entry": fill, "_prepare_attention": prepare, "_capture": capture}
        steps[failure_stage].side_effect = RuntimeError("graph preparation failed")
        with pytest.raises(RuntimeError, match="graph preparation failed"):
            runner._prepare_graph_entry(input_ids, positions, metadata, graph_key=graph_key)

    assert graph_key not in runner._graphs


def test_dspark_graph_replay_preserves_host_capacity_and_updates_device_lengths() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    first_metadata = _block_draft_metadata(torch.tensor([[10, 11, 12]], dtype=torch.int32))
    entry = runner._allocate_entry(input_ids, positions, first_metadata, sequence_count=1, page_table_capacity=4)
    prepared = entry.static_metadata.prepared_attention_state
    host_bounds = prepared.actual_seq_kv
    device_lengths = entry.static_metadata.kv_seq_lens
    assert host_bounds == [16]

    for block_cols in (1, 4, 2):
        metadata = _block_draft_metadata(torch.full((1, block_cols), 20, dtype=torch.int32))
        runner._fill_entry(entry, input_ids, positions, metadata)
        assert prepared.actual_seq_q == [2]
        assert prepared.actual_seq_kv is host_bounds
        assert host_bounds == [16]
        assert entry.static_metadata.kv_seq_lens is device_lengths
        assert device_lengths.tolist() == [block_cols * 4]
        assert entry.static_metadata.kv_seq_lens_host_values == [block_cols * 4]
        assert prepared.query_ends == [2]


def test_dspark_graph_rejects_missing_host_kv_lengths() -> None:
    runner = _block_draft_runner()
    metadata = _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None

    assert not runner.can_execute(torch.zeros(2, dtype=torch.int32), metadata)


def test_dspark_graph_rejects_inconsistent_paged_metadata() -> None:
    runner = _block_draft_runner()
    metadata = _block_draft_metadata(torch.zeros(1, 3, dtype=torch.int32))
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)

    assert not runner.can_execute(torch.zeros(2, dtype=torch.int32), metadata)


def test_dspark_graph_replay_clears_stale_page_table_tail() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    first_metadata = _block_draft_metadata(
        torch.tensor([[10, 11, 12]], dtype=torch.int32),
    )
    entry = runner._allocate_entry(input_ids, positions, first_metadata, sequence_count=1, page_table_capacity=4)

    runner._fill_entry(entry, input_ids, positions, first_metadata)
    assert entry.static_metadata.block_table.shape == (1, 4)
    assert entry.static_metadata.block_table[0].tolist() == [10, 11, 12, 0]

    second_metadata = _block_draft_metadata(
        torch.tensor([[20]], dtype=torch.int32),
    )
    runner._fill_entry(entry, input_ids, positions, second_metadata)
    assert entry.static_metadata.block_table[0].tolist() == [20, 0, 0, 0]
    assert entry.static_metadata.paged_kv_indices.tolist() == [20, 0, 0, 0]

    with pytest.raises(RuntimeError, match="exceeds the captured page-table capacity"):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            _block_draft_metadata(torch.zeros(1, 5, dtype=torch.int32)),
        )


@pytest.mark.parametrize(
    ("invalid", "error_type", "message"),
    [
        ("paged_metadata", ValueError, "paged KV metadata is inconsistent"),
        ("table_capacity", RuntimeError, "exceeds the captured page-table capacity"),
        ("host_lengths", ValueError, "requires one Host KV length per sequence"),
        ("prepared_state", RuntimeError, "prepared attention state is missing"),
    ],
)
def test_dspark_graph_validation_preserves_static_inputs(
    invalid: str,
    error_type: type[Exception],
    message: str,
) -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _block_draft_metadata(torch.tensor([[10, 11, 12]], dtype=torch.int32))
    entry = runner._allocate_entry(input_ids, positions, metadata, sequence_count=1, page_table_capacity=4)
    runner._fill_entry(entry, input_ids, positions, metadata)
    static = entry.static_metadata
    tensor_names = (
        "slot_mapping",
        "block_table",
        "kv_seq_lens",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
    )
    snapshots = {name: getattr(static, name).clone() for name in tensor_names}
    old_input_ids = entry.static_input_ids.clone()
    old_positions = entry.static_positions.clone()
    old_host_lengths = list(static.kv_seq_lens_host_values)
    prepared = static.prepared_attention_state
    old_query_ends = list(prepared.query_ends)
    old_query_lengths = list(prepared.actual_seq_q)
    old_kv_lengths = list(prepared.actual_seq_kv)
    invalid_metadata = _block_draft_metadata(torch.tensor([[20]], dtype=torch.int32))
    invalid_metadata.slot_mapping.add_(100)
    if invalid == "paged_metadata":
        invalid_metadata.paged_kv_indices = torch.arange(5, dtype=torch.int32)
        invalid_metadata.paged_kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
    elif invalid == "table_capacity":
        invalid_metadata = _block_draft_metadata(torch.zeros(1, 5, dtype=torch.int32))
    elif invalid == "host_lengths":
        invalid_metadata.kv_seq_lens_host_values = None
    else:
        static.prepared_attention_state = None

    with pytest.raises(error_type, match=message):
        runner._fill_entry(entry, input_ids + 100, positions + 100, invalid_metadata)

    torch.testing.assert_close(entry.static_input_ids, old_input_ids)
    torch.testing.assert_close(entry.static_positions, old_positions)
    for name, snapshot in snapshots.items():
        torch.testing.assert_close(getattr(static, name), snapshot)
    assert static.kv_seq_lens_host_values == old_host_lengths
    assert prepared.query_ends == old_query_ends
    assert prepared.actual_seq_q == old_query_lengths
    assert prepared.actual_seq_kv == old_kv_lengths


def test_block_draft_inherited_execute_keeps_entry_preparation_signature() -> None:
    runner = _block_draft_runner()
    runner._stream = Mock()
    caller = Mock()
    metadata = _block_draft_metadata(torch.tensor([[10, 11, 12]], dtype=torch.int32))
    input_ids = torch.arange(2, dtype=torch.int32)
    graph_key = runner._graph_key(input_ids, metadata, 1, 2)
    entry = SimpleNamespace(
        graph_tasks=[],
        graph=Mock(),
        static_output=torch.arange(8, dtype=torch.float32).view(4, 2),
        replay_logged=True,
    )

    with (
        patch.object(runner, "_prepare_graph_entry", autospec=True, return_value=entry) as prepare,
        patch.object(torch.npu, "current_stream", return_value=caller),
        patch.object(torch.npu, "stream", side_effect=lambda _: nullcontext()) as stream_context,
    ):
        output = runner.execute(input_ids, input_ids, metadata, graph_key=graph_key)

    prepare.assert_called_once_with(
        input_ids, input_ids, metadata, None, None, graph_key=graph_key, eplb=None, input_batch=None
    )
    entry.graph.replay.assert_called_once_with()
    stream_context.assert_called_once_with(caller)
    torch.testing.assert_close(output, entry.static_output[:2])
    assert output.data_ptr() == entry.static_output.data_ptr()


@pytest.mark.parametrize("has_tasks", [False, True])
@pytest.mark.parametrize("has_mtp_inputs", [False, True])
@pytest.mark.parametrize("mode", ["plain", "mtp_target", "spec_draft"])
def test_replay_selects_stream_and_preserves_output_ownership(has_tasks: bool, has_mtp_inputs: bool, mode: str) -> None:
    runner = _runner()
    runner.num_decoding_tokens = 4 if mode == "mtp_target" else 1
    runner._is_spec_draft = mode == "spec_draft"
    caller = Mock()
    capture = Mock()
    runner._stream = capture
    entry = SimpleNamespace(
        graph_tasks=[Mock()] if has_tasks else [],
        graph=Mock(),
        static_output=torch.ones((4, 2)),
        replay_logged=True,
    )
    update = Mock()
    with (
        patch.object(runner, "_prepare_graph_entry", return_value=entry),
        patch.object(runner, "_update_after_replay", update),
        patch.object(torch.npu, "current_stream", return_value=caller),
        patch.object(torch.npu, "stream", side_effect=lambda _: nullcontext()) as stream_context,
        patch.object(torch.npu, "Stream", return_value=Mock()),
        patch.object(torch.npu, "Event", return_value=Mock()),
    ):
        output = runner.execute(
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1),
            _metadata(torch.zeros(1)),
            input_embedding=torch.ones((1, 2)) if has_mtp_inputs else None,
        )
    replay_stream = capture if has_tasks else caller
    stream_context.assert_called_once_with(replay_stream)
    entry.graph.replay.assert_called_once_with()
    assert (output.data_ptr() != entry.static_output.data_ptr()) is (has_mtp_inputs or mode != "plain")
    if has_tasks:
        update.assert_called_once_with(entry, capture, runner._update_done_event)
        capture.wait_stream.assert_called_once_with(caller)
        caller.wait_stream.assert_called_once_with(capture)
        caller.wait_event.assert_called_once_with(runner._update_done_event)
    else:
        update.assert_not_called()
        caller.wait_stream.assert_not_called()
        caller.wait_event.assert_not_called()
        assert runner._update_stream is None
        assert runner._replay_done_event is None
        assert runner._taskless_replay_stream is caller


def test_taskless_update_does_not_require_events() -> None:
    runner = _runner()
    runner._update_after_replay(SimpleNamespace(graph_tasks=[]), Mock())
    assert runner._replay_done_event is None


@pytest.mark.parametrize("same_stream", [False, True])
def test_taskless_metadata_fill_waits_only_when_stream_changes(same_stream: bool) -> None:
    runner = _runner()
    runner._stream = Mock()
    previous = Mock(npu_stream=1)
    current = Mock(npu_stream=1 if same_stream else 2)
    runner._taskless_replay_stream = previous
    metadata = _metadata(torch.zeros(1, dtype=torch.int32))
    input_ids = torch.zeros(1, dtype=torch.int32)
    key = runner._graph_key(1, False, None, None)
    entry = SimpleNamespace(static_metadata=metadata, eplb=None)
    runner._graphs[key] = entry

    def fill_entry(*_args: object, expanded: object = None) -> None:
        if same_stream:
            current.wait_stream.assert_not_called()
        else:
            current.wait_stream.assert_called_once_with(previous)

    with (
        patch.object(torch.npu, "current_stream", return_value=current),
        patch.object(runner, "_prepare_attention"),
        patch.object(runner, "_fill_entry", side_effect=fill_entry),
    ):
        assert runner._prepare_graph_entry(input_ids, input_ids, metadata, None, graph_key=key) is entry


def test_slice_output_preserves_aux_hidden_tuple() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    aux_hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)

    output = DecodeAclGraphRunner._slice_output((hidden, aux_hidden), 3)

    assert isinstance(output, tuple)
    torch.testing.assert_close(output[0], hidden[:3])
    torch.testing.assert_close(output[1], aux_hidden[:3])


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    metadata.kpool_query_lens = [4]
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()
    assert entry.static_metadata.kpool_query_lens == (4, 4)
    assert entry.static_metadata.kpool_query_lens_device.tolist() == [4, 4]
    assert DecodeAclGraphRunner._graph_key(
        8,
        True,
        None,
        kpool_query_lens=entry.static_metadata.kpool_query_lens,
    ) != DecodeAclGraphRunner._graph_key(8, True, None, kpool_query_lens=(2, 2, 2, 2))

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )
        assert static_indices.tolist() == [3, 7, 11, 15, 0, 0, 0, 0]

        metadata.linear_state_indices = torch.tensor(
            [4, 8, 12, 16],
            dtype=torch.int32,
        )
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert static_indices.data_ptr() == data_ptr
    assert static_indices.tolist() == [4, 8, 12, 16, 0, 0, 0, 0]


def test_accepted_counts_refresh_in_stable_graph_buffer() -> None:
    runner = _runner()
    runner._has_linear_cache = True
    runner._model_owns_linear_spec_verify = True
    input_ids = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32), rows=4)
    metadata.q_seq_lens_host = torch.tensor([2, 2], dtype=torch.int32)
    metadata.q_cu_seq_lens = torch.tensor([0, 2, 4], dtype=torch.int32)
    metadata.expanded_decode_metadata = SimpleNamespace(
        enabled=True,
        kv_seq_lens=metadata.kv_seq_lens,
        block_table=metadata.block_table,
        paged_kv_indptr=metadata.paged_kv_indptr,
        paged_kv_indices=metadata.paged_kv_indices,
        paged_kv_last_page_len=metadata.paged_kv_last_page_len,
        paged_attention_tiling_data=None,
        kv_seq_lens_host=None,
        kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
    )
    metadata.num_accepted_tokens = torch.tensor([1, 2], dtype=torch.int32)
    metadata.num_accepted_tokens_host_values = (1, 2)
    entry = runner._allocate_entry(8, input_ids, metadata)
    counts = entry.static_metadata.num_accepted_tokens
    pointer = counts.data_ptr()
    expanded = runner._expanded_verify_view(metadata)
    assert expanded is not None
    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        runner._fill_entry(entry, input_ids, input_ids, metadata, 4, None, expanded=expanded)
        assert counts.tolist() == [1, 2, 1, 1]
        metadata.num_accepted_tokens = torch.tensor([2, 1], dtype=torch.int32)
        metadata.num_accepted_tokens_host_values = (2, 1)
        runner._fill_entry(entry, input_ids, input_ids, metadata, 4, None, expanded=expanded)
    assert counts.data_ptr() == pointer
    assert counts.tolist() == [2, 1, 1, 1]
    assert entry.static_metadata.linear_state_indices.tolist() == [1, 2, 0, 0]
    assert entry.static_metadata.q_cu_seq_lens.tolist() == [0, 1, 2, 3, 4, 4, 4, 4, 4]


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("state_ids", [(1,), (0,) * 64, (2, 0, 2, 1, 0), tuple(range(64))])
@pytest.mark.parametrize("capture_fails", [False, True])
def test_capture_preserves_gdn_live_checkpoints(
    index_dtype: torch.dtype, state_ids: tuple[int, ...], capture_fails: bool
) -> None:
    runner = _runner(max_batch=max(8, len(state_ids)), max_model_len=64)
    indices = torch.tensor(state_ids, dtype=index_dtype)
    slots = max(3, max(state_ids) + 1)
    conv = torch.arange(slots * 20, dtype=torch.float32).reshape(slots, 10, 2)
    ssm = torch.arange(slots * 32, dtype=torch.float32).reshape(slots * 8, 1, 2, 2)
    original_conv, original_ssm = conv.clone(), ssm.clone()
    runner.bind_layer_caches([LayerCache(key=None, value=None, conv=conv, ssm=ssm)])
    runner.attention_backend.prepare = Mock()
    runner.attention_backend.prepare_graph_replay = Mock()
    metadata = _metadata(indices)
    input_ids = torch.arange(indices.numel(), dtype=torch.int32)

    def capture(entry: AclGraphEntry, stream: object) -> None:
        selected = entry.static_metadata.linear_state_indices.long()
        checkpoints = (selected[:, None] * 8 + torch.arange(8)).reshape(-1)
        conv.index_fill_(0, selected, -1)
        ssm.index_fill_(0, checkpoints, -1)
        if capture_fails:
            raise RuntimeError("capture failed")

    with (
        patch.object(AclGraphRunner, "_capture", side_effect=capture),
        patch.object(torch.npu, "current_stream", return_value=Mock()),
        patch.object(torch.npu, "Stream", return_value=Mock()),
        patch(
            "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
            create=True,
        ),
    ):
        failure = pytest.raises(RuntimeError, match="capture failed") if capture_fails else nullcontext()
        with failure:
            runner._prepare_graph_entry(input_ids, input_ids, metadata, None)

    torch.testing.assert_close(conv[1:], original_conv[1:])
    torch.testing.assert_close(ssm[8:], original_ssm[8:])
    torch.testing.assert_close(metadata.linear_state_indices, torch.tensor(state_ids, dtype=index_dtype))


@pytest.mark.parametrize(
    ("width", "is_mla"),
    [(width, False) for width in (1, 2, 3, 4, 9, 17)] + [(3, True)],
)
@pytest.mark.parametrize("no_padding", [False, True])
@pytest.mark.parametrize("device_kind", ["cpu", "npu"])
def test_capture_preserves_live_state_until_first_replay(
    width: int,
    is_mla: bool,
    no_padding: bool,
    device_kind: str,
    request: pytest.FixtureRequest,
) -> None:
    if device_kind == "npu":
        device_name = os.getenv("XLLM_KDA_TEST_NPU_DEVICE")
        if not device_name:
            pytest.skip("NPU device not configured")
        pytest.importorskip("torch_npu")
        device = torch.device(device_name)
        previous_device = torch.npu.current_device()
        request.addfinalizer(lambda: torch.npu.set_device(previous_device))
        torch.npu.set_device(device)
    else:
        device = torch.device("cpu")
    runner = _runner(
        max_batch=max(8, 2 * width),
        max_model_len=max(8, 2 * width),
        num_decoding_tokens=width,
        spec_verify_width=1 if is_mla else width,
        enable_graph_mode_decode_no_padding=no_padding,
    )
    runner.attention_backend.is_mla = is_mla
    slots = torch.tensor([1, 3], dtype=torch.int32)
    rows = slots.numel() * width

    class NativeMetadata(SimpleNamespace):
        def __copy__(self) -> None:
            raise TypeError("native metadata views cannot be copied")

    metadata = NativeMetadata(**vars(_metadata(slots, rows=rows)))
    page_size = runner.attention_backend.page_size
    block_cols = (rows + page_size - 1) // page_size
    metadata.block_table = torch.arange(1, rows * block_cols + 1, dtype=torch.int32).reshape(rows, block_cols)
    metadata.paged_kv_indptr, metadata.paged_kv_indices, metadata.paged_kv_last_page_len = (
        build_row_aligned_paged_kv_metadata(metadata.block_table, metadata.kv_seq_lens_host_values, page_size)
    )
    metadata.has_initial_state = torch.ones_like(slots, dtype=torch.bool)
    metadata.num_accepted_tokens = torch.tensor([width, 1], dtype=torch.int32) if width > 1 else None
    metadata.num_accepted_tokens_host_values = (width, 1) if width > 1 else ()
    if width > 1:
        metadata.is_spec_verify = True
        metadata.q_seq_lens_host = torch.full((slots.numel(),), width, dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.arange(slots.numel() + 1, dtype=torch.int32) * width
        metadata.expanded_decode_metadata = SimpleNamespace(
            enabled=True,
            kv_seq_lens=metadata.kv_seq_lens,
            block_table=metadata.block_table,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=metadata.kv_seq_lens_host_values,
        )
    for name, value in vars(metadata).items():
        if isinstance(value, torch.Tensor) and not name.endswith("_host"):
            setattr(metadata, name, value.to(device))
    conv = torch.arange(8, dtype=torch.float32, device=device).reshape(8, 1)
    ssm = torch.arange(8 * width, dtype=torch.float32, device=device).reshape(8 * width, 1)
    original_conv, original_ssm = conv.clone(), ssm.clone()
    runner.bind_layer_caches([LayerCache(key=None, value=None, conv=conv, ssm=ssm)])
    output = torch.empty(rows, 1, device=device)
    prepared: list[StaticGraphAttentionMetadata] = []
    pointers: list[int] = []

    def prepare(static: StaticGraphAttentionMetadata, **_: object) -> None:
        prepared.append(static)
        pointers.append(static.linear_state_indices.data_ptr())

    def advance() -> torch.Tensor:
        static = prepared[-1]
        indices = static.linear_state_indices.long()
        checkpoints = indices[:, None] * width + torch.arange(width, device=device)
        accepted = static.num_accepted_tokens
        columns = accepted.long() - 1 if accepted is not None else torch.zeros_like(indices)
        values = ssm[checkpoints[torch.arange(indices.numel(), device=device), columns]].flatten() + 1
        conv.index_copy_(0, indices, conv[indices] + 1)
        ssm.index_copy_(0, checkpoints.flatten().long(), values.repeat_interleave(width).view(-1, 1))
        output.copy_(values[: slots.numel()].repeat_interleave(width).view(-1, 1))
        return output

    def capture(entry: AclGraphEntry, stream: object) -> None:
        assert not entry.static_metadata.linear_state_indices.any()
        for _ in range(3):
            advance()
        torch.testing.assert_close(conv[1:], original_conv[1:])
        torch.testing.assert_close(ssm[width:], original_ssm[width:])
        entry.static_output = output
        entry.graph_tasks = []
        entry.graph = Mock()
        entry.graph.replay.side_effect = advance

    runner.attention_backend.prepare = prepare
    runner.attention_backend.prepare_graph_replay = prepare
    input_ids = torch.arange(rows, dtype=torch.int32, device=device)
    with ExitStack() as stack:
        if device_kind == "cpu":
            stack.enter_context(patch.object(AclGraphRunner, "_capture", side_effect=capture))
            stack.enter_context(patch.object(torch.npu, "current_stream", return_value=Mock()))
            stack.enter_context(patch.object(torch.npu, "stream", side_effect=lambda _: nullcontext()))
            stack.enter_context(patch.object(torch.npu, "Stream", return_value=Mock()))
            stack.enter_context(
                patch(
                    "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
                    create=True,
                )
            )
        else:
            stack.enter_context(patch.object(runner, "_forward_static", side_effect=lambda _: advance()))
        result = runner.execute(input_ids, input_ids, metadata)
    slots = slots.to(device)
    columns = torch.tensor([width - 1, 0], device=device)
    expected = original_ssm[slots.long() * width + columns].flatten() + 1
    torch.testing.assert_close(result.flatten(), expected.repeat_interleave(width))
    torch.testing.assert_close(conv[slots.long()], original_conv[slots.long()] + 1)
    checkpoints = slots.long()[:, None] * width + torch.arange(width, device=device)
    torch.testing.assert_close(ssm[checkpoints].squeeze(-1), expected[:, None].expand_as(checkpoints))
    assert len(set(pointers)) == 1
    torch.testing.assert_close(metadata.linear_state_indices, slots)
    assert metadata.has_initial_state.all()
    if width > 1:
        assert metadata.num_accepted_tokens.tolist() == [width, 1]


def test_untyped_verify_reuses_row_aligned_paging_metadata() -> None:
    runner = _runner()
    metadata = _metadata(torch.tensor([1, 2], dtype=torch.int32), rows=4)

    expanded = runner._expanded_verify_view(metadata)

    assert expanded is not None
    assert expanded.paged_kv_indptr is metadata.paged_kv_indptr
    assert expanded.paged_kv_indices is metadata.paged_kv_indices
    assert expanded.paged_kv_last_page_len is metadata.paged_kv_last_page_len


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = _runner(dp_size=dp_size, enable_mega_moe_token_mask=True)
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        metadata=metadata,
    )
    mask = entry.static_metadata.mega_moe_token_mask
    data_ptr = mask.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for remote_count in (3, 1):
            metadata.dp_execution_token_counts = (4, remote_count)[:dp_size]
            runner._fill_entry(entry, input_ids, positions, metadata, batch_size=4, input_embedding=None)
            expected = [True] * 4 + [False] * 4
            if dp_size == 2:
                expected += [True] * remote_count + [False] * (8 - remote_count)
            assert mask.tolist() == expected
            assert entry.static_metadata.mega_moe_token_mask.data_ptr() == data_ptr


def test_dsa_graph_tables_use_compressed_block_counts() -> None:
    attention_backend = DsaAttentionBackend(
        [1, 4, 128], 128, 3, 16, 512, 512, 32, 128, 64, torch.device("cpu"), torch.bfloat16
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=32768,
    )
    runner._max_blocks_per_sequence = 256

    tables = attention_backend.create_graph_block_tables(4, runner.max_model_len, runner._max_blocks_per_sequence)

    assert [tuple(table.shape) for table in tables] == [(4, 256), (4, 64), (4, 2)]
    assert all(torch.all(table == 0) for table in tables)

    block_table = torch.tensor([[10, 11], [20, 21]], dtype=torch.int32)
    indptr, indices, last_page_len = build_row_aligned_paged_kv_metadata(
        block_table,
        [129, 132],
        attention_backend.page_size,
    )
    assert indptr.tolist() == [0, 2, 4]
    assert indices.tolist() == [10, 11, 20, 21]
    assert last_page_len.tolist() == [1, 4]

    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.kv_seq_lens_host_values = None
    metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
    metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_dsa_graph_refreshes_every_manager_and_clears_tails() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    attention_backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
    )
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10, 11], [12, 13]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 11, 0, 0],
        [12, 13, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [20, 0],
        [21, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[30], [31], [0], [0]]

    metadata.multi_block_tables = (
        torch.tensor([[40]], dtype=torch.int32),
        torch.tensor([[50]], dtype=torch.int32),
        torch.tensor([[60]], dtype=torch.int32),
    )
    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        batch_size=1,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [40, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert static_metadata.multi_block_tables[1].tolist() == [
        [50, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    assert static_metadata.multi_block_tables[2].tolist() == [[60], [0], [0], [0]]


def test_dsa_graph_padding_uses_reserved_single_kv_length() -> None:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(
            page_size=128,
            is_mla=False,
            group_infos=group_infos,
        ),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(kv_seq_lens_host_values=[1, 1, 1, 1]),
    )

    runner._fill_host_metadata(entry, [9, 17], batch_size=2)

    assert entry.static_metadata.kv_seq_lens_host_values == [9, 17, 1, 1]


def test_dsa_graph_positions_are_refreshed_from_current_input() -> None:
    runner = _runner()
    entry = SimpleNamespace(
        batch_size=4,
        static_positions=torch.tensor([1, 2, 3, 4], dtype=torch.int32),
        static_metadata=SimpleNamespace(dsa_positions=None),
    )

    runner._fill_graph_dsa_positions(
        entry,
        torch.tensor([20, 21], dtype=torch.int32),
    )

    assert entry.static_metadata.dsa_positions.tolist() == [20, 21, 0, 0]


@pytest.mark.parametrize(
    "capacity,peer_rows,padded_rows",
    [(8, 5, 8), (8, 8, 8), (7, 5, 7), (8, 9, None)],
    ids=["peer-bucket", "at-capacity", "partial-bucket", "over-capacity"],
)
def test_dp_empty_rank_uses_group_wide_acl_graph_bucket(capacity: int, peer_rows: int, padded_rows: int | None) -> None:
    runner = _runner()
    runner.dp_size = 2
    runner.dp_rank = 1
    runner.max_batch = capacity
    metadata = _metadata(torch.zeros(1, dtype=torch.int32))
    metadata.dp_execution_token_counts = (peer_rows, 1)
    metadata.dp_is_decode = (1, 1)
    input_ids = torch.zeros(1, dtype=torch.int32)
    assert runner.can_execute(input_ids, metadata) is (padded_rows is not None)
    if padded_rows is None:
        with pytest.raises(ValueError, match="decode batch exceeds ACL graph capacity"):
            runner._padded_batch_size(1, metadata)
    else:
        assert runner._padded_batch_size(1, metadata) == padded_rows


@pytest.mark.parametrize("counts,phases", [((3, 2), (0, 1)), ((3,), (1, 1))])
def test_dp_acl_graph_rejects_invalid_peers(counts: tuple[int, ...], phases: tuple[int, ...]) -> None:
    runner = _runner()
    runner.dp_size = 2
    metadata = _metadata(torch.arange(3, dtype=torch.int32))
    metadata.dp_execution_token_counts = counts
    metadata.dp_is_decode = phases
    if len(counts) != 2:
        with pytest.raises(RuntimeError, match="valid dp_execution_token_counts"):
            runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)
    else:
        assert not runner.can_execute(torch.zeros(3, dtype=torch.int32), metadata)


def test_dp_graph_variant_ids_distinguish_mtp_input_signatures() -> None:
    runner = _runner()
    runner.dp_size = 2
    ids = torch.ones(1, dtype=torch.int32)
    topk = torch.ones((1, 1, 4), dtype=torch.int32)
    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("xllm.python.distributed.all_gather") as gather,
    ):
        keys = []
        for indices, variants in ((None, (1, 7)), (topk, (2, 11)), (None, (1, 13))):
            gather.return_value = torch.tensor(variants, dtype=torch.int32)
            local_key = runner._graph_key(2, False, None, indices)
            key = runner._synchronize_dp_graph_key(local_key, ids)
            assert key == (*local_key, variants)
            keys.append(key)
        assert keys[0] != keys[2]
    assert [int(call.args[0].item()) for call in gather.call_args_list] == [1, 2, 1]


@pytest.mark.parametrize("detach", [False, True], ids=["view", "detached"])
def test_mtp_graph_output_slices_and_detaches_replay_buffers(detach: bool) -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    topk = torch.arange(24, dtype=torch.int64).reshape(4, 2, 3)

    sliced = DecodeAclGraphRunner._slice_output((hidden, None, topk), 2, detach=detach)

    assert isinstance(sliced, tuple)
    sliced_hidden, sliced_aux, sliced_topk = sliced
    assert sliced_aux is None
    assert torch.equal(sliced_hidden, hidden[:2])
    assert torch.equal(sliced_topk, topk[:2])
    assert (sliced_hidden.data_ptr() != hidden.data_ptr()) is detach
    assert (sliced_topk.data_ptr() != topk.data_ptr()) is detach
    hidden.zero_()
    topk.zero_()
    assert (torch.count_nonzero(sliced_hidden).item() > 0) is detach
    assert (torch.count_nonzero(sliced_topk).item() > 0) is detach


def test_mtp_graph_key_separates_first_step_and_topk_shapes() -> None:
    topk = torch.ones((4, 1, 8), dtype=torch.int32)
    key = DecodeAclGraphRunner._graph_key(8, False, None, topk)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None)
    assert key == DecodeAclGraphRunner._graph_key(8, False, None, topk + 1)
    assert key != DecodeAclGraphRunner._graph_key(8, False, None, topk[:, :, :4])


def test_mtp_topk_input_changes_without_reallocating_capture_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = input_ids.clone()
    metadata = _metadata(input_ids)
    topk = torch.arange(24, dtype=torch.int32).reshape(4, 2, 3)
    entry = runner._allocate_entry(8, input_ids, metadata, topk)
    address = entry.static_mtp_topk_indices.data_ptr()

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ):
        for source in (topk, topk.flip(0) + 7):
            runner._fill_entry(entry, input_ids, positions, metadata, 4, None, source)
            assert entry.static_mtp_topk_indices.data_ptr() == address
            torch.testing.assert_close(entry.static_mtp_topk_indices[:4], source)
            assert torch.count_nonzero(entry.static_mtp_topk_indices[4:]) == 0


def test_dsa_graph_clamps_target_tables_to_draft_groups() -> None:
    _, group_infos = build_cache_specs([1], 128, 1)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False, group_infos=group_infos),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
    )
    static_metadata = SimpleNamespace(multi_block_tables=(torch.full((4, 2), 99, dtype=torch.int32),))
    metadata = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[10], [11]], dtype=torch.int32),
            torch.tensor([[20], [21]], dtype=torch.int32),
            torch.tensor([[30], [31]], dtype=torch.int32),
        )
    )

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        batch_size=2,
    )

    assert static_metadata.multi_block_tables[0].tolist() == [
        [10, 0],
        [11, 0],
        [0, 0],
        [0, 0],
    ]


def test_dsa_graph_pads_new_cache_slots_to_graph_bucket() -> None:
    entry = SimpleNamespace(
        batch_size=16,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[301, 302, 401, 402])

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=4,
    )

    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402] + [0] * 12


def test_dsa_graph_padding_preserves_scheduler_resolved_swa_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[]),
    )
    metadata = SimpleNamespace(
        new_cache_slots_host_values=[1280, 2560, 3840],
    )
    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=3,
    )

    caches_info, group_infos = build_cache_specs([0, 4, 128, 4], 128, 4)
    builder = DsaMetadataBuilder(caches_info, group_infos)
    dsa = builder.build(
        multi_block_tables=[
            torch.tensor(
                [
                    [10, 11, 0, 0],
                    [20, 21, 0, 0],
                    [30, 31, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=torch.int32,
            ),
            torch.zeros((4, 4), dtype=torch.int32),
            torch.zeros((4, 4), dtype=torch.int32),
        ],
        kv_seq_lens=[257, 257, 257, 1],
        q_seq_lens=[1, 1, 1, 1],
        positions=torch.tensor([256, 256, 256, 0], dtype=torch.int64),
        is_prefill=False,
        is_chunked_prefill=False,
        new_cache_slots=entry.static_metadata.new_cache_slots_host_values,
        enable_graph=True,
        graph_block_table_capacity_cols=4,
    )

    assert dsa.slot_mappings[0][0].tolist() == [1280, 2560, 3840, 0]


def test_dsa_graph_allows_dummy_without_scheduler_cache_slots() -> None:
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=SimpleNamespace(new_cache_slots_host_values=[99]),
    )
    metadata = SimpleNamespace(new_cache_slots_host_values=[], is_dummy=True)

    DecodeAclGraphRunner._fill_new_cache_slots_host_metadata(
        entry,
        metadata,
        batch_size=1,
    )

    assert entry.static_metadata.new_cache_slots_host_values == []


def test_dsv4_decode_admits_scheduler_slots_when_top_level_mapping_is_empty() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.block_table = None
    metadata.multi_block_tables = (torch.tensor([[10, 0], [20, 0], [30, 0], [40, 0]], dtype=torch.int32),)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]

    assert runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)
    assert runner._effective_slot_mapping(metadata, 4, torch.device("cpu")).tolist() == [
        301,
        302,
        401,
        402,
    ]


def test_fill_entry_uses_scheduler_slots_for_dsv4_decode() -> None:
    runner = _runner()
    runner.attention_backend = DsaAttentionBackend(
        [1], 4, 1, 16, 512, 512, 32, 128, 64, torch.device("cpu"), torch.bfloat16
    )

    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)
    metadata.multi_block_tables = (metadata.block_table,)
    metadata.new_cache_slots_host_values = [301, 302, 401, 402]
    entry = runner._allocate_entry(8, input_ids, metadata)

    with patch(
        "xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata",
        create=True,
    ) as update_metadata:
        runner._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size=4,
            input_embedding=None,
        )

    assert update_metadata.call_args.args[2].tolist() == [301, 302, 401, 402]
    assert entry.static_metadata.new_cache_slots_host_values == [301, 302, 401, 402, 0, 0, 0, 0]


def test_dp_real_and_dummy_decode_share_graph_admission() -> None:
    def make_metadata(*, is_dummy: bool) -> SimpleNamespace:
        metadata = _metadata(torch.tensor([0], dtype=torch.int32))
        metadata.slot_mapping = torch.tensor([0], dtype=torch.int32) if is_dummy else torch.empty(0, dtype=torch.int32)
        metadata.block_table = None
        metadata.multi_block_tables = (torch.tensor([[0]], dtype=torch.int32),)
        metadata.kv_seq_lens = torch.tensor([1], dtype=torch.int32)
        metadata.kv_seq_lens_host_values = [1]
        metadata.kv_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.q_cu_seq_lens = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32)
        metadata.paged_kv_indices = torch.tensor([0], dtype=torch.int32)
        metadata.paged_kv_last_page_len = torch.tensor([1], dtype=torch.int32)
        metadata.new_cache_slots_host_values = [] if is_dummy else [301]
        metadata.dp_execution_token_counts = (1, 1)
        metadata.dp_is_decode = (1, 1)
        metadata.is_dummy = is_dummy
        return metadata

    real_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=0,
    )
    dummy_runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=8,
        dp_size=2,
        dp_rank=1,
    )
    input_ids = torch.tensor([1], dtype=torch.int32)

    assert real_runner.can_execute(input_ids, make_metadata(is_dummy=False))
    assert dummy_runner.can_execute(input_ids, make_metadata(is_dummy=True))


def test_decode_rejects_mismatched_linear_state_validity_mask() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.has_initial_state = torch.ones(3, dtype=torch.int32)

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_decode_rejects_mismatched_linear_state_indices_without_cache_probe() -> None:
    runner = _runner()
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.linear_state_indices = torch.arange(3, dtype=torch.int32)

    with patch.object(runner, "_has_compatible_index_history", side_effect=AssertionError):
        assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


@pytest.mark.parametrize("speculative_tokens", [3, 7], ids=["mtp", "dspark"])
@pytest.mark.parametrize("dp_size", [1, 2])
def test_mla_speculative_decode_ignores_unused_linear_state_rows(speculative_tokens: int, dp_size: int) -> None:
    rows = speculative_tokens + 1
    backend = SimpleNamespace(page_size=4, is_mla=True, graph_index_history_max_kv=32)
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        backend,
        torch.device("cpu"),
        max_batch=dp_size * rows,
        max_model_len=32,
        dp_size=dp_size,
        num_decoding_tokens=rows,
    )
    runner.bind_layer_caches([LayerCache(torch.zeros(1), torch.zeros(1), index=torch.zeros(1))])
    metadata = _metadata(torch.arange(rows, dtype=torch.int32))
    metadata.linear_state_indices = torch.tensor([-1], dtype=torch.int32)
    metadata.has_initial_state = torch.zeros(1, dtype=torch.bool)
    metadata.dp_global_sequence_nums = (1,) * dp_size
    metadata.dp_execution_token_counts = (rows,) * dp_size
    metadata.dp_global_kv_max_seq_lens = (rows,) * dp_size
    metadata.dp_is_decode = (1,) * dp_size

    assert runner.can_execute(torch.arange(rows, dtype=torch.int32), metadata)


@pytest.mark.parametrize("cache_slot", ["conv", "ssm"])
@pytest.mark.parametrize("invalid_field", ["has_initial_state", "linear_state_indices"])
def test_hybrid_mla_rejects_mismatched_recurrent_metadata(cache_slot: str, invalid_field: str) -> None:
    runner = _runner()
    runner.attention_backend.is_mla = True
    runner.bind_layer_caches([LayerCache(torch.zeros(1), torch.zeros(1), **{cache_slot: torch.zeros(1)})])
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    setattr(metadata, invalid_field, torch.zeros(1, dtype=torch.int32))

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)


def test_mla_admission_retains_index_history_capacity_check() -> None:
    runner = _runner()
    runner.attention_backend.is_mla = True
    runner.attention_backend.graph_index_history_max_kv = 4
    runner.bind_layer_caches([LayerCache(torch.zeros(1), torch.zeros(1), index=torch.zeros(1))])
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    metadata.linear_state_indices = torch.tensor([-1], dtype=torch.int32)

    assert not runner.can_execute(torch.arange(4, dtype=torch.int32), metadata)
