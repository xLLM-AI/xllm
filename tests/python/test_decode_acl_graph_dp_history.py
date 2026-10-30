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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _runner(
    dp_rank: int,
    *,
    dp_size: int = 2,
    cap: int = 32768,
    page_size: int = 128,
    is_spec_draft: bool = False,
    num_decoding_tokens: int = 1,
) -> DecodeAclGraphRunner:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=page_size, is_mla=False, graph_index_history_max_kv=cap),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=65536,
        dp_size=dp_size,
        dp_rank=dp_rank,
        num_decoding_tokens=num_decoding_tokens,
        is_spec_draft=is_spec_draft,
    )
    runner.bind_layer_caches([LayerCache(torch.empty(1), torch.empty(1), index=torch.empty(1))])
    return runner


def _metadata(
    local_length: int,
    global_lengths: tuple[int, ...],
    *,
    page_size: int = 128,
    table_columns: int | None = None,
    is_dummy: bool = False,
) -> SimpleNamespace:
    columns = table_columns if table_columns is not None else (local_length + page_size - 1) // page_size
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        is_dummy=is_dummy,
        dp_execution_token_counts=(1, 1),
        dp_is_decode=(1, 1),
        dp_global_kv_max_seq_lens=global_lengths,
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        new_cache_slots_host_values=[0],
        paged_kv_indptr=torch.tensor([0, columns], dtype=torch.int32),
        paged_kv_indices=torch.arange(columns, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([((local_length - 1) % page_size) + 1], dtype=torch.int32),
        block_table=torch.arange(columns, dtype=torch.int32).reshape(1, columns),
        kv_seq_lens=torch.tensor([local_length], dtype=torch.int32),
        kv_seq_lens_host_values=[local_length],
        max_seq_len=local_length,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_cu_seq_lens=torch.tensor([0, local_length], dtype=torch.int32),
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        expanded_decode_metadata=None,
        multi_block_tables=(),
    )


@pytest.mark.parametrize("active_dp", [0, 1])
@pytest.mark.parametrize("page_size,cap", [(128, 32768), (128, 256), (64, 256)])
@pytest.mark.parametrize("offset,expected", [(-1, True), (0, True), (1, False)])
def test_history_admission_is_shared(active_dp: int, page_size: int, cap: int, offset: int, expected: bool) -> None:
    lengths = [0, 0]
    lengths[active_dp] = cap + offset
    input_ids = torch.tensor([1], dtype=torch.int32)
    decisions = []
    for rank in range(2):
        is_dummy = rank != active_dp
        metadata = _metadata(
            1 if is_dummy else cap + offset,
            tuple(lengths),
            page_size=page_size,
            is_dummy=is_dummy,
        )
        decisions.append(_runner(rank, cap=cap, page_size=page_size).can_execute(input_ids, metadata))
    assert decisions == [expected, expected]


@pytest.mark.parametrize("long_dp", [0, 1])
@pytest.mark.parametrize(
    "short_length,long_length,expected",
    [
        (1024, 32767, True),
        (1024, 32768, True),
        (1024, 32769, False),
        (32769, 32896, False),
    ],
    ids=["below-cap", "at-cap", "cross-cap", "both-above-cap"],
)
def test_nonempty_dp_histories_share_admission(
    long_dp: int, short_length: int, long_length: int, expected: bool
) -> None:
    lengths = [short_length, short_length]
    lengths[long_dp] = long_length
    input_ids = torch.tensor([1], dtype=torch.int32)
    decisions = []
    for rank, local_length in enumerate(lengths):
        metadata = _metadata(local_length, tuple(lengths), is_dummy=False)
        decisions.append(_runner(rank).can_execute(input_ids, metadata))
    assert decisions == [expected, expected]


@pytest.mark.parametrize("active_dp", [0, 1])
def test_history_ignores_unused_reserved_pages(active_dp: int) -> None:
    lengths = [0, 0]
    lengths[active_dp] = 128
    input_ids = torch.tensor([1], dtype=torch.int32)
    decisions = []
    for rank in range(2):
        metadata = _metadata(
            128 if rank == active_dp else 1,
            tuple(lengths),
            table_columns=3 if rank == active_dp else 1,
            is_dummy=rank != active_dp,
        )
        decisions.append(_runner(rank, cap=128).can_execute(input_ids, metadata))
    assert decisions == [True, True]


def test_history_uses_full_global_length_with_short_local_tables() -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    decisions = []
    for rank in range(2):
        metadata = _metadata(129 if rank == 0 else 1, (257, 0), page_size=64)
        decisions.append(_runner(rank, cap=256, page_size=64).can_execute(input_ids, metadata))
    assert decisions == [False, False]


@pytest.mark.parametrize("global_length,expected", [(253, True), (254, False)])
def test_verify_rows_count_against_the_shared_cap(global_length: int, expected: bool) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    runner = _runner(0, cap=256, num_decoding_tokens=4)
    assert runner.can_execute(input_ids, _metadata(1, (global_length, 0))) is expected


@pytest.mark.parametrize("legacy_value", [None, (), []])
def test_missing_global_history_uses_eager(legacy_value: object) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(2):
        metadata = _metadata(1, (1, 0))
        if legacy_value is None:
            del metadata.dp_global_kv_max_seq_lens
        else:
            metadata.dp_global_kv_max_seq_lens = legacy_value
        assert not _runner(rank).can_execute(input_ids, metadata)


@pytest.mark.parametrize("lengths", [(1,), (-1, 0), (True, 0), (1.0, 0), torch.tensor([1, 0])])
def test_invalid_shared_host_history_is_rejected(lengths: object) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(2):
        metadata = _metadata(1, (1, 0))
        metadata.dp_global_kv_max_seq_lens = lengths
        with pytest.raises(RuntimeError, match="DP index-history lengths"):
            _runner(rank).can_execute(input_ids, metadata)


@pytest.mark.parametrize("cap", [0, 127, 129])
def test_unsupported_page_capacity_uses_eager(cap: int) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(2):
        assert not _runner(rank, cap=cap).can_execute(input_ids, _metadata(1, (1, 0)))


def test_non_index_dp_does_not_require_global_history() -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(2):
        runner = _runner(rank)
        runner.bind_layer_caches([])
        metadata = _metadata(1, (1, 0))
        del metadata.dp_global_kv_max_seq_lens
        assert runner.can_execute(input_ids, metadata)


@pytest.mark.parametrize("table_columns,expected", [(1, True), (2, False)])
def test_tp_only_preserves_local_table_rule(table_columns: int, expected: bool) -> None:
    runner = _runner(0, dp_size=1, cap=128)
    metadata = _metadata(1, (65536,), table_columns=table_columns)
    assert runner.can_execute(torch.tensor([1], dtype=torch.int32), metadata) is expected


def test_mtp_target_without_global_history_uses_eager() -> None:
    for rank in range(2):
        runner = _runner(rank, num_decoding_tokens=4)
        metadata = _metadata(1, ())
        del metadata.dp_global_kv_max_seq_lens
        assert not runner.can_execute(torch.tensor([1], dtype=torch.int32), metadata)


@pytest.mark.parametrize("rank", [0, 1])
def test_dp_index_draft_does_not_read_stale_history(rank: int) -> None:
    class DraftMetadata(SimpleNamespace):
        @property
        def dp_global_kv_max_seq_lens(self) -> object:
            raise AssertionError("draft global history is not fresh")

    values = vars(_metadata(32769 if rank == 0 else 1, (1, 0))).copy()
    del values["dp_global_kv_max_seq_lens"]
    metadata = DraftMetadata(**values)
    assert not _runner(rank, is_spec_draft=True).can_execute(torch.tensor([1], dtype=torch.int32), metadata)


def test_tp_only_index_draft_remains_graphable() -> None:
    runner = _runner(0, dp_size=1, is_spec_draft=True)
    assert runner.can_execute(torch.tensor([1], dtype=torch.int32), _metadata(1, ()))


def test_dp_non_index_draft_remains_graphable() -> None:
    for rank in range(2):
        runner = _runner(rank, is_spec_draft=True)
        runner.bind_layer_caches([])
        assert runner.can_execute(torch.tensor([1], dtype=torch.int32), _metadata(1, ()))
