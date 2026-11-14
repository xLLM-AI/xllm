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

"""Empty-DP coverage for the NPU ACL decode-graph runner."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.dsa_metadata import build_cache_specs
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def test_dsa_graph_empty_rank_uses_reserved_block_zero() -> None:
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
    static_metadata = SimpleNamespace(
        multi_block_tables=(
            torch.full((4, 4), 99, dtype=torch.int32),
            torch.full((4, 2), 99, dtype=torch.int32),
            torch.full((4, 1), 99, dtype=torch.int32),
        )
    )
    metadata = SimpleNamespace(multi_block_tables=(), is_dummy=True)

    runner._fill_dsa_block_tables(
        static_metadata,
        metadata,
        batch_size=1,
    )

    for table in static_metadata.multi_block_tables:
        assert torch.all(table == 0)


def test_linear_verify_state_pads_accepted_counts_and_indices_for_bucket_tail() -> None:
    counts = (3, 1)
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.full((5,), 7, dtype=torch.int32),
            num_accepted_tokens=torch.full((5,), 5, dtype=torch.int32),
            num_accepted_tokens_host_values=(9, 9),
        ),
    )
    DecodeAclGraphRunner._fill_linear_verify_state(
        entry,
        SimpleNamespace(
            num_accepted_tokens=torch.tensor(counts, dtype=torch.int32),
            num_accepted_tokens_host_values=tuple(counts),
        ),
        (4, 4),
        torch.tensor([5, 9], dtype=torch.int32),
    )
    assert entry.static_metadata.linear_state_indices.tolist() == [5, 9, 0, 0, 0]
    assert entry.static_metadata.num_accepted_tokens.tolist() == [*counts, 1, 1, 1]
    assert entry.static_metadata.num_accepted_tokens_host_values == (*counts, 1, 1, 1)


@pytest.mark.parametrize("counts", [(0, 1), (1, 5)])
def test_linear_verify_state_rejects_counts_outside_verify_width(counts: tuple[int, ...]) -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens=torch.ones(5, dtype=torch.int32),
            num_accepted_tokens_host_values=(),
        ),
    )
    with pytest.raises(ValueError, match=r"in \[1, verify width\]"):
        DecodeAclGraphRunner._fill_linear_verify_state(
            entry,
            SimpleNamespace(
                num_accepted_tokens=torch.tensor(counts, dtype=torch.int32),
                num_accepted_tokens_host_values=tuple(counts),
            ),
            (4, 4),
            torch.tensor([5, 9], dtype=torch.int32),
        )


@pytest.mark.parametrize("bad_accepted", [None, torch.ones(3, dtype=torch.int32)])
def test_linear_verify_state_requires_sequence_scoped_accepted_counts(bad_accepted: torch.Tensor | None) -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens=torch.zeros(5, dtype=torch.int32),
            num_accepted_tokens_host_values=(),
        ),
    )
    with pytest.raises(ValueError, match="sequence-scoped"):
        DecodeAclGraphRunner._fill_linear_verify_state(
            entry,
            SimpleNamespace(num_accepted_tokens=bad_accepted, num_accepted_tokens_host_values=()),
            (4, 4),
            torch.tensor([5, 9], dtype=torch.int32),
        )


def test_linear_verify_state_accepts_row_expanded_state_indices() -> None:
    entry = SimpleNamespace(
        static_metadata=SimpleNamespace(
            linear_state_indices=torch.full((4,), 7, dtype=torch.int32),
            num_accepted_tokens=torch.full((4,), 5, dtype=torch.int32),
            num_accepted_tokens_host_values=(9,),
        ),
    )
    DecodeAclGraphRunner._fill_linear_verify_state(
        entry,
        SimpleNamespace(num_accepted_tokens=torch.tensor([2], dtype=torch.int32)),
        (3,),
        torch.tensor([8, 8, 8], dtype=torch.int32),
    )
    assert entry.static_metadata.linear_state_indices.tolist() == [8, 0, 0, 0]
    assert entry.static_metadata.num_accepted_tokens.tolist() == [2, 1, 1, 1]


def _dsa_metadata(tables: tuple = (), *, dummy: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        slot_mapping=torch.empty(0, dtype=torch.int32),
        paged_kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        paged_kv_indices=torch.tensor([0], dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(1, dtype=torch.int32),
        block_table=None,
        kv_seq_lens=torch.ones(1, dtype=torch.int32),
        kv_seq_lens_host_values=[1],
        kv_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        linear_state_indices=None,
        expanded_decode_metadata=None,
        multi_block_tables=tables,
        new_cache_slots_host_values=[] if dummy else [301],
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=dummy,
        dp_execution_token_counts=(1, 1),
        dp_is_decode=(1, 1),
        dp_global_kv_max_seq_lens=(1, 1),
    )


def _graph_block_tables(
    batch_size: int,
    max_model_len: int,
    max_block_columns: int,
) -> tuple[torch.Tensor, ...]:
    return tuple(torch.zeros((batch_size, max_block_columns), dtype=torch.int32) for _ in range(3))


def _dsa_runner(**backend_overrides: object) -> DecodeAclGraphRunner:
    _, group_infos = build_cache_specs([1, 4, 128], 128, 3)
    backend = SimpleNamespace(
        page_size=128,
        is_mla=False,
        group_infos=group_infos,
        create_graph_block_tables=_graph_block_tables,
        **backend_overrides,
    )
    return DecodeAclGraphRunner(
        nn.Identity(),
        backend,
        torch.device("cpu"),
        max_batch=4,
        max_model_len=512,
        dp_size=2,
        dp_rank=1,
    )


def test_empty_dp_shard_normalizes_reserved_graph_tables() -> None:
    runner = _dsa_runner()
    metadata = _dsa_metadata()

    view = runner._normalize_dsa_metadata(metadata)
    assert view is not metadata
    assert tuple(view.block_table.shape) == (1, 1)
    assert view.slot_mapping.tolist() == [0]
    assert [tuple(table.shape) for table in view.multi_block_tables] == [(1, 1)] * 3
    assert view.kv_seq_lens_host_values == [1]
    assert view.is_dummy is True


def test_empty_dp_shard_normalized_entry_fills_and_keeps_table_zero() -> None:
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    positions = torch.zeros_like(input_ids)
    metadata = runner._normalize_dsa_metadata(_dsa_metadata())

    entry = runner._allocate_entry(4, input_ids, metadata)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)

    for table in entry.static_metadata.multi_block_tables:
        assert torch.all(table == 0)


def test_token_capacity_bucket_uses_group_maximum() -> None:
    runner = _dsa_runner(graph_token_capacity_granularity=128)

    in_range = _dsa_metadata()
    in_range.dp_global_kv_max_seq_lens = (1, 200)
    assert runner._token_capacity_bucket(in_range) == 2

    beyond_model_len = _dsa_metadata()
    beyond_model_len.dp_global_kv_max_seq_lens = (1, 100000)
    assert runner._token_capacity_bucket(beyond_model_len) > (512 + 127) // 128


def test_busy_dsa_rank_fills_graph_without_flat_scheduler_slots() -> None:
    runner = _dsa_runner()
    input_ids = torch.tensor([1], dtype=torch.int32)
    manager_table = torch.tensor([[3]], dtype=torch.int32)
    metadata = _dsa_metadata((manager_table,) * 3, dummy=False)
    metadata.new_cache_slots_host_values = []
    metadata.slot_mapping = torch.empty(0, dtype=torch.int32)

    metadata = runner._normalize_dsa_metadata(metadata)
    assert runner._effective_slot_mapping(metadata, 1, torch.device("cpu")).tolist() == [0]

    positions = torch.zeros_like(input_ids)
    entry = runner._allocate_entry(4, input_ids, metadata)
    with patch("xllm.python.model_executor.runners.decode_acl_graph.kernels.update_decode_graph_metadata", create=True):
        runner._fill_entry(entry, input_ids, positions, metadata, batch_size=1, input_embedding=None)
    assert entry.static_metadata.new_cache_slots_host_values == []
    zero_row = [0] * 5
    expected = [[3, 0, 0, 0, 0], zero_row, zero_row, zero_row]
    for table in entry.static_metadata.multi_block_tables:
        assert table.tolist() == expected


@pytest.mark.parametrize("has_manager_tables", [False, True])
def test_non_dsa_rank_still_requires_a_flat_slot_source(has_manager_tables: bool) -> None:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=4, is_mla=False),
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
    )
    metadata = _dsa_metadata(dummy=False)
    metadata.multi_block_tables = (torch.ones((1, 1), dtype=torch.int32),) if has_manager_tables else ()
    metadata.new_cache_slots_host_values = []

    with pytest.raises(RuntimeError, match="one scheduler cache slot per token"):
        runner._effective_slot_mapping(metadata, 1, torch.device("cpu"))
    entry = SimpleNamespace(static_metadata=SimpleNamespace(new_cache_slots_host_values=[7]))
    with pytest.raises(RuntimeError, match="one host cache slot per token"):
        runner._fill_new_cache_slots_host_metadata(entry, metadata, 1)
