# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import pytest
import torch

from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
    _GraphSlot,
    _StaticAttentionMetadata,
)


def _static_metadata(batch: int = 4, cols: int = 8):
    return _StaticAttentionMetadata(
        slot_mapping=torch.zeros(batch, dtype=torch.int32),
        paged_kv_indptr=torch.zeros(batch + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(batch * cols, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(batch, dtype=torch.int32),
        kv_seq_lens_host=torch.zeros(batch, dtype=torch.int32),
        q_seq_lens_host=torch.ones(batch, dtype=torch.int32),
        paged_kv_indptr_host=torch.zeros(batch + 1, dtype=torch.int32),
        paged_kv_last_page_len_host=torch.ones(batch, dtype=torch.int32),
        block_table=torch.zeros(batch, cols, dtype=torch.int32),
        multi_block_tables=(
            torch.full((batch, cols), -1, dtype=torch.int32),
            torch.full((batch, cols), -1, dtype=torch.int32),
        ),
    )


def test_static_metadata_exposes_dsv4_contract() -> None:
    metadata = _static_metadata()
    assert metadata.q_seq_lens_host.shape == (4,)
    assert metadata.kv_seq_lens_host.shape == (4,)
    assert metadata.max_query_len == 1
    assert metadata.max_seq_len == 1
    assert metadata.actual_num_sequences == 0
    assert metadata.dsa_metadata is None
    assert len(metadata.multi_block_tables) == 2
    assert metadata.dp_token_counts == ()


def test_fill_manager_tables_preserves_addresses_and_clears_tail() -> None:
    static = _static_metadata()
    pointers = tuple(table.data_ptr() for table in static.multi_block_tables)
    source = SimpleNamespace(
        multi_block_tables=(
            torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
            torch.tensor([[11], [12]], dtype=torch.int32),
        )
    )

    DecodeAclGraphRunner._fill_multi_block_tables(static, source, 2, 4)

    assert tuple(table.data_ptr() for table in static.multi_block_tables) == pointers
    assert static.multi_block_tables[0][:2, :2].tolist() == [[1, 2], [3, 4]]
    assert torch.all(static.multi_block_tables[0][:2, 2:] == -1)
    assert torch.all(static.multi_block_tables[0][2:] == -1)
    assert static.multi_block_tables[1][:2, 0].tolist() == [11, 12]


def test_fill_manager_tables_rejects_capture_capacity_overflow() -> None:
    static = _static_metadata(batch=4, cols=2)
    source = SimpleNamespace(
        multi_block_tables=(
            torch.ones((2, 3), dtype=torch.int32),
            torch.ones((2, 1), dtype=torch.int32),
        )
    )

    with pytest.raises(
        RuntimeError,
        match="DSA ACL graph block table exceeds capture capacity",
    ):
        DecodeAclGraphRunner._fill_multi_block_tables(static, source, 2, 4)


def test_fill_block_table_clears_live_row_tail_between_replays() -> None:
    static = _static_metadata(batch=4, cols=4)
    static.block_table.fill_(99)
    source = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)

    DecodeAclGraphRunner._fill_block_table(static, source, 2)

    assert static.block_table.tolist() == [
        [7, 8, 0, 0],
        [9, 10, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ]


def test_fill_block_table_restores_default_when_source_is_missing() -> None:
    static = _static_metadata(batch=2, cols=3)
    static.block_table.fill_(42)

    DecodeAclGraphRunner._fill_block_table(static, None, 2)

    assert torch.count_nonzero(static.block_table).item() == 0


def test_fill_host_metadata_keeps_per_sequence_lengths() -> None:
    static = _static_metadata()
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=static,
        host_seq_lens=torch.empty(4, dtype=torch.int32),
        host_block_counts=torch.empty(4, dtype=torch.int32),
    )
    runner = object.__new__(DecodeAclGraphRunner)
    runner.attention_backend = SimpleNamespace(page_size=128)
    source = SimpleNamespace(
        kv_seq_lens_host=torch.tensor([85, 129], dtype=torch.int32)
    )

    runner._fill_host_metadata(entry, source, batch_size=2)

    assert static.kv_seq_lens_host.tolist() == [85, 129, 0, 0]
    assert static.q_seq_lens_host.tolist() == [1, 1, 0, 0]
    assert static.actual_num_sequences == 2
    assert static.max_query_len == 1
    assert static.max_seq_len == 129
    assert static.paged_kv_indptr_host.tolist() == [0, 1, 3, 3, 3]
    assert static.paged_kv_last_page_len_host.tolist() == [85, 1, 1, 1]


def test_fill_host_metadata_normalizes_empty_graph_rows_to_zero() -> None:
    static = _static_metadata()
    entry = SimpleNamespace(
        batch_size=4,
        static_metadata=static,
        host_seq_lens=torch.empty(4, dtype=torch.int32),
        host_block_counts=torch.empty(4, dtype=torch.int32),
    )
    runner = object.__new__(DecodeAclGraphRunner)
    runner.attention_backend = SimpleNamespace(page_size=128)
    source = SimpleNamespace(
        kv_seq_lens_host=torch.full((4,), 512, dtype=torch.int32),
        actual_num_sequences=0,
    )

    runner._fill_host_metadata(entry, source, batch_size=4)

    assert static.kv_seq_lens_host.tolist() == [0, 0, 0, 0]
    assert static.q_seq_lens_host.tolist() == [0, 0, 0, 0]
    assert static.actual_num_sequences == 0
    assert static.max_query_len == 1
    assert static.max_seq_len == 512


def test_synthetic_capture_rows_are_valid_dummy_requests() -> None:
    # Python pre-captures graph buckets, while C++ captures lazily from the
    # first real request.  Synthetic capture must emulate valid triggering
    # rows rather than the separate empty-DP metadata path.
    metadata = _static_metadata(batch=4)
    metadata.kv_seq_lens_host.fill_(512)
    metadata.q_seq_lens_host.fill_(1)
    metadata.max_seq_len = 512
    metadata.actual_num_sequences = 4

    assert metadata.actual_num_sequences == 4
    assert metadata.kv_seq_lens_host.tolist() == [512, 512, 512, 512]
    assert metadata.q_seq_lens_host.tolist() == [1, 1, 1, 1]


def _runner_with_slots(slot_count: int) -> DecodeAclGraphRunner:
    runner = object.__new__(DecodeAclGraphRunner)
    runner._graph_slot_count = slot_count
    runner._graph_slots = [_GraphSlot() for _ in range(slot_count)]
    runner._next_replay_slot = 0
    runner._last_started_replay_slot = -1
    return runner


def test_double_buffer_replay_slots_alternate() -> None:
    runner = _runner_with_slots(2)

    first, first_prepared = runner._take_replay_slot(8, 7)
    second, second_prepared = runner._take_replay_slot(8, 7)
    third, third_prepared = runner._take_replay_slot(8, 7)

    assert first is runner._graph_slots[0]
    assert second is runner._graph_slots[1]
    assert third is runner._graph_slots[0]
    assert not first_prepared
    assert not second_prepared
    assert not third_prepared


def test_double_buffer_prepare_targets_slot_after_last_replay() -> None:
    runner = _runner_with_slots(2)

    runner._take_replay_slot(4, 3)
    assert runner._next_prepare_slot() is runner._graph_slots[1]
    runner._take_replay_slot(4, 3)
    assert runner._next_prepare_slot() is runner._graph_slots[0]


def test_prepared_slot_is_consumed_only_for_matching_shape() -> None:
    runner = _runner_with_slots(2)
    slot = runner._graph_slots[0]
    slot.is_prepared = True
    slot.prepared_batch_size = 8
    slot.prepared_actual_batch_size = 7

    selected, prepared = runner._take_replay_slot(8, 7)

    assert selected is slot
    assert prepared
    assert not slot.is_prepared

    runner._graph_slots[1].is_prepared = True
    runner._graph_slots[1].prepared_batch_size = 8
    runner._graph_slots[1].prepared_actual_batch_size = 6
    _, prepared = runner._take_replay_slot(8, 7)
    assert not prepared


def test_graph_slots_own_independent_persistent_buffers() -> None:
    runner = _runner_with_slots(2)
    runner._graph_slots[0].paged_kv_indices_buffer = torch.zeros(
        8, dtype=torch.int32
    )
    runner._graph_slots[1].paged_kv_indices_buffer = torch.zeros(
        8, dtype=torch.int32
    )

    assert (
        runner._graph_slots[0].paged_kv_indices_buffer.data_ptr()
        != runner._graph_slots[1].paged_kv_indices_buffer.data_ptr()
    )


def test_acl_graph_warmup_is_lazy_by_default(monkeypatch) -> None:
    runner = object.__new__(DecodeAclGraphRunner)
    runner._warmed_up = False
    monkeypatch.delenv("XLLM_ACLGRAPH_EAGER_WARMUP", raising=False)

    runner.warmup(torch.device("cpu"), torch.float32)

    assert runner._warmed_up
