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

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.csa_attention import DsaAttentionBackend
from xllm.python.attention.dsa_metadata import DsaMetadataBuilder, build_cache_specs
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry
from xllm.python.model_executor.runners.block_draft_acl_graph import (
    BlockDraftAclGraphRunner,
)
from xllm.python.model_executor.runners.decode_acl_graph import (
    DecodeAclGraphRunner,
)


def _runner(**kwargs) -> DecodeAclGraphRunner:
    attention_backend = SimpleNamespace(page_size=4, is_mla=False, create_graph_block_tables=lambda *_: ())
    return DecodeAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=8,
        **kwargs,
    )


def _metadata(linear_state_indices: torch.Tensor) -> SimpleNamespace:
    rows = linear_state_indices.numel()
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


def _block_draft_runner() -> BlockDraftAclGraphRunner:
    attention_backend = _FakeBlockDraftAttentionBackend()
    return BlockDraftAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=1,
        max_model_len=8,
    )


def _block_draft_metadata(block_table: torch.Tensor) -> SimpleNamespace:
    sequence_count, block_cols = block_table.shape
    query_width = 2
    token_count = sequence_count * query_width
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
        q_seq_lens=torch.full((sequence_count,), query_width, dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([0, token_count], dtype=torch.int32),
        q_cu_seq_lens_host_values=[token_count],
        q_seq_lens_host=None,
        block_table=block_table,
        slot_mapping=torch.arange(token_count, dtype=torch.int32),
        kv_seq_lens=torch.full((sequence_count,), block_cols * 4, dtype=torch.int32),
        kv_cu_seq_lens=None,
        kv_seq_lens_host_values=[block_cols * 4],
        paged_kv_indptr=torch.tensor([0, block_cols], dtype=torch.int32),
        paged_kv_indices=torch.arange(block_cols, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([4], dtype=torch.int32),
    )


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

    entry = runner._allocate_entry(input_ids, positions, metadata)

    assert entry.static_metadata.block_table.shape == (1, 32)
    assert entry.static_metadata.paged_kv_indices.numel() == 32
    assert entry.static_metadata.kv_seq_lens_host_values == [4096]


def test_dspark_warmup_captures_once_per_page_capacity_bucket() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()

    def prepare_entry(*args: object, graph_key: tuple[object, ...]) -> None:
        runner._graphs[graph_key] = object()

    with patch.object(runner, "_prepare_graph_entry", side_effect=prepare_entry) as prepare:
        keys = [
            runner.warmup(input_ids, positions, _block_draft_metadata(torch.zeros(1, width, dtype=torch.int32)))
            for width in (3, 4, 5, 3)
        ]
    assert prepare.call_count == 2
    assert keys[0] == keys[1] == keys[3]
    assert keys[0] != keys[2]


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


def test_dspark_graph_replay_refreshes_prepared_lengths() -> None:
    runner = _block_draft_runner()
    input_ids = torch.arange(2, dtype=torch.int32)
    positions = input_ids.clone()
    first_metadata = _block_draft_metadata(torch.tensor([[10, 11, 12]], dtype=torch.int32))
    entry = runner._allocate_entry(input_ids, positions, first_metadata)

    second_metadata = _block_draft_metadata(torch.tensor([[20]], dtype=torch.int32))
    second_metadata.kv_seq_lens.fill_(4)
    second_metadata.kv_seq_lens_host_values = [4]
    runner._fill_entry(entry, input_ids, positions, second_metadata)

    prepared = entry.static_metadata.prepared_attention_state
    assert prepared.actual_seq_q == [2]
    assert prepared.actual_seq_kv == [4]
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
    entry = runner._allocate_entry(input_ids, positions, first_metadata)

    runner._fill_entry(entry, input_ids, positions, first_metadata)
    assert entry.static_metadata.block_table.shape == (1, 4)
    assert entry.static_metadata.block_table[0].tolist() == [10, 11, 12, 0]

    second_metadata = _block_draft_metadata(
        torch.tensor([[20]], dtype=torch.int32),
    )
    runner._fill_entry(entry, input_ids, positions, second_metadata)
    assert entry.static_metadata.block_table[0].tolist() == [20, 0, 0, 0]
    assert entry.static_metadata.paged_kv_indices.tolist() == [0, 0, 0, 0]

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
        ("indices_capacity", RuntimeError, "paged KV indices exceed the captured capacity"),
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
    entry = runner._allocate_entry(input_ids, positions, metadata)
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
    if invalid == "indices_capacity":
        invalid_metadata.paged_kv_indices = torch.arange(5, dtype=torch.int32)
        invalid_metadata.paged_kv_indptr = torch.tensor([0, 5], dtype=torch.int32)
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


def test_linear_state_indices_use_stable_graph_buffer() -> None:
    runner = _runner()
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.tensor([3, 7, 11, 15], dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
        metadata=metadata,
    )
    static_indices = entry.static_metadata.linear_state_indices
    data_ptr = static_indices.data_ptr()

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


@pytest.mark.parametrize("dp_size", [1, 2])
def test_fill_entry_refreshes_mega_moe_mask_in_place(dp_size: int) -> None:
    runner = _runner(dp_size=dp_size, enable_mega_moe_token_mask=True)
    input_ids = torch.arange(4, dtype=torch.int32)
    positions = torch.arange(4, dtype=torch.int32)
    metadata = _metadata(torch.arange(4, dtype=torch.int32))
    entry = runner._allocate_entry(
        padded_batch_size=8,
        input_ids=input_ids,
        positions=positions,
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
            assert key == (*local_key[:-1], variants)
            keys.append(key)
        assert keys[0] != keys[2]
    assert [int(call.args[0].item()) for call in gather.call_args_list] == [1, 2, 1]


def test_mtp_graph_output_slices_and_detaches_replay_buffers() -> None:
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    topk = torch.arange(24, dtype=torch.int64).reshape(4, 2, 3)

    sliced = DecodeAclGraphRunner._slice_output((hidden, None, topk), 2)

    assert isinstance(sliced, tuple)
    sliced_hidden, sliced_aux, sliced_topk = sliced
    assert sliced_aux is None
    assert torch.equal(sliced_hidden, hidden[:2])
    assert torch.equal(sliced_topk, topk[:2])
    assert sliced_hidden.data_ptr() != hidden.data_ptr()
    assert sliced_topk.data_ptr() != topk.data_ptr()
    hidden.zero_()
    topk.zero_()
    assert torch.count_nonzero(sliced_hidden) > 0
    assert torch.count_nonzero(sliced_topk) > 0


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
    entry = runner._allocate_entry(8, input_ids, positions, metadata, topk)
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
    entry = runner._allocate_entry(8, input_ids, positions, metadata)

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
