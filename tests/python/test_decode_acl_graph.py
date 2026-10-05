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


def _block_draft_runner() -> BlockDraftAclGraphRunner:
    attention_backend = SimpleNamespace(
        page_size=4,
        is_mla=False,
        prepare_metadata=lambda metadata, **_: metadata,
    )
    return BlockDraftAclGraphRunner(
        nn.Identity(),
        attention_backend,
        torch.device("cpu"),
        max_batch=8,
        max_model_len=16,
    )


def _block_draft_metadata(block_cols: int) -> SimpleNamespace:
    return SimpleNamespace(
        q_seq_lens=torch.tensor([7], dtype=torch.int32),
        q_cu_seq_lens=torch.tensor([7], dtype=torch.int32),
        q_cu_seq_lens_host_values=[7],
        q_seq_lens_host=None,
        kv_cu_seq_lens=None,
        slot_mapping=torch.arange(7, dtype=torch.int32),
        block_table=torch.arange(block_cols, dtype=torch.int32).view(1, block_cols),
        kv_seq_lens=torch.tensor([block_cols * 4], dtype=torch.int32),
        kv_seq_lens_host_values=[block_cols * 4],
        paged_kv_indptr=torch.tensor([0, block_cols], dtype=torch.int32),
        paged_kv_indices=torch.arange(block_cols, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([4], dtype=torch.int32),
        is_prefill=False,
        is_chunked_prefill=True,
        is_spec_verify=False,
    )


def test_block_draft_graph_key_ignores_live_page_table_width() -> None:
    runner = _block_draft_runner()
    input_ids = torch.zeros(7, dtype=torch.int32)
    narrow = _block_draft_metadata(block_cols=1)
    wide = _block_draft_metadata(block_cols=2)

    narrow_key = runner._graph_key(input_ids, narrow.q_seq_lens.numel(), int(narrow.q_seq_lens[0]))
    wide_key = runner._graph_key(input_ids, wide.q_seq_lens.numel(), int(wide.q_seq_lens[0]))
    assert narrow_key == wide_key


def test_block_draft_graph_uses_stable_page_table_capacity() -> None:
    runner = _block_draft_runner()
    input_ids = torch.zeros(7, dtype=torch.int32)
    positions = torch.arange(7, dtype=torch.int32)
    metadata = _block_draft_metadata(block_cols=1)
    entry = runner._allocate_entry(input_ids, positions, metadata)

    assert entry.static_metadata.block_table.shape == (1, 5)
    assert entry.static_metadata.paged_kv_indices.numel() == 5

    metadata.block_table = torch.tensor([[10]], dtype=torch.int32)
    runner._fill_entry(entry, input_ids, positions, metadata)
    assert entry.static_metadata.block_table.tolist() == [[10, 0, 0, 0, 0]]

    metadata = _block_draft_metadata(block_cols=2)
    metadata.block_table = torch.tensor([[11, 12]], dtype=torch.int32)
    runner._fill_entry(entry, input_ids, positions, metadata)
    assert entry.static_metadata.block_table.tolist() == [[11, 12, 0, 0, 0]]


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
