# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""CPU tests for SFA DCP prepared metadata and indexer paging."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

pytest.importorskip("torch_npu", reason="SFA DCP backend tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.sfa_dcp_backend import SfaDcpAttentionBackend
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)


class _FakeDcpGroup:
    def size(self) -> int:
        return 4

    def rank(self) -> int:
        return 0


def _cpu_context(execution_state: AclGraphExecutionState) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


@pytest.fixture(name="prepared_backend")
def _prepared_backend() -> SfaDcpAttentionBackend:
    backend = SfaDcpAttentionBackend(
        num_heads=8,
        num_kv_heads=1,
        head_dim=256,
        scale=0.1,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        dcp_group=_FakeDcpGroup(),
        index_topk=2048,
        max_num_reqs=8,
    )
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(16, 128, 1, 512),
                value=torch.empty(16, 128, 1, 64),
                index=torch.empty(64, 128, 1, 128),
            )
        ]
    )
    return backend


def _prepared_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        prepared_attention_state=None,
        slot_mapping=torch.tensor([127, 128, 511, 512], dtype=torch.int32),
        block_table=torch.tensor([[0, 1]] * 4, dtype=torch.int32),
        kv_seq_lens=torch.tensor([128, 129, 512, 513], dtype=torch.int32),
        kv_seq_lens_host_values=[128, 129, 512, 513],
        q_cu_seq_lens=torch.arange(1, 5, dtype=torch.int32),
        q_cu_seq_lens_host_values=[1, 2, 3, 4],
        q_seq_lens=torch.ones(4, dtype=torch.int32),
        expanded_decode_metadata=None,
        has_kv_shard=False,
        is_spec_verify=False,
        is_prefill=False,
        is_chunked_prefill=False,
    )


def test_prepare_metadata_preserves_active_dcp_forward(
    prepared_backend: SfaDcpAttentionBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = prepared_backend
    active = _prepared_metadata()
    active.prepared_attention_state = backend.prepare_metadata(active)
    backend.prepare(active)
    active_context = backend._sfa_metadata
    active_slots = backend._local_slot_mapping
    builder_lengths = backend._builder.dcp_local_seq_lens_buf.clone()
    candidate = _prepared_metadata()

    with monkeypatch.context() as patch:
        for method in ("localize_slots", "local_seq_lens", "expand_indexer_block_table"):
            patch.setattr(backend._kv_layout, method, MagicMock(side_effect=AssertionError("Device work in Prepare")))
        for method in ("to", "clone", "item", "tolist"):
            patch.setattr(torch.Tensor, method, MagicMock(side_effect=AssertionError("Device work in Prepare")))
        prepared = backend.prepare_metadata(candidate, device_kv_lengths=True)

    assert backend._metadata is active.prepared_attention_state
    assert backend._sfa_metadata is active_context
    assert backend._local_slot_mapping is active_slots
    assert torch.equal(backend._builder.dcp_local_seq_lens_buf, builder_lengths)
    assert prepared.slot_mapping is candidate.slot_mapping
    assert prepared.block_table is candidate.block_table
    assert prepared.actual_seq_kv is candidate.kv_seq_lens
    candidate.kv_seq_lens_host_values[0] = 999
    candidate.q_cu_seq_lens_host_values[0] = 999
    assert prepared.kv_lengths == [128, 129, 512, 513]
    assert prepared.query_ends == [1, 2, 3, 4]


@pytest.mark.parametrize("graph_mode", [False, True])
def test_prepared_dcp_uses_live_device_lengths_and_slots(
    prepared_backend: SfaDcpAttentionBackend, monkeypatch: pytest.MonkeyPatch, graph_mode: bool
) -> None:
    backend = prepared_backend
    metadata = _prepared_metadata()
    # Native MTP Prepare supplies an upper bound for every invocation, which
    # remains safe as Launch updates the Device lengths after acceptance.
    metadata.kv_seq_lens_host_values = [641] * 4
    metadata.prepared_attention_state = backend.prepare_metadata(metadata, device_kv_lengths=True)
    monkeypatch.setattr(backend._builder, "build", MagicMock(side_effect=AssertionError("shared builder used")))
    execution_state = AclGraphExecutionState({})
    with forward_context(_cpu_context(execution_state)):
        backend.prepare(metadata, graph_mode=graph_mode)
        first_context = backend._sfa_metadata.dcp_context
        assert first_context.seq_lens.tolist() == [128, 128, 128, 129]
        assert first_context.slot_mapping.tolist() == [127, -1, -1, 128]
        assert backend._mla_max_seqlen_k == (1024 if graph_mode else 641)

        # MTP advances Device tensors after Prepare; the Host snapshot stays put.
        metadata.kv_seq_lens.copy_(torch.tensor([512, 513, 640, 641], dtype=torch.int32))
        metadata.slot_mapping.copy_(torch.tensor([511, 512, 639, 640], dtype=torch.int32))
        backend.prepare(metadata, graph_mode=graph_mode)
        next_context = backend._sfa_metadata.dcp_context

    assert metadata.kv_seq_lens_host_values == [641] * 4
    assert next_context.seq_lens.tolist() == [128, 129, 256, 256]
    assert next_context.slot_mapping.tolist() == [-1, 128, 255, -1]
    assert backend._mla_actual_seq_kv is metadata.kv_seq_lens
    assert backend._mla_max_seqlen_k == (1024 if graph_mode else 641)
    if graph_mode:
        assert next_context.seq_lens.data_ptr() == first_context.seq_lens.data_ptr()
        assert next_context.slot_mapping.data_ptr() == first_context.slot_mapping.data_ptr()


def test_dcp_graph_replay_reuses_isolated_entry_buffers(
    prepared_backend: SfaDcpAttentionBackend, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = prepared_backend
    first = _prepared_metadata()
    second = _prepared_metadata()
    second.kv_seq_lens.fill_(1024)
    second.slot_mapping.fill_(-1)
    second.block_table.fill_(2)
    first.prepared_attention_state = backend.prepare_metadata(first)
    second.prepared_attention_state = backend.prepare_metadata(second)
    first_state = AclGraphExecutionState({})
    second_state = AclGraphExecutionState({})

    with forward_context(_cpu_context(first_state)):
        backend.prepare(first, graph_mode=True)
        first_context = backend._sfa_metadata.dcp_context
        first_indexer = backend._expanded_indexer_block_table
    with forward_context(_cpu_context(second_state)):
        backend.prepare(second, graph_mode=True)
        second_context = backend._sfa_metadata.dcp_context
        second_indexer = backend._expanded_indexer_block_table

    assert first_context.seq_lens.tolist() == [128, 128, 128, 129]
    assert second_context.seq_lens.tolist() == [256] * 4
    assert first_context.seq_lens.data_ptr() != second_context.seq_lens.data_ptr()
    assert first_context.slot_mapping.data_ptr() != second_context.slot_mapping.data_ptr()
    assert first_indexer.data_ptr() != second_indexer.data_ptr()

    first.kv_seq_lens.fill_(513)
    first.slot_mapping.fill_(512)
    first.block_table.fill_(3)
    with forward_context(_cpu_context(first_state)):
        backend.prepare(first, graph_mode=True)
        assert backend._sfa_metadata.dcp_context.seq_lens is first_context.seq_lens
        assert backend._expanded_indexer_block_table is first_indexer

    assert first_context.seq_lens.tolist() == [129] * 4
    assert first_context.slot_mapping.tolist() == [128] * 4
    assert first_indexer[0].tolist() == [12, 13, 14, 15] * 2
    assert second_context.seq_lens.tolist() == [256] * 4
    assert second_context.slot_mapping.tolist() == [-1] * 4
    assert second_indexer[0].tolist() == [8, 9, 10, 11] * 2

    first.kv_seq_lens.fill_(640)
    first.kv_seq_lens_host_values[:] = [640] * 4
    first.prepared_attention_state = backend.prepare_metadata(first)
    with monkeypatch.context() as patch:
        for method in ("localize_slots", "local_seq_lens", "expand_indexer_block_table"):
            patch.setattr(
                backend._kv_layout, method, MagicMock(side_effect=AssertionError("Device work before replay"))
            )
        for method in ("to", "clone", "copy_", "item", "tolist"):
            patch.setattr(torch.Tensor, method, MagicMock(side_effect=AssertionError("Device work before replay")))
        patch.setattr(torch, "empty_like", MagicMock(side_effect=AssertionError("new graph buffer")))
        for metadata, state, context, indexer in (
            (first, first_state, first_context, first_indexer),
            (second, second_state, second_context, second_indexer),
        ):
            with forward_context(_cpu_context(state)):
                backend.prepare_graph_replay(metadata)
                assert backend._metadata is metadata.prepared_attention_state
                assert backend._metadata.kv_lengths == metadata.kv_seq_lens_host_values
                assert backend._local_slot_mapping is context.slot_mapping
                assert backend._sfa_metadata.dcp_context.seq_lens is context.seq_lens
                assert backend._expanded_indexer_block_table is indexer
                assert backend._mla_actual_seq_kv is metadata.kv_seq_lens
                assert backend._mla_max_seqlen_k == 1024

    # The replay hook binds only. Captured operators will refresh the first
    # entry from its changed inputs when the actual graph runs.
    assert first_context.seq_lens.tolist() == [129] * 4
    assert second_context.seq_lens.tolist() == [256] * 4
    with (
        forward_context(_cpu_context(AclGraphExecutionState({}))),
        pytest.raises(RuntimeError, match="captured metadata buffers"),
    ):
        backend.prepare_graph_replay(first)
    assert backend._metadata is second.prepared_attention_state


def test_fused_dcp_preprocess_uses_local_slots_and_indexer_uses_logical_slots(
    prepared_backend: SfaDcpAttentionBackend,
) -> None:
    backend = prepared_backend
    metadata = _prepared_metadata()
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata)
    layer = SimpleNamespace(layer_id=0)
    context = backend.mla_preprocess_context(layer)
    assert context is not None
    assert context.slot_mapping.tolist() == [127, -1, -1, 128]
    assert context.kv_cache is backend._kv_caches[0].key

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        index_context = backend.mla_index_context(layer)
        index_cache, _, block_table = index_context.materialize_index_cache()

    assert index_context.slot_mapping is metadata.slot_mapping
    assert index_cache is backend._kv_caches[0].index
    assert block_table is index_context.block_table
    assert block_table[0].tolist() == list(range(8))


@pytest.mark.parametrize("prefill_field", ["is_prefill", "is_chunked_prefill"])
def test_prepared_prefill_keeps_compact_kv_gather(prepared_backend: SfaDcpAttentionBackend, prefill_field: str) -> None:
    backend = prepared_backend
    metadata = _prepared_metadata()
    setattr(metadata, prefill_field, True)
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata)

    assert backend._sfa_metadata.num_prefills == 4
    context = backend._sfa_metadata.dcp_context
    assert context.seq_lens.tolist() == [128, 128, 128, 129]
    assert context.kv_gather_block_ids.tolist() == [0, 1]
    assert context.kv_gather_block_table[0].tolist() == [0, 2, 4, 6, 1, 3, 5, 7]
    assert backend.mla_preprocess_context(SimpleNamespace(layer_id=0)) is None


@pytest.mark.parametrize("cache_is_preprocessed", [False, True])
def test_execute_mla_preserves_fused_cache_write(
    prepared_backend: SfaDcpAttentionBackend,
    monkeypatch: pytest.MonkeyPatch,
    cache_is_preprocessed: bool,
) -> None:
    backend = prepared_backend
    metadata = _prepared_metadata()
    metadata.prepared_attention_state = backend.prepare_metadata(metadata)
    backend.prepare(metadata)
    backend._impl = MagicMock()
    query = torch.empty(4, 8, 512)
    query_rope = torch.empty(4, 8, 64)
    key = None if cache_is_preprocessed else torch.empty(4, 1, 512)
    key_rope = None if cache_is_preprocessed else torch.empty(4, 1, 64)
    topk = torch.zeros(4, 1, 8, dtype=torch.int32)
    output = torch.empty_like(query)
    backend._impl._execute_sparse_flash_attention_process.return_value = output
    reshape = MagicMock()
    monkeypatch.setattr(torch.ops.xllm_ops, "reshape_paged_cache", reshape, raising=False)
    context = ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=backend._kv_caches,
    )
    with forward_context(context):
        actual = backend.execute_mla(
            query,
            query_rope,
            key,
            key_rope,
            SimpleNamespace(layer_id=0),
            topk=topk,
            cache_is_preprocessed=cache_is_preprocessed,
        )

    assert actual is output
    if cache_is_preprocessed:
        reshape.assert_not_called()
    else:
        reshape.assert_called_once()
        assert reshape.call_args.args[0] is backend._local_slot_mapping
    backend._impl._record_query_gather_context.assert_called_once()


def test_graph_prepare_keeps_valid_indexer_pages_for_padded_lanes(prepared_backend: SfaDcpAttentionBackend) -> None:
    backend = prepared_backend
    block_table = torch.zeros((8, 2), dtype=torch.int32)
    block_table[:7] = torch.tensor([[1, 2]] * 7, dtype=torch.int32)
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5, 6, -1], dtype=torch.int32)
    kv_seq_lens = torch.tensor([1022, 1022, 1022, 1022, 1022, 1022, 1022, 1], dtype=torch.int32)
    metadata = SimpleNamespace(
        slot_mapping=slot_mapping,
        block_table=block_table,
        kv_seq_lens=kv_seq_lens,
        kv_seq_lens_host_values=None,
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=None,
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    expanded = backend._expanded_indexer_block_table
    assert expanded is not None
    assert (expanded[-1] >= 0).all()
    assert torch.equal(expanded[-1], torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int32))
    assert torch.equal(expanded[0, :4], torch.tensor([4, 5, 6, 7], dtype=torch.int32))


@pytest.mark.parametrize("first_kv_len", [3, 511])
def test_prepare_uses_expanded_rows_for_mtp_verify(prepared_backend: SfaDcpAttentionBackend, first_kv_len: int) -> None:
    backend = prepared_backend
    captured: dict[str, object] = {}

    class _Builder:
        dcp_local_seq_lens_buf = torch.empty(8)

        @staticmethod
        def build(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(dcp_context=SimpleNamespace())

    backend._builder = _Builder()
    metadata = SimpleNamespace(
        slot_mapping=torch.arange(4, dtype=torch.int32),
        block_table=torch.tensor([[10, 11], [20, 21]], dtype=torch.int32),
        kv_seq_lens=torch.tensor([first_kv_len + 1, first_kv_len + 5], dtype=torch.int32),
        kv_seq_lens_host_values=[first_kv_len + 1, first_kv_len + 5],
        q_cu_seq_lens=None,
        q_seq_lens=None,
        expanded_decode_metadata=SimpleNamespace(
            enabled=True,
            kv_seq_lens=torch.tensor([first_kv_len + offset for offset in (0, 1, 4, 5)], dtype=torch.int32),
            block_table=torch.tensor([[10, 11], [10, 11], [20, 21], [20, 21]], dtype=torch.int32),
            paged_kv_indptr=torch.tensor([0, 1, 2, 3, 4] if first_kv_len == 3 else [0, 1, 2, 4, 6], dtype=torch.int32),
            paged_kv_indices=torch.tensor(
                [10, 10, 20, 20] if first_kv_len == 3 else [10, 10, 20, 21, 20, 21], dtype=torch.int32
            ),
            paged_kv_last_page_len=torch.tensor(
                [3, 4, 7, 8] if first_kv_len == 3 else [511, 512, 3, 4], dtype=torch.int32
            ),
            paged_attention_tiling_data=None,
            kv_seq_lens_host=None,
            kv_seq_lens_host_values=[first_kv_len + offset for offset in (0, 1, 4, 5)],
        ),
        is_prefill=False,
        is_chunked_prefill=False,
    )

    with forward_context(_cpu_context(AclGraphExecutionState({}))):
        backend.prepare(metadata, graph_mode=True)

    assert captured["num_reqs"] == 4
    assert captured["num_input_tokens"] == 4
    assert captured["seq_lens"].tolist() == [first_kv_len + offset for offset in (0, 1, 4, 5)]
    assert captured["block_table"].shape == (4, 2)
    assert backend._mla_max_seqlen_k == 1024
