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

"""CPU contract tests for dense Qwen DCP attention."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu", reason="Dense DCP backend tests import the NPU attention backend")

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dense_dcp_backend import DenseDcpAttentionBackend
from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.attention.dense_dcp_metadata_builder import DenseDcpMetadataBuilder
from xllm.python.distributed import AllGatherContext
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.input_batch import InputBatch


class _FakeDcpGroup:
    def __init__(self, rank: int = 1) -> None:
        self._rank = rank

    def size(self) -> int:
        return 2

    def rank(self) -> int:
        return self._rank


def _backend(rank: int = 1) -> DenseDcpAttentionBackend:
    backend = DenseDcpAttentionBackend(
        num_heads=2,
        num_kv_heads=1,
        head_dim=8,
        scale=0.25,
        sliding_window=0,
        device=torch.device("cpu"),
        dtype=torch.float16,
        dcp_group=_FakeDcpGroup(rank),
    )
    backend.bind_kv_caches(
        [
            LayerCache(
                key=torch.empty(8, 4, 1, 8),
                value=torch.empty(8, 4, 1, 8),
            )
        ]
    )
    return backend


def _metadata(
    kv_lengths: list[int],
    *,
    slot_mapping: torch.Tensor | None = None,
    local_slot_mapping: torch.Tensor | None = None,
    query_lengths: list[int] | None = None,
    is_prefill: bool = False,
    is_chunked_prefill: bool = False,
    is_mixed: bool = False,
) -> SimpleNamespace:
    num_reqs = len(kv_lengths)
    if slot_mapping is None:
        num_tokens = sum(query_lengths) if query_lengths is not None else num_reqs
        slot_mapping = torch.arange(num_tokens, dtype=torch.int32)
    q_cu_seq_lens = None
    if query_lengths is not None:
        q_cu_seq_lens = torch.tensor(
            [0, *torch.tensor(query_lengths, dtype=torch.int32).cumsum(0).tolist()],
            dtype=torch.int32,
        )
    return SimpleNamespace(
        slot_mapping=slot_mapping,
        local_slot_mapping=local_slot_mapping,
        kv_split_size=2,
        kv_split_rank=1,
        has_kv_shard=local_slot_mapping is not None,
        block_table=torch.zeros((num_reqs, 2), dtype=torch.int32),
        kv_seq_lens=torch.tensor(kv_lengths, dtype=torch.int32),
        kv_seq_lens_host=torch.tensor(kv_lengths, dtype=torch.int32),
        kv_seq_lens_host_values=kv_lengths,
        q_cu_host_values=None,
        q_cu_seq_lens=q_cu_seq_lens,
        q_seq_lens=(torch.tensor(query_lengths, dtype=torch.int32) if query_lengths is not None else None),
        expanded_decode_metadata=None,
        is_prefill=is_prefill,
        is_chunked_prefill=is_chunked_prefill,
        is_mixed=is_mixed,
        is_spec_verify=False,
    )


def _context(
    backend: DenseDcpAttentionBackend,
    metadata: SimpleNamespace,
    execution_state: AclGraphExecutionState | None = None,
) -> ForwardContext:
    builder = DenseDcpMetadataBuilder(2, backend._dcp_group.rank())
    builder.bind_layer_caches(backend._kv_caches)
    num_tokens = metadata.slot_mapping.numel()
    query_lengths = metadata.q_seq_lens.tolist() if metadata.q_seq_lens is not None else [1] * num_tokens
    batch = InputBatch.from_runtime(
        torch.zeros(num_tokens, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=len(query_lengths),
            num_tokens=num_tokens,
            num_scheduled_tokens=query_lengths,
            num_computed_tokens=[0] * len(query_lengths),
            query_start_loc=[0, *torch.tensor(query_lengths).cumsum(0).tolist()],
            is_prefilling=[metadata.is_prefill or metadata.is_chunked_prefill] * len(query_lengths),
        ),
    )
    layer_shared_cache = {}
    if metadata.local_slot_mapping is not None:
        if execution_state is None:
            dcp_metadata = builder.build(batch, metadata)
        else:
            dcp_metadata = builder.allocate_persistent(batch, metadata)
            builder.update_persistent(dcp_metadata, batch, metadata)
        layer_shared_cache[DenseDcpMetadata] = dcp_metadata
    return ForwardContext(
        attention_backend=backend,
        device=torch.device("cpu"),
        metadata=metadata,
        layer_caches=backend._kv_caches,
        execution_state=execution_state,
        layer_shared_cache=layer_shared_cache,
    )


def _query_gather(query: torch.Tensor) -> AllGatherContext:
    return AllGatherContext(gathered=query, handle=None, restore_perm=None)


@contextmanager
def _mock_dcp_exchange(result: torch.Tensor | None = None) -> Iterator[tuple[MagicMock, MagicMock, MagicMock]]:
    pack = patch("xllm.python.attention.dense_dcp_backend.pack_dcp_output_lse")
    combine = patch(
        "xllm.python.attention.dense_dcp_backend.fused_dcp_lse_combine",
        side_effect=lambda *args, **kwargs: (kwargs["output"] if result is None else kwargs["output"].copy_(result)),
    )
    exchange = patch("xllm.python.attention.dense_dcp_backend.dist.all_to_all_single")
    with pack as pack_mock, combine as combine_mock, exchange as exchange_mock:
        yield pack_mock, combine_mock, exchange_mock


def test_builder_computes_rank_local_kv_lengths_without_localizing_slots() -> None:
    backend = _backend(rank=1)
    local_slots = torch.tensor([-1, -1, 0, 3, -1], dtype=torch.int32)
    metadata = _metadata(
        [1, 4, 5, 8, 9],
        local_slot_mapping=local_slots,
    )

    with patch("xllm.python.attention.kv_shard_layout.KVShardLayout.localize_slots") as localize_slots:
        context = _context(backend, metadata)
        backend.prepare(metadata)

    assert context.layer_shared_cache[DenseDcpMetadata].local_kv_lengths == [0, 0, 1, 4, 4]
    localize_slots.assert_not_called()


def test_eager_prepare_does_not_require_forward_context() -> None:
    backend = _backend(rank=1)
    metadata = _metadata(
        [5],
        local_slot_mapping=torch.tensor([0], dtype=torch.int32),
    )

    backend.prepare(metadata)

    dcp_metadata = _context(backend, metadata).layer_shared_cache[DenseDcpMetadata]
    assert dcp_metadata.local_kv_lengths == [1]
    assert dcp_metadata.empty_kv_shards.tolist() == [[False]]


@pytest.mark.parametrize("graph_metadata", [False, True])
def test_execute_gathers_query_heads_and_writes_only_local_slots(graph_metadata: bool) -> None:
    backend = _backend(rank=1)
    local_slots = torch.tensor([-1, 0], dtype=torch.int32)
    metadata = _metadata([1, 5], local_slot_mapping=local_slots)
    query = torch.arange(2 * 2 * 8, dtype=torch.float16).view(2, -1)
    key = torch.ones(2, 8, dtype=torch.float16)
    value = torch.ones_like(key)
    gathered_query = torch.empty(2, 4, 8, dtype=torch.float16)
    context_output = torch.ones(2, 4, 8, dtype=torch.float16)
    context_lse = torch.zeros(2, 4, 1, dtype=torch.float32)
    merged = torch.arange(2 * 2 * 8, dtype=torch.float16).view(2, 2, 8) + 1
    gather_context = _query_gather(gathered_query)
    layer = SimpleNamespace(
        layer_id=0,
        causal=True,
        attention_window=None,
    )
    context = _context(backend, metadata)
    # The backend must use the builder's execution metadata, not this legacy field.
    if graph_metadata:
        metadata.local_slot_mapping = None
        metadata.has_kv_shard = False
        metadata.kv_split_size = 1
        metadata.kv_split_rank = 0
    else:
        metadata.local_slot_mapping = torch.full_like(local_slots, 99)

    with (
        forward_context(context),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather", return_value=gather_context),
        patch(
            "xllm.python.attention.dense_dcp_backend.finish_all_gather",
            return_value=gathered_query,
        ),
        patch(
            "xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache",
            create=True,
        ) as reshape_cache,
        patch.object(
            backend._fia,
            "run",
            return_value=(context_output, context_lse),
        ) as fia,
        _mock_dcp_exchange(merged),
    ):
        backend.prepare(metadata)
        output = backend.execute(query, key, value, layer)

    assert reshape_cache.call_args.args[0] is local_slots
    assert fia.call_args.args[0] is gathered_query
    assert tuple(fia.call_args.args[0].shape) == (2, 4, 8)
    torch.testing.assert_close(output, merged.reshape(2, 16))


def test_execute_requires_builder_metadata_before_collectives() -> None:
    backend = _backend()
    metadata = _metadata([5], local_slot_mapping=torch.tensor([0], dtype=torch.int32))
    context = _context(backend, metadata)
    context.layer_shared_cache.clear()
    backend.prepare(metadata)
    with (
        forward_context(context),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather") as gather,
        pytest.raises(RuntimeError, match="requires DenseDcpMetadata"),
    ):
        backend.execute(torch.empty(1, 16), torch.empty(1, 8), torch.empty(1, 8), SimpleNamespace(causal=True))
    gather.assert_not_called()


def test_chunked_prefill_merges_current_chunk_with_sharded_prefix() -> None:
    backend = _backend(rank=1)
    local_slots = torch.tensor([-1, -1, 0, 1, 2], dtype=torch.int32)
    metadata = _metadata(
        [2, 8],
        local_slot_mapping=local_slots,
        query_lengths=[2, 3],
        is_chunked_prefill=True,
    )
    query = torch.arange(5 * 2 * 8, dtype=torch.float16).view(5, -1)
    key = torch.ones(5, 8, dtype=torch.float16)
    value = torch.ones_like(key)
    gathered_query = torch.empty(5, 4, 8, dtype=torch.float16)
    current_output = torch.arange(5 * 4 * 8, dtype=torch.float16).view(5, 4, 8)
    current_lse = torch.arange(5 * 4, dtype=torch.float32).view(5, 4, 1)
    context_output = torch.ones_like(gathered_query)
    context_lse = torch.zeros(5, 4, 1, dtype=torch.float32)
    merged = torch.zeros(5, 2, 8, dtype=torch.float16)
    gather_context = _query_gather(gathered_query)
    layer = SimpleNamespace(
        layer_id=0,
        causal=True,
        attention_window=None,
    )

    context = _context(backend, metadata)
    with (
        forward_context(context),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather", return_value=gather_context),
        patch(
            "xllm.python.attention.dense_dcp_backend.finish_all_gather",
            return_value=gathered_query,
        ),
        patch(
            "xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache",
            create=True,
        ) as reshape_cache,
        patch.object(
            backend,
            "_run_current_chunk_attention",
            return_value=(current_output, current_lse),
        ),
        patch.object(
            backend._fia,
            "run",
            return_value=(context_output, context_lse),
        ) as context_attention,
        _mock_dcp_exchange(merged) as (_, combine, _),
    ):
        backend.prepare(metadata)
        output = backend.execute(query, key, value, layer)

    dcp_metadata = context.layer_shared_cache[DenseDcpMetadata]
    assert dcp_metadata.query_seq_ends == [2, 5]
    assert dcp_metadata.local_kv_lengths == [0, 1]
    assert dcp_metadata.empty_kv_shards.flatten().tolist() == [True, True, False, False, False]
    assert reshape_cache.call_args.args[0] is local_slots
    context_attention.assert_called_once()
    torch.testing.assert_close(combine.call_args.kwargs["local_output"], current_output[:, 2:4])
    torch.testing.assert_close(combine.call_args.kwargs["local_lse"], current_lse[:, 2:4, 0])
    torch.testing.assert_close(output, merged.reshape(5, 16))


def test_chunked_prefill_rejects_query_longer_than_kv_history() -> None:
    backend = _backend(rank=1)
    metadata = _metadata(
        [1],
        local_slot_mapping=torch.tensor([-1, -1], dtype=torch.int32),
        query_lengths=[2],
        is_chunked_prefill=True,
    )

    with pytest.raises(RuntimeError, match="query length exceeds"):
        _context(backend, metadata)


def test_mixed_batch_uses_chunked_prefill_metadata_contract() -> None:
    backend = _backend(rank=1)
    metadata = _metadata(
        [6, 9],
        local_slot_mapping=torch.tensor([-1, -1, 0], dtype=torch.int32),
        query_lengths=[1, 2],
        is_chunked_prefill=True,
        is_mixed=True,
    )

    dcp_metadata = _context(backend, metadata).layer_shared_cache[DenseDcpMetadata]
    assert dcp_metadata.query_seq_ends == [1, 3]
    assert dcp_metadata.local_kv_lengths == [1, 3]


def test_chunked_prefill_participates_when_prefix_is_remote_only() -> None:
    backend = _backend(rank=1)
    metadata = _metadata(
        [6],
        local_slot_mapping=torch.tensor([-1, -1], dtype=torch.int32),
        query_lengths=[2],
        is_chunked_prefill=True,
    )
    query = torch.zeros(2, 2 * 8, dtype=torch.float16)
    key = torch.zeros(2, 1, 8, dtype=torch.float16)
    value = torch.zeros_like(key)
    gathered_query = torch.empty(2, 4, 8, dtype=torch.float16)
    current_output = torch.arange(2 * 4 * 8, dtype=torch.float16).view(2, 4, 8)
    current_lse = torch.arange(2 * 4, dtype=torch.float32).view(2, 4, 1)
    merged = torch.zeros(2, 2, 8, dtype=torch.float16)
    gather_context = _query_gather(gathered_query)
    layer = SimpleNamespace(
        layer_id=0,
        causal=True,
        attention_window=None,
    )

    context = _context(backend, metadata)
    dcp_metadata = context.layer_shared_cache[DenseDcpMetadata]
    with (
        forward_context(context),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather", return_value=gather_context),
        patch(
            "xllm.python.attention.dense_dcp_backend.finish_all_gather",
            return_value=gathered_query,
        ),
        patch("xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache", create=True),
        patch.object(
            torch.ops.npu,
            "npu_fused_infer_attention_score",
            return_value=(current_output, current_lse),
        ) as current_attention,
        patch.object(backend._fia, "run") as context_attention,
        _mock_dcp_exchange(merged) as (pack, combine, _),
    ):
        backend.prepare(metadata)
        output = backend.execute(query, key, value, layer)

    assert current_attention.call_args.args[0] is gathered_query
    assert dcp_metadata.has_context
    assert dcp_metadata.local_kv_lengths == [0]
    context_attention.assert_not_called()
    assert torch.isneginf(pack.call_args.args[1]).all()
    torch.testing.assert_close(combine.call_args.kwargs["local_output"], current_output[:, 2:4])
    torch.testing.assert_close(combine.call_args.kwargs["local_lse"], current_lse[:, 2:4, 0])
    torch.testing.assert_close(output, merged.reshape(2, 16))


def test_decode_marks_empty_local_kv_shards_with_negative_infinity() -> None:
    backend = _backend(rank=1)
    metadata = _metadata([1, 5], local_slot_mapping=torch.tensor([-1, 0], dtype=torch.int32))
    query = torch.zeros(2, 2 * 8, dtype=torch.float16)
    key = torch.zeros(2, 1, 8, dtype=torch.float16)
    value = torch.zeros_like(key)
    gathered_query = torch.empty(2, 4, 8, dtype=torch.float16)
    partial_output = torch.ones(2, 4, 8, dtype=torch.float16)
    partial_lse = torch.zeros(2, 4, 1, dtype=torch.float32)
    merged = torch.arange(2 * 2 * 8, dtype=torch.float16).view(2, 2, 8)
    gather_context = _query_gather(gathered_query)
    layer = SimpleNamespace(
        layer_id=0,
        causal=True,
        attention_window=None,
    )

    context = _context(backend, metadata)
    with (
        forward_context(context),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather", return_value=gather_context),
        patch(
            "xllm.python.attention.dense_dcp_backend.finish_all_gather",
            return_value=gathered_query,
        ),
        patch("xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache", create=True),
        patch.object(
            backend._fia,
            "run",
            return_value=(partial_output, partial_lse),
        ),
        _mock_dcp_exchange(merged) as (pack, _, _),
    ):
        backend.prepare(metadata)
        output = backend.execute(query, key, value, layer)

    assert output.shape == (2, 16)
    merged_lse = pack.call_args.args[1]
    assert torch.isneginf(merged_lse[0]).all()
    assert torch.equal(merged_lse[1], torch.zeros(4))
    torch.testing.assert_close(output, merged.reshape(2, 16))


def test_prepare_graph_buffers_use_gathered_query_head_shape() -> None:
    backend = _backend(rank=1)
    metadata = _metadata(
        [5, 9],
        local_slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
    )
    execution_state = AclGraphExecutionState({})

    with (
        forward_context(_context(backend, metadata, execution_state)),
        patch.object(
            torch_npu,
            "_npu_fused_infer_attention_score_get_max_workspace",
            return_value=torch.empty(16, dtype=torch.uint8),
        ),
    ):
        backend.prepare(metadata, graph_mode=True)

    buffers = backend._fia._buffers
    assert buffers is not None
    assert tuple(buffers.output.shape) == (2, 4, 8)
    assert tuple(buffers.lse.shape) == (2, 4, 1)
    assert buffers.lse.dtype == torch.float32


def test_graph_prepare_reuses_buffers_and_queries_workspace_only_once() -> None:
    backend = _backend()
    metadata = _metadata([5, 9], local_slot_mapping=torch.tensor([0, 1], dtype=torch.int32))
    context = _context(backend, metadata, AclGraphExecutionState({}))
    with (
        forward_context(context),
        patch.object(
            torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", return_value=torch.empty(16)
        ) as workspace,
    ):
        backend.prepare(metadata, graph_mode=True)
        first = backend._fia._buffers
        metadata.kv_seq_lens_host = torch.tensor([8, 13])
        with patch("torch.empty", wraps=torch.empty) as allocate:
            backend.prepare(metadata, graph_mode=True)
        allocate.assert_not_called()
        second = backend._fia._buffers
    workspace.assert_called_once()
    assert first is not None and second is not None
    assert first.output is second.output
    assert first.lse is second.lse
    assert first.workspace is second.workspace
    assert first.block_table is metadata.block_table
    assert workspace.call_args.args[1].data_ptr() == backend._kv_caches[0].key.data_ptr()


@pytest.mark.parametrize(("second_bucket", "share_workspace"), [(1, False), (2, False), (2, True)])
def test_graph_tasks_keep_entry_resources_and_layer_operands_after_switching_buckets(
    second_bucket: int, share_workspace: bool
) -> None:
    backend = _backend()
    backend.bind_kv_caches([LayerCache(torch.empty(8, 4, 1, 8), torch.empty(8, 4, 1, 8)) for _ in range(2)])
    metadata_a = _metadata([5, 9], local_slot_mapping=torch.tensor([0, 1], dtype=torch.int32))
    metadata_b = _metadata([13] * second_bucket, local_slot_mapping=torch.zeros(second_bucket, dtype=torch.int32))
    workspace_pool = [] if share_workspace else None
    context_a = _context(backend, metadata_a, AclGraphExecutionState({}, serial_workspace_pool=workspace_pool))
    context_b = _context(backend, metadata_b, AclGraphExecutionState({}, serial_workspace_pool=workspace_pool))
    capture_a = AclGraphCaptureContext(object(), [])
    capture_b = AclGraphCaptureContext(object(), [])
    queries_a = [torch.empty(2, 4, 8) for _ in range(2)]
    query_b = torch.empty(second_bucket, 4, 8)
    calls = []

    def record_fia(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, **kwargs: object) -> None:
        calls.append((query, key, value, {**kwargs, "actual_seq_lengths_kv": list(kwargs["actual_seq_lengths_kv"])}))

    with (
        patch.object(
            torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", side_effect=lambda *a, **k: torch.empty(16)
        ),
        patch.object(torch.ops.npu.npu_fused_infer_attention_score, "out", side_effect=record_fia),
        patch.object(torch.npu, "ExternalEvent", side_effect=lambda: MagicMock()),
        patch.object(torch.npu, "graph_task_group_begin"),
        patch.object(torch.npu, "graph_task_group_end", side_effect=lambda _: object()),
    ):
        with forward_context(replace(context_a, acl_graph=capture_a)):
            backend.prepare(metadata_a, graph_mode=True)
            module_a = context_a.layer_shared_cache[DenseDcpMetadata]
            outputs_a = [
                backend._fia.run(query, cache.key, cache.value, module_a)
                for query, cache in zip(queries_a, backend._kv_caches, strict=True)
            ]
        with forward_context(replace(context_b, acl_graph=capture_b)):
            backend.prepare(metadata_b, graph_mode=True)
            module_b = context_b.layer_shared_cache[DenseDcpMetadata]
            output_b, _ = backend._fia.run(query_b, backend._kv_caches[0].key, backend._kv_caches[0].value, module_b)
        # Replay callbacks run without A's ForwardContext, after B changed the
        # adapter's current selection. They must still bind A and its latest lengths.
        module_a.local_kv_lengths[:] = [4, 7]
        calls.clear()
        for task in capture_a.tasks:
            task.update()
        capture_b.tasks[0].update()

    assert len(capture_a.tasks) == 2
    assert capture_a.tasks[0].event is not capture_a.tasks[1].event
    assert capture_a.tasks[0].handle is not capture_a.tasks[1].handle
    assert outputs_a[0][0] is outputs_a[1][0]
    assert outputs_a[0][1] is outputs_a[1][1]
    assert output_b.data_ptr() != outputs_a[0][0].data_ptr()
    assert (calls[0][3]["workspace"].data_ptr() == calls[2][3]["workspace"].data_ptr()) is share_workspace
    for index, (query, key, value, kwargs) in enumerate(calls[:2]):
        assert query is queries_a[index]
        assert key.data_ptr() == backend._kv_caches[index].key.data_ptr()
        assert value.data_ptr() == backend._kv_caches[index].value.data_ptr()
        assert kwargs["actual_seq_lengths_kv"] == [4, 7]
        assert kwargs["out"][0] is outputs_a[0][0]
        assert kwargs["block_table"] is metadata_a.block_table
    assert calls[2][0] is query_b
    assert calls[2][3]["actual_seq_lengths_kv"] == [5] * second_bucket
    assert calls[2][3]["out"][0] is output_b


@pytest.mark.parametrize("with_table", [False, True])
def test_pure_prefill_uses_current_kv_even_if_upstream_supplies_a_block_table(with_table: bool) -> None:
    backend = _backend()
    metadata = _metadata(
        [6], query_lengths=[6], is_prefill=True, local_slot_mapping=torch.tensor([-1, -1, -1, -1, 0, 1])
    )
    if not with_table:
        metadata.block_table = None
    query = torch.ones(6, 16)
    key = torch.ones(6, 8)
    value = torch.ones_like(key)
    expected = torch.ones(6, 2, 8)
    with (
        forward_context(_context(backend, metadata)),
        patch("xllm.python.attention.dense_dcp_backend.start_all_gather") as gather,
        patch("xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache", create=True),
        patch.object(torch.ops.npu, "npu_fused_infer_attention_score", return_value=(expected, None)) as fia,
    ):
        backend.prepare(metadata)
        result = backend.execute(query, key, value, SimpleNamespace(causal=True, layer_id=0))
    gather.assert_not_called()
    assert fia.call_args.args[1].data_ptr() == key.data_ptr()
    assert "block_table" not in fia.call_args.kwargs
    assert fia.call_args.kwargs["actual_seq_lengths_kv"] == [6]
    assert fia.call_args.kwargs["num_heads"] == 2
    torch.testing.assert_close(result, expected.reshape(6, 16))


def test_graph_capture_failure_closes_task_group_without_registering_a_task() -> None:
    backend = _backend()
    metadata = _metadata([5], local_slot_mapping=torch.tensor([0], dtype=torch.int32))
    context = _context(backend, metadata, AclGraphExecutionState({}))
    capture = AclGraphCaptureContext(object(), [])
    with (
        forward_context(replace(context, acl_graph=capture)),
        patch.object(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", return_value=torch.empty(16)),
        patch.object(torch.ops.npu.npu_fused_infer_attention_score, "out", side_effect=RuntimeError("FIA failed")),
        patch.object(torch.npu, "ExternalEvent"),
        patch.object(torch.npu, "graph_task_group_begin") as begin,
        patch.object(torch.npu, "graph_task_group_end") as end,
    ):
        backend.prepare(metadata, graph_mode=True)
        with pytest.raises(RuntimeError, match="FIA failed"):
            backend._fia.run(
                torch.empty(1, 4, 8),
                backend._kv_caches[0].key,
                backend._kv_caches[0].value,
                context.layer_shared_cache[DenseDcpMetadata],
            )
    begin.assert_called_once_with(capture.stream)
    end.assert_called_once_with(capture.stream)
    assert capture.tasks == []


def test_graph_prepare_rejects_replacing_an_entry_block_table() -> None:
    backend = _backend()
    metadata = _metadata([5], local_slot_mapping=torch.tensor([0], dtype=torch.int32))
    context = _context(backend, metadata, AclGraphExecutionState({}))
    with (
        forward_context(context),
        patch.object(torch_npu, "_npu_fused_infer_attention_score_get_max_workspace", return_value=torch.empty(16)),
    ):
        backend.prepare(metadata, graph_mode=True)
        metadata.block_table = metadata.block_table.clone()
        with pytest.raises(RuntimeError, match="address changed"):
            backend.prepare(metadata, graph_mode=True)


def test_merge_scratch_is_shared_within_an_entry_but_not_between_entries() -> None:
    backend = _backend()
    metadata = _metadata([5], local_slot_mapping=torch.tensor([0], dtype=torch.int32))
    context_a = _context(backend, metadata, AclGraphExecutionState({}))
    context_b = _context(backend, metadata, AclGraphExecutionState({}))
    backend._fia = MagicMock()
    backend._fia.run.return_value = (
        torch.ones(1, 4, 8, dtype=torch.float16),
        torch.zeros(1, 4, dtype=torch.float32),
    )
    query = torch.zeros(1, 2 * 8, dtype=torch.float16)
    key = torch.zeros(1, 1, 8, dtype=torch.float16)
    value = torch.zeros_like(key)
    gathered_query = torch.empty(1, 4, 8, dtype=torch.float16)
    gather_context = _query_gather(gathered_query)
    layer = SimpleNamespace(layer_id=0, causal=True, attention_window=None)

    def run(context: ForwardContext) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with (
            forward_context(context),
            patch("xllm.python.attention.dense_dcp_backend.start_all_gather", return_value=gather_context),
            patch(
                "xllm.python.attention.dense_dcp_backend.finish_all_gather",
                return_value=gathered_query,
            ),
            patch("xllm.python.attention.dense_dcp_backend.kernels.reshape_paged_cache", create=True),
            _mock_dcp_exchange() as (pack, _, exchange),
        ):
            output = backend.execute(query, key, value, layer)
        return output, pack.call_args.kwargs["send"], exchange.call_args.args[0]

    first_output, first_send, first_recv = run(context_a)
    second_output, second_send, second_recv = run(context_a)
    other_output, other_send, other_recv = run(context_b)

    assert first_output.shape == (1, 16)
    assert first_output.data_ptr() == second_output.data_ptr()
    assert first_send.data_ptr() == second_send.data_ptr()
    assert first_recv.data_ptr() == second_recv.data_ptr()
    assert first_output.data_ptr() != other_output.data_ptr()
    assert first_send.data_ptr() != other_send.data_ptr()
    assert first_recv.data_ptr() != other_recv.data_ptr()
    assert first_send.shape == first_recv.shape
    assert first_send.shape[0] == 2
