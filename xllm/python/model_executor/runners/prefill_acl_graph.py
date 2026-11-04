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

"""Exact-geometry ACL graphs for Qwen3.5 multimodal and chunked prefill."""

from __future__ import annotations

from itertools import accumulate

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata, resolve_linear_state_io_indices
from xllm.python.layers.npu.qwen3_5.gdn_metadata import GdnPrefillMetadata
from xllm.python.model_executor.forward_context import AclGraphExecutionState, EplbRuntimeState, LayerSynchronizer
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.base import ModelExecutionOutput

_GraphKey = tuple[object, ...]


class PrefillAclGraphRunner(AclGraphRunner):
    """Capture a bounded set of exact packed-prefill geometries.

    Direct prefill FIA captures Host query and KV lengths, so a graph may only
    replay the same request lengths. Token IDs, visual embeddings, mRoPE
    positions, page IDs, and recurrent state indices remain dynamic tensors.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_seqs: int,
        max_tokens: int,
        max_variants: int,
    ) -> None:
        super().__init__(model, attention_backend, device)
        if min(max_seqs, max_tokens, max_variants) <= 0:
            raise ValueError("prefill ACL graph capacities must be positive")
        self._max_seqs = max_seqs
        self._max_tokens = max_tokens
        self._max_variants = max_variants
        self._graphs: dict[_GraphKey, AclGraphEntry] = {}
        self._stream: torch.npu.Stream | None = None
        self._last_replay_stream: torch.npu.Stream | None = None
        self._update_done_event: torch.npu.Event | None = None

    def can_execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        input_batch: InputBatch | None,
    ) -> bool:
        if input_batch is None or input_batch.is_dummy or metadata.is_spec_verify or metadata.is_mixed:
            return False
        if not (metadata.is_prefill or metadata.is_chunked_prefill):
            return False
        if metadata.prepared_attention_state is not None or metadata.has_kv_shard:
            return False
        token_count = input_ids.numel()
        if (
            input_ids.ndim != 1
            or token_count == 0
            or token_count > self._max_tokens
            or input_batch.num_reqs > self._max_seqs
            or input_batch.num_tokens != token_count
            or input_batch.query_start_loc[-1] != token_count
            or positions.shape not in ((token_count,), (3, token_count))
            or positions.device != input_ids.device
        ):
            return False
        if input_embedding is not None and (
            input_embedding.ndim != 2
            or input_embedding.shape[0] != token_count
            or input_embedding.device != input_ids.device
        ):
            return False
        block_table = metadata.block_table
        slot_mapping = metadata.slot_mapping
        query_ends = metadata.q_cu_seq_lens
        q_lengths = metadata.q_seq_lens_host
        kv_lengths = metadata.kv_seq_lens_host_values
        read_indices, write_indices = resolve_linear_state_io_indices(metadata)
        if (
            block_table is None
            or block_table.ndim != 2
            or block_table.shape[0] != input_batch.num_reqs
            or block_table.device != input_ids.device
            or slot_mapping is None
            or slot_mapping.shape != (token_count,)
            or slot_mapping.device != input_ids.device
            or query_ends is None
            or query_ends.ndim != 1
            or query_ends.numel() not in (input_batch.num_reqs, input_batch.num_reqs + 1)
            or query_ends.device != input_ids.device
            or q_lengths is None
            or q_lengths.device.type != "cpu"
            or q_lengths.ndim != 1
            or q_lengths.dtype not in (torch.int32, torch.int64)
            or kv_lengths is None
            or len(kv_lengths) != input_batch.num_reqs
            or metadata.has_initial_state is None
            or metadata.has_initial_state.device != input_ids.device
            or read_indices is None
            or write_indices is None
            or read_indices.shape != (input_batch.num_reqs,)
            or write_indices.shape != (input_batch.num_reqs,)
            or read_indices.device != input_ids.device
            or write_indices.device != input_ids.device
        ):
            return False
        query_lengths = tuple(int(value) for value in q_lengths.tolist())
        if query_lengths != input_batch.num_scheduled_tokens:
            return False
        if tuple(accumulate(query_lengths, initial=0)) != input_batch.query_start_loc:
            return False
        if any(length < query for length, query in zip(kv_lengths, query_lengths, strict=True)):
            return False
        if metadata.has_initial_state.shape != (input_batch.num_reqs,):
            return False
        key = self._graph_key(input_ids, positions, metadata, input_embedding, query_lengths, tuple(kv_lengths))
        return key in self._graphs or len(self._graphs) < self._max_variants

    @staticmethod
    def _graph_key(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        query_lengths: tuple[int, ...],
        kv_lengths: tuple[int, ...],
    ) -> _GraphKey:
        def tensor_signature(tensor: torch.Tensor | None) -> tuple[object, ...] | None:
            return None if tensor is None else (tuple(tensor.shape), tensor.dtype)

        embedding_signature = None if input_embedding is None else (input_embedding.dtype, input_embedding.shape[1])
        return (
            query_lengths,
            kv_lengths,
            metadata.block_table.shape[1],
            metadata.is_prefill,
            metadata.is_chunked_prefill,
            input_ids.dtype,
            positions.dtype,
            positions.ndim,
            metadata.slot_mapping.dtype,
            metadata.block_table.dtype,
            embedding_signature,
            tensor_signature(metadata.paged_kv_indptr),
            tensor_signature(metadata.paged_kv_indices),
            tensor_signature(metadata.paged_kv_last_page_len),
            tensor_signature(metadata.q_cu_seq_lens),
            tensor_signature(metadata.q_seq_lens),
            tensor_signature(metadata.kv_cu_seq_lens),
            tensor_signature(metadata.kv_seq_lens),
            tensor_signature(metadata.linear_state_indices),
            tensor_signature(metadata.linear_state_read_indices),
            tensor_signature(metadata.linear_state_write_indices),
            tensor_signature(metadata.has_initial_state),
        )

    @staticmethod
    def _copy_tensor(source: torch.Tensor | None) -> torch.Tensor | None:
        return None if source is None else source.clone()

    def _allocate_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
    ) -> AclGraphEntry:
        query_ends = list(accumulate(metadata.q_seq_lens_host.tolist(), initial=0))
        host_query_ends = metadata.q_cu_seq_lens_host_values
        if host_query_ends is None:
            host_query_ends = query_ends[1:] if metadata.q_cu_seq_lens.numel() == len(query_ends) - 1 else query_ends
        entry = AclGraphEntry()
        entry.batch_size = input_ids.numel()
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        entry.static_input_ids = torch.empty_like(input_ids)
        entry.static_positions = torch.empty_like(positions)
        entry.static_input_embedding = self._copy_tensor(input_embedding)
        entry.static_mtp_topk_indices = None
        entry.static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=torch.empty_like(metadata.slot_mapping),
            paged_kv_indptr=torch.empty_like(metadata.paged_kv_indptr),
            paged_kv_indices=torch.empty_like(metadata.paged_kv_indices),
            paged_kv_last_page_len=torch.empty_like(metadata.paged_kv_last_page_len),
            q_cu_seq_lens=self._copy_tensor(metadata.q_cu_seq_lens),
            q_cu_seq_lens_host_values=list(host_query_ends),
            q_seq_lens=self._copy_tensor(metadata.q_seq_lens),
            q_seq_lens_host=self._copy_tensor(metadata.q_seq_lens_host),
            kv_cu_seq_lens=self._copy_tensor(metadata.kv_cu_seq_lens),
            kv_seq_lens=self._copy_tensor(metadata.kv_seq_lens),
            kv_seq_lens_host=self._copy_tensor(metadata.kv_seq_lens_host),
            kv_seq_lens_host_values=list(metadata.kv_seq_lens_host_values),
            block_table=torch.empty_like(metadata.block_table),
            linear_state_indices=self._copy_tensor(metadata.linear_state_indices),
            linear_state_read_indices=self._copy_tensor(metadata.linear_state_read_indices),
            linear_state_write_indices=self._copy_tensor(metadata.linear_state_write_indices),
            has_initial_state=self._copy_tensor(metadata.has_initial_state),
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
        )
        return entry

    @staticmethod
    def _fill_entry(
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
    ) -> None:
        entry.static_input_ids.copy_(input_ids)
        entry.static_positions.copy_(positions)
        if input_embedding is not None:
            assert entry.static_input_embedding is not None
            entry.static_input_embedding.copy_(input_embedding)
        static = entry.static_metadata
        for field_name in (
            "slot_mapping",
            "paged_kv_indptr",
            "paged_kv_indices",
            "paged_kv_last_page_len",
            "q_cu_seq_lens",
            "q_seq_lens",
            "kv_cu_seq_lens",
            "kv_seq_lens",
            "block_table",
            "linear_state_indices",
            "linear_state_read_indices",
            "linear_state_write_indices",
            "has_initial_state",
        ):
            target = getattr(static, field_name)
            source = getattr(metadata, field_name)
            if target is not None and source is not None:
                target.copy_(source)

    def _snapshot_capture_state(self, entry: AclGraphEntry) -> tuple[object, ...]:
        full_snapshot = self.attention_backend.snapshot_full_cache_state([entry.static_metadata.slot_mapping])
        snapshots = [full_snapshot]
        for builder in self.execution_metadata_builders:
            persistent = entry.layer_shared_cache[builder.metadata_type]
            if isinstance(persistent, GdnPrefillMetadata):
                snapshots.append(builder.snapshot_prefill_graph_state(persistent))
        return tuple(snapshots)

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        if layer_synchronizer is not None or eplb is not None:
            raise ValueError("prefill ACL graph does not support layer synchronization or EPLB")
        if not self.can_execute(input_ids, positions, metadata, input_embedding, input_batch):
            raise ValueError("prefill ACL graph received an unsupported layout")
        assert input_batch is not None
        query_lengths = input_batch.num_scheduled_tokens
        key = self._graph_key(
            input_ids, positions, metadata, input_embedding, query_lengths, tuple(metadata.kv_seq_lens_host_values)
        )
        entry = self._graphs.get(key)
        first_capture = entry is None
        if entry is None:
            entry = self._allocate_entry(input_ids, positions, metadata, input_embedding)
        current_stream = torch.npu.current_stream()
        if self._last_replay_stream is not None:
            current_stream.wait_stream(self._last_replay_stream)
        self._fill_entry(entry, input_ids, positions, metadata, input_embedding)
        self._prepare_attention(entry, entry.static_metadata, replay=not first_capture)
        self._bind_graph_input_batch(entry, input_batch)
        if first_capture:
            if self._stream is None:
                self._stream = torch.npu.Stream(device=input_ids.device)
            snapshots = self._snapshot_capture_state(entry)

            def restore_capture_state() -> None:
                for snapshot in snapshots:
                    snapshot.restore()

            try:
                self._capture(entry, self._stream, reset_state=restore_capture_state)
            finally:
                current_stream.wait_stream(self._stream)
                restore_capture_state()
            self._prepare_attention(entry, entry.static_metadata, replay=True)
        replay_stream = self._stream if entry.graph_tasks else current_stream
        if entry.graph_tasks:
            assert replay_stream is not None
            replay_stream.wait_stream(current_stream)
        with torch.npu.stream(replay_stream):
            entry.graph.replay()
            output = self._slice_output(entry.static_output, input_ids.numel())
        if entry.graph_tasks:
            self._initialize_task_updates()
            if self._update_done_event is None:
                self._update_done_event = torch.npu.Event()
            self._update_after_replay(entry, replay_stream, self._update_done_event)
            current_stream.wait_event(self._update_done_event)
        if first_capture:
            self._graphs[key] = entry
        self._last_replay_stream = replay_stream
        if not entry.replay_logged:
            logger.info(
                "Python prefill ACL graph first replay: model=%s tokens=%d requests=%d chunked=%s",
                type(self.model).__name__,
                input_ids.numel(),
                input_batch.num_reqs,
                metadata.is_chunked_prefill,
            )
            entry.replay_logged = True
        return output
