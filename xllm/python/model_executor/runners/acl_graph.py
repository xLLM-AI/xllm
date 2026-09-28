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

"""Capture and task-update mechanics shared by the two ACL input owners."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.attention.expanded_decode_metadata import ExpandedDecodeMetadata
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphTask,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import _CAPTURE_WARMUP_STEPS


@dataclass(slots=True)
class StaticGraphAttentionMetadata:
    slot_mapping: torch.Tensor
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor
    qo_indptr: torch.Tensor | None = None
    q_cu_seq_lens: torch.Tensor | None = None
    kv_cu_seq_lens: torch.Tensor | None = None
    kv_seq_lens_host: torch.Tensor | None = None
    kv_seq_lens_host_values: list[int] | None = None
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    linear_state_indices: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
    dp_execution_token_counts: tuple[int, ...] = ()
    dp_global_sequence_nums: tuple[int, ...] = ()
    dp_is_decode: tuple[int, ...] = ()
    q_seq_lens: torch.Tensor | None = None
    expanded_decode_metadata: ExpandedDecodeMetadata | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False
    is_mixed: bool = False
    is_spec_verify: bool = False
    local_slot_mapping: torch.Tensor | None = None
    kv_split_size: int = 1
    kv_split_rank: int = 0
    has_kv_shard: bool = False
    prepared_attention_state: object | None = None


class AclGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_mtp_topk_indices",
        "static_metadata",
        "kv_seq_lens_delta",
        "graph_tasks",
        "execution_state",
        "replay_logged",
    )


class AclGraphRunner(BaseRunner):
    """Shared device operations; subclasses own input storage and cache policy."""

    def __init__(self, model: nn.Module, attention_backend: AttentionBackend, device: torch.device) -> None:
        super().__init__(model, attention_backend, device)
        self._update_stream: torch.npu.Stream | None = None
        self._replay_done_event: torch.npu.Event | None = None

    def _initialize_task_updates(self) -> None:
        if self._update_stream is None:
            self._update_stream = torch.npu.Stream(device=self.device, priority=-1)
            self._replay_done_event = torch.npu.Event()

    def _prepare_attention(self, entry: AclGraphEntry, metadata: AttentionMetadata, *, replay: bool = False) -> None:
        context = ForwardContext(
            self.attention_backend,
            self.device,
            metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
        )
        with forward_context(context):
            if replay:
                self.attention_backend.prepare_graph_replay(metadata)
            else:
                self.attention_backend.prepare(metadata, graph_mode=True)

    def _update_after_replay(
        self,
        entry: AclGraphEntry,
        stream: torch.npu.Stream,
        completion_event: torch.npu.Event | None = None,
    ) -> None:
        assert self._update_stream is not None
        assert self._replay_done_event is not None
        with torch.npu.stream(self._update_stream):
            self._update_stream.wait_event(self._replay_done_event)
            self._update_graph_tasks(self._update_stream, entry.graph_tasks)
            if completion_event is not None:
                completion_event.record(self._update_stream)
        self._replay_done_event.record(stream)

    @staticmethod
    def _validate_decode_token_layout(
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        metadata_row_count: int,
    ) -> None:
        if input_ids.dim() != 1 or input_ids.numel() != metadata_row_count:
            raise RuntimeError("ACL graph decode input_ids must contain one token per metadata row")
        if slot_mapping.dim() != 1 or slot_mapping.numel() != metadata_row_count:
            raise RuntimeError("ACL graph decode slot_mapping must contain one slot per token")
        if positions is not None and (positions.dim() != 1 or positions.numel() != metadata_row_count):
            raise RuntimeError("ACL graph decode positions must contain one value per token")

    def _capture(
        self,
        entry: AclGraphEntry,
        stream: torch.npu.Stream,
    ) -> None:
        assert stream is not None
        logger.info(
            "Python ACL graph capture start: model=%s bucket=%d mtp_topk=%s",
            type(self.model).__name__,
            entry.batch_size,
            entry.static_mtp_topk_indices is not None,
        )
        # Input copies and backend metadata preparation run on the caller's
        # stream. A new bucket's eager warmup must observe those writes before
        # reading token IDs or page tables, just like an existing graph replay.
        stream.wait_stream(torch.npu.current_stream())
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
        )
        with forward_context(context), torch.npu.stream(stream):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self._forward_static(entry)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            acl_graph=capture_context,
            execution_state=entry.execution_state,
        )
        with forward_context(context), torch.npu.graph(entry.graph, stream=stream):
            entry.static_output = self._forward_static(entry)
        entry.graph_tasks = capture_context.tasks
        logger.info(
            "Python ACL graph captured: model=%s bucket=%d mtp_topk=%s tasks=%d",
            type(self.model).__name__,
            entry.batch_size,
            entry.static_mtp_topk_indices is not None,
            len(entry.graph_tasks),
        )

    def _forward_static(self, entry: AclGraphEntry) -> ModelExecutionOutput:
        if entry.static_metadata.prepared_attention_state is not None:
            # Quant-indexer metadata must be computed inside capture against
            # the Slot length tensors, not reused from capture warmup.
            self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
        if entry.static_input_embedding is None:
            if entry.static_mtp_topk_indices is None:
                return self.model(entry.static_input_ids, entry.static_positions)
            return self.model(
                entry.static_input_ids,
                entry.static_positions,
                None,
                entry.static_mtp_topk_indices,
            )
        if entry.static_mtp_topk_indices is None:
            return self.model(
                entry.static_input_ids,
                entry.static_positions,
                entry.static_input_embedding,
            )
        return self.model(
            entry.static_input_ids,
            entry.static_positions,
            entry.static_input_embedding,
            entry.static_mtp_topk_indices,
        )

    @staticmethod
    def _update_graph_tasks(
        stream: torch.npu.Stream,
        graph_tasks: list[AclGraphTask],
    ) -> None:
        for task in graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            task.update()
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)
