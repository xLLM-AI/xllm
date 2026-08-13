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

"""NPU (Ascend) ACL graph runner for the Python model executor.

Captures and replays decode-step graphs using ``torch.npu.NPUGraph``.
Mirrors the structure of ``decode_cuda_graph.py`` but adds NPU-specific
logic:

* ``torch.npu.graph_task_group_begin/end`` around FIA ``.out`` calls during
  capture.
* ``torch.npu.graph_task_update_begin/end`` to refresh FIA host params before
  replay.
* Static ``block_table`` and ``slot_mapping`` tensors so the graph records
  fixed addresses whose *contents* are updated via ``_fill_entry`` each step.
* C++ ACLNN ops (RMSNorm, SiLU, reshape_paged_cache) are used in both eager
  and capture modes — no PyTorch fallbacks needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadata,
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
    AclGraphExecutionState,
    AclGraphTask,
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner
from xllm.python.model_executor.runners.decode_cuda_graph import (
    _CAPTURE_WARMUP_STEPS,
    _decode_bucket,
)


@dataclass(slots=True)
class _StaticAttentionMetadata:
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
    dp_token_counts: tuple[int, ...] = ()
    q_seq_lens: torch.Tensor | None = None
    expanded_decode_metadata: ExpandedDecodeMetadata | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_input_embedding",
        "static_metadata",
        "kv_seq_lens_delta",
        "graph_tasks",
        "execution_state",
    )


_GraphKey = tuple[
    int,
    bool,
    torch.dtype | None,
    torch.device | None,
    tuple[int, ...] | None,
]


class DecodeAclGraphRunner(BaseRunner):
    """Decode graph runner for NPU (Ascend) using ACL graph capture/replay."""

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self.max_model_len = max_model_len
        self._graphs: dict[_GraphKey, _DecodeGraphEntry] = {}
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._stream: torch.npu.Stream | None = None
        self._update_stream: torch.npu.Stream | None = None
        self._replay_done_event: torch.npu.Event | None = None
        self._warmed_up = False

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> bool:
        if input_ids.dim() != 1:
            return False
        batch_size = input_ids.numel()
        bucket_size = _decode_bucket(batch_size)
        is_expanded_spec_verify = resolve_expanded_decode_metadata(metadata) is not None
        return (
            ((not metadata.is_prefill and not metadata.is_chunked_prefill) or is_expanded_spec_verify)
            and self._has_compatible_decode_metadata(input_ids, metadata)
            and (input_embedding is None or input_embedding.shape[0] == batch_size)
            and bucket_size <= self.max_batch
        )

    def _decode_metadata(
        self, metadata: AttentionMetadata
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        list[int] | None,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Return per-row KV and paging metadata for decode graph replay."""
        expanded = resolve_expanded_decode_metadata(metadata, block_size=self.attention_backend.page_size)
        block_table = expanded.block_table if expanded is not None else metadata.block_table
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        if block_table is None or kv_seq_lens is None:
            raise RuntimeError("decode graph requires block and KV metadata")
        block_table = block_table.to(torch.int32)
        kv_seq_lens = kv_seq_lens.to(torch.int32)
        is_mla = getattr(self.attention_backend, "is_mla", False)
        if kv_seq_lens_host_values is None and not is_mla:
            raise RuntimeError("decode graph requires scheduler-provided host KV lengths")

        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        if paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None:
            raise RuntimeError("decode graph requires paged KV metadata")
        self._validate_decode_metadata_shapes(
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        return (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    @staticmethod
    def _validate_decode_metadata_shapes(
        block_table: torch.Tensor,
        kv_seq_lens: torch.Tensor,
        kv_seq_lens_host_values: list[int] | None,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
    ) -> None:
        if block_table.dim() != 2:
            raise RuntimeError("decode block_table must be two-dimensional")
        sequence_count = block_table.shape[0]
        per_sequence_tensors = (
            ("kv_seq_lens", kv_seq_lens),
            ("paged_kv_last_page_len", paged_kv_last_page_len),
        )
        for name, tensor in per_sequence_tensors:
            if tensor.dim() != 1 or tensor.numel() != sequence_count:
                raise RuntimeError(f"decode {name} must contain one value per sequence")
        if kv_seq_lens_host_values is not None and len(kv_seq_lens_host_values) != sequence_count:
            raise RuntimeError("decode kv_seq_lens_host_values must contain one value per sequence")
        if paged_kv_indptr.dim() != 1 or paged_kv_indptr.numel() != sequence_count + 1:
            raise RuntimeError("decode paged_kv_indptr must contain one offset per sequence plus the terminal offset")
        if paged_kv_indices.dim() != 1 or paged_kv_indices.numel() == 0:
            raise RuntimeError("decode paged_kv_indices must be a non-empty flat page list")

    @staticmethod
    def _validate_decode_token_layout(
        input_ids: torch.Tensor,
        positions: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        sequence_count: int,
    ) -> None:
        if input_ids.dim() != 1 or input_ids.numel() != sequence_count:
            raise RuntimeError("ACL graph decode input_ids must contain one token per sequence")
        if slot_mapping.dim() != 1 or slot_mapping.numel() != sequence_count:
            raise RuntimeError("ACL graph decode slot_mapping must contain one slot per token")
        if positions is not None and (positions.dim() != 1 or positions.numel() != sequence_count):
            raise RuntimeError("ACL graph decode positions must contain one value per token")

    def _has_compatible_decode_metadata(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> bool:
        """Check the one-token-per-sequence contract of ACL decode graphs."""
        try:
            (
                block_table,
                _,
                _,
                _,
                _,
                _,
            ) = self._decode_metadata(metadata)
            self._validate_decode_token_layout(
                input_ids,
                None,
                metadata.slot_mapping,
                block_table.shape[0],
            )
        except (RuntimeError, ValueError):
            return False
        batch_size = input_ids.numel()
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        if not is_expanded and metadata.kv_cu_seq_lens is not None:
            if metadata.kv_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        if metadata.q_cu_seq_lens is not None and not is_expanded:
            if metadata.q_cu_seq_lens.numel() not in (
                batch_size,
                batch_size + 1,
            ):
                return False
        return True

    @staticmethod
    def _cumulative_lengths(
        sequence_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize NPU sequence ends to a cumulative tensor with a zero."""
        batch_size = sequence_lengths.numel()
        if cumulative_lengths is None:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=sequence_lengths.device,
                    ),
                    torch.cumsum(sequence_lengths, dim=0),
                )
            )
        cumulative_lengths = cumulative_lengths.to(torch.int32)
        if cumulative_lengths.numel() == batch_size + 1:
            return cumulative_lengths
        if cumulative_lengths.numel() == batch_size:
            return torch.cat(
                (
                    torch.zeros(
                        1,
                        dtype=torch.int32,
                        device=cumulative_lengths.device,
                    ),
                    cumulative_lengths,
                )
            )
        raise RuntimeError(
            "cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    def warmup(
        self,
        device: torch.device,
        _dtype: torch.dtype,
        input_embedding: torch.Tensor | None = None,
    ) -> None:
        if self._warmed_up:
            return
        self._warmed_up = True

        # MLA/SFA graph inputs include Lightning Indexer state and paged KV
        # metadata.  Capturing these graphs with dummy warmup data would make
        # the first real replay consume stale indexer/KV arguments.  Let the
        # first real request lazily capture its bucket instead.
        if getattr(self.attention_backend, "is_mla", False):
            return

        # TODO: Warmup capture with dummy data causes garbled
        # output on the first real request for dense FIA models.  The root
        # cause is likely related to 529d0b21's kv_cu_seq_lens cumulative
        # format change interacting with the kv_seq_lens_delta tensor that is
        # now aliased to static_metadata.kv_seq_lens during capture.  Disable
        # warmup for now and let the first real decode lazily capture its
        # graph bucket.  This trades slightly higher first-token latency for
        # correctness.
        return

        buckets = [size for size in (1, 2, 4, 8) if size <= self.max_batch]
        buckets.extend(range(16, self.max_batch + 1, 16))
        page_size = self.attention_backend.page_size
        for batch_size in buckets:
            slot_base = torch.arange(batch_size, dtype=torch.int32, device=device).mul_(page_size)
            block_ids = torch.arange(batch_size, dtype=torch.int32, device=device)
            metadata = _StaticAttentionMetadata(
                slot_mapping=slot_base,
                paged_kv_indptr=torch.arange(batch_size + 1, dtype=torch.int32, device=device),
                paged_kv_indices=block_ids,
                paged_kv_last_page_len=torch.ones(batch_size, dtype=torch.int32, device=device),
                kv_seq_lens_host=torch.full((batch_size,), 2, dtype=torch.int32, device="cpu"),
                kv_seq_lens_host_values=[2] * batch_size,
                # _decode_metadata (added by the expanded-decode path) reads the
                # device kv_seq_lens directly; the warmup must provide one so the
                # graph captures. The value is a dummy -- at replay the real
                # metadata's kv_seq_lens is copied over the static buffer.
                kv_seq_lens=torch.full((batch_size,), 2, dtype=torch.int32, device=device),
                kv_cu_seq_lens=torch.arange(batch_size + 1, dtype=torch.int32, device=device).mul_(2),
                block_table=block_ids.unsqueeze(1),
            )
            warmup_embedding = None
            if input_embedding is not None:
                warmup_embedding = torch.zeros(
                    batch_size,
                    input_embedding.shape[-1],
                    dtype=input_embedding.dtype,
                    device=device,
                )
            self.execute(
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                metadata,
                warmup_embedding,
            )

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        padded_batch_size = _decode_bucket(batch_size)
        if padded_batch_size > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        graph_key = self._graph_key(padded_batch_size, is_expanded, input_embedding)
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(padded_batch_size, input_ids, positions, metadata)
            self._graphs[graph_key] = entry

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._update_stream = torch.npu.Stream(device=input_ids.device, priority=-1)
            self._replay_done_event = torch.npu.Event()

        assert self._update_stream is not None
        assert self._replay_done_event is not None

        self._fill_entry(entry, input_ids, positions, metadata, batch_size, input_embedding)

        prepare_context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
        )
        with forward_context(prepare_context):
            self.attention_backend.prepare(entry.static_metadata, graph_mode=True)

        if first_capture:
            self._capture(entry)

        self._stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(self._stream):
            entry.graph.replay()
            output = entry.static_output[:batch_size].clone()

        # A captured FIA task waits on its update event before execution.  This
        # lets replay run concurrently with the host-side updates for later
        # layers while preventing the graph from observing stale parameters.
        with torch.npu.stream(self._update_stream):
            self._update_stream.wait_event(self._replay_done_event)
            self._update_graph_tasks(self._update_stream, entry.graph_tasks)

        # The next update must not overwrite task parameters until this replay
        # has consumed them.
        self._replay_done_event.record(self._stream)

        torch.npu.current_stream().wait_stream(self._stream)
        return output

    @staticmethod
    def _graph_key(
        padded_batch_size: int,
        is_expanded: bool,
        input_embedding: torch.Tensor | None,
    ) -> _GraphKey:
        """Return the key for a shape- and metadata-specific graph."""
        if input_embedding is None:
            return padded_batch_size, is_expanded, None, None, None
        return (
            padded_batch_size,
            is_expanded,
            input_embedding.dtype,
            input_embedding.device,
            tuple(input_embedding.shape[1:]),
        )

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        (
            _,
            _,
            _,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if self._paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (self.max_model_len + page_size - 1) // page_size
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * max_blocks_per_sequence,
                dtype=paged_kv_indices.dtype,
                device=device,
            )
            self._max_blocks_per_sequence = max_blocks_per_sequence

        static_block_table = torch.zeros(
            padded_batch_size,
            self._max_blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        entry = _DecodeGraphEntry()
        entry.batch_size = padded_batch_size
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.static_input_ids = torch.zeros(padded_batch_size, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(padded_batch_size, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=metadata.slot_mapping.dtype,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_indptr_host=torch.zeros(padded_batch_size + 1, dtype=torch.int32, device="cpu"),
            paged_kv_last_page_len_host=torch.ones(padded_batch_size, dtype=torch.int32, device="cpu"),
            kv_seq_lens_host_values=[1] * padded_batch_size,
            block_table=static_block_table,
        )
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        entry.kv_seq_lens_delta = torch.empty(padded_batch_size, dtype=torch.int32, device=device)
        # The graph metadata update writes per-sequence KV lengths into this
        # buffer.  MLA/SFA consumes the same stable buffer as its key lengths.
        entry.static_metadata.kv_seq_lens = entry.kv_seq_lens_delta
        if is_expanded:
            entry.static_metadata.expanded_decode_metadata = ExpandedDecodeMetadata(
                kv_seq_lens=entry.kv_seq_lens_delta,
                block_table=entry.static_metadata.block_table,
                paged_kv_indptr=entry.static_metadata.paged_kv_indptr,
                paged_kv_indices=entry.static_metadata.paged_kv_indices,
                paged_kv_last_page_len=(entry.static_metadata.paged_kv_last_page_len),
                paged_attention_tiling_data=None,
                kv_seq_lens_host=None,
                kv_seq_lens_host_values=(
                    None
                    if getattr(self.attention_backend, "is_mla", False)
                    else entry.static_metadata.kv_seq_lens_host_values
                ),
            )
        return entry

    def _fill_entry(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
        input_embedding: torch.Tensor | None,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        (
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        ) = self._decode_metadata(metadata)
        if batch_size != block_table.shape[0]:
            raise RuntimeError("ACL graph decode batch size must match metadata sequences")
        self._validate_decode_token_layout(
            input_ids,
            positions,
            metadata.slot_mapping,
            block_table.shape[0],
        )
        is_expanded = resolve_expanded_decode_metadata(metadata) is not None
        cumulative_kv_seq_lens = self._cumulative_lengths(
            kv_seq_lens,
            None if is_expanded else metadata.kv_cu_seq_lens,
        )
        graph_positions = positions.to(torch.int32).contiguous()
        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            metadata.slot_mapping,
            cumulative_kv_seq_lens,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            entry.static_input_ids,
            entry.static_positions,
            static_metadata.slot_mapping,
            static_metadata.kv_cu_seq_lens,
            entry.kv_seq_lens_delta,
            static_metadata.paged_kv_indptr,
            static_metadata.paged_kv_indices,
            static_metadata.paged_kv_last_page_len,
            padded_batch_size,
        )
        self._fill_host_metadata(entry, kv_seq_lens_host_values, batch_size)

        if input_embedding is not None:
            if input_embedding.shape[0] != batch_size:
                raise ValueError("ACL graph input_embedding token count must match input_ids")
            if entry.static_input_embedding is None:
                entry.static_input_embedding = torch.zeros(
                    entry.batch_size,
                    input_embedding.shape[-1],
                    dtype=input_embedding.dtype,
                    device=input_embedding.device,
                )
            elif entry.static_input_embedding.shape[1:] != input_embedding.shape[1:]:
                raise ValueError("ACL graph input_embedding shape changed for a graph bucket")
            entry.static_input_embedding[:batch_size].copy_(input_embedding)
            if entry.batch_size > batch_size:
                entry.static_input_embedding[batch_size:].zero_()
        elif entry.static_input_embedding is not None:
            entry.static_input_embedding.zero_()

        if static_metadata.block_table is not None:
            src_bt = block_table
            copy_cols = min(
                src_bt.shape[1],
                static_metadata.block_table.shape[1],
            )
            static_metadata.block_table[:batch_size, :copy_cols].copy_(src_bt[:batch_size, :copy_cols])
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()

        # Padded lanes must remain valid inputs for sparse MLA tiling.  Their
        # token and slot mapping are dummy values, so one KV token is safe.
        if padded_batch_size > batch_size:
            entry.kv_seq_lens_delta[batch_size:].fill_(1)

    def _fill_host_metadata(
        self,
        entry: _DecodeGraphEntry,
        kv_seq_lens: list[int] | None,
        batch_size: int,
    ) -> None:
        if getattr(self.attention_backend, "is_mla", False):
            return
        if kv_seq_lens is None:
            raise RuntimeError("decode ACL graph requires scheduler-provided host KV lengths")
        if len(kv_seq_lens) != batch_size:
            raise RuntimeError("decode ACL graph requires per-sequence host KV lengths")

        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_kv_seq_lens = static_metadata.kv_seq_lens_host_values
        if static_kv_seq_lens is None:
            raise RuntimeError("decode ACL graph host KV buffer is missing")
        static_kv_seq_lens[:batch_size] = kv_seq_lens
        if padded_batch_size > batch_size:
            static_kv_seq_lens[batch_size:] = [1] * (padded_batch_size - batch_size)

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            execution_state=entry.execution_state,
        )
        with forward_context(context):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self._forward_static(entry)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(self._stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            acl_graph=capture_context,
            execution_state=entry.execution_state,
        )
        with forward_context(context), torch.npu.graph(entry.graph, stream=self._stream):
            entry.static_output = self._forward_static(entry)
        entry.graph_tasks = capture_context.tasks

    def _forward_static(self, entry: _DecodeGraphEntry) -> torch.Tensor:
        if entry.static_input_embedding is None:
            return self.model(entry.static_input_ids, entry.static_positions)
        return self.model(
            entry.static_input_ids,
            entry.static_positions,
            entry.static_input_embedding,
        )

    @staticmethod
    def _update_graph_tasks(
        stream: torch.npu.Stream,
        graph_tasks: list[AclGraphTask],
    ) -> None:
        for task in graph_tasks:
            torch.npu.graph_task_update_begin(stream, task.handle)
            try:
                task.update()
            except Exception:
                torch.npu.graph_task_update_end(stream)
                raise
            torch.npu.graph_task_update_end(stream)
            task.event.record(stream)
