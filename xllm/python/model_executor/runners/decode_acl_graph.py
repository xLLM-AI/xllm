# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
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

from xllm.python import ops
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.forward_context import (
    AclGraphCaptureContext,
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
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    is_prefill: bool = False
    is_chunked_prefill: bool = False


class _DecodeGraphEntry:
    __slots__ = (
        "batch_size",
        "graph",
        "static_output",
        "static_input_ids",
        "static_positions",
        "static_metadata",
        "kv_seq_lens_delta",
        "host_seq_lens",
        "host_block_counts",
        "graph_tasks",
    )


class DecodeAclGraphRunner(BaseRunner):
    """Decode graph runner for NPU (Ascend) using ACL graph capture/replay."""

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self._graphs: dict[int, _DecodeGraphEntry] = {}
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._stream: torch.npu.Stream | None = None
        self._update_stream: torch.npu.Stream | None = None
        self._replay_done_event: torch.npu.Event | None = None
        self._warmed_up = False

    def can_execute(
        self, input_ids: torch.Tensor, metadata: AttentionMetadata
    ) -> bool:
        return (
            not metadata.is_prefill
            and not metadata.is_chunked_prefill
            and _decode_bucket(input_ids.shape[0]) <= self.max_batch
        )

    def warmup(self, device: torch.device, _dtype: torch.dtype) -> None:
        if self._warmed_up:
            return
        self._warmed_up = True

        buckets = [size for size in (1, 2, 4, 8) if size <= self.max_batch]
        buckets.extend(range(16, self.max_batch + 1, 16))
        page_size = self.attention_backend.page_size
        for batch_size in buckets:
            slot_base = torch.arange(
                batch_size, dtype=torch.int32, device=device
            ).mul_(page_size)
            block_ids = torch.arange(
                batch_size, dtype=torch.int32, device=device
            )
            metadata = _StaticAttentionMetadata(
                slot_mapping=slot_base,
                paged_kv_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=device
                ),
                paged_kv_indices=block_ids,
                paged_kv_last_page_len=torch.ones(
                    batch_size, dtype=torch.int32, device=device
                ),
                kv_seq_lens_host=torch.arange(
                    batch_size + 1, dtype=torch.int32, device="cpu"
                ).mul_(2),
                kv_cu_seq_lens=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=device
                ).mul_(2),
                block_table=block_ids.unsqueeze(1),
            )
            self.execute(
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                torch.zeros(batch_size, dtype=torch.int32, device=device),
                metadata,
            )

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        batch_size = input_ids.shape[0]
        padded_batch_size = _decode_bucket(batch_size)
        if padded_batch_size > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")

        entry = self._graphs.get(padded_batch_size)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(
                padded_batch_size, input_ids, positions, metadata
            )
            self._graphs[padded_batch_size] = entry

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._update_stream = torch.npu.Stream(
                device=input_ids.device, priority=-1
            )
            self._replay_done_event = torch.npu.Event()

        assert self._update_stream is not None
        assert self._replay_done_event is not None

        self._fill_entry(entry, input_ids, positions, metadata, batch_size)

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

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        if self._paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (
                self.max_model_len + page_size - 1
            ) // page_size
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * max_blocks_per_sequence,
                dtype=metadata.paged_kv_indices.dtype,
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
        entry.static_input_ids = torch.zeros(
            padded_batch_size, dtype=input_ids.dtype, device=device
        )
        entry.static_positions = torch.zeros(
            padded_batch_size, dtype=torch.int32, device=device
        )
        entry.static_metadata = _StaticAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=metadata.slot_mapping.dtype,
                device=device,
            ),
            paged_kv_indptr=torch.zeros(
                padded_batch_size + 1,
                dtype=metadata.paged_kv_indptr.dtype,
                device=device,
            ),
            paged_kv_indices=self._paged_kv_indices_buffer,
            paged_kv_last_page_len=torch.zeros(
                padded_batch_size,
                dtype=metadata.paged_kv_last_page_len.dtype,
                device=device,
            ),
            kv_cu_seq_lens=torch.zeros(
                padded_batch_size + 1,
                dtype=torch.int32,
                device=device,
            ),
            paged_kv_indptr_host=torch.zeros(
                padded_batch_size + 1, dtype=torch.int32, device="cpu"
            ),
            paged_kv_last_page_len_host=torch.ones(
                padded_batch_size, dtype=torch.int32, device="cpu"
            ),
            kv_seq_lens_host=torch.zeros(
                padded_batch_size + 1, dtype=torch.int32, device="cpu"
            ),
            block_table=static_block_table,
        )
        entry.kv_seq_lens_delta = torch.empty(
            padded_batch_size, dtype=torch.int32, device=device
        )
        entry.host_seq_lens = torch.empty(
            padded_batch_size, dtype=torch.int32, device="cpu"
        )
        entry.host_block_counts = torch.empty(
            padded_batch_size, dtype=torch.int32, device="cpu"
        )
        return entry

    def _fill_entry(
        self,
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        if metadata.kv_cu_seq_lens is None:
            raise RuntimeError(
                "decode ACL graph requires device cumulative KV lengths"
            )
        graph_positions = positions.to(torch.int32).contiguous()
        ops.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            metadata.slot_mapping,
            metadata.kv_cu_seq_lens,
            metadata.paged_kv_indptr,
            metadata.paged_kv_indices,
            metadata.paged_kv_last_page_len,
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
        self._fill_host_metadata(entry, metadata, batch_size)

        if (
            metadata.block_table is not None
            and static_metadata.block_table is not None
        ):
            src_bt = metadata.block_table.to(torch.int32)
            copy_cols = min(
                src_bt.shape[1],
                static_metadata.block_table.shape[1],
            )
            static_metadata.block_table[:batch_size, :copy_cols].copy_(
                src_bt[:batch_size, :copy_cols]
            )
            if padded_batch_size > batch_size:
                static_metadata.block_table[batch_size:].zero_()

    def _fill_host_metadata(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        cumulative_seq_lens = metadata.kv_seq_lens_host
        if cumulative_seq_lens is None:
            raise RuntimeError("decode ACL graph requires host KV lengths")
        cumulative_seq_lens = cumulative_seq_lens.cpu()
        if cumulative_seq_lens.numel() == batch_size:
            cum = torch.zeros(
                batch_size + 1,
                dtype=cumulative_seq_lens.dtype,
                device=cumulative_seq_lens.device,
            )
            cum[1:] = torch.cumsum(cumulative_seq_lens, dim=0)
            cumulative_seq_lens = cum
        if cumulative_seq_lens.numel() != batch_size + 1:
            raise RuntimeError(
                "decode ACL graph requires cumulative host KV lengths"
            )

        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        torch.sub(
            cumulative_seq_lens[1:],
            cumulative_seq_lens[:-1],
            out=entry.host_seq_lens[:batch_size],
        )
        if padded_batch_size > batch_size:
            entry.host_seq_lens[batch_size:padded_batch_size].fill_(1)
        torch.cumsum(
            entry.host_seq_lens,
            dim=0,
            out=static_metadata.kv_seq_lens_host[1:],
        )

        page_size = self.attention_backend.page_size
        if page_size == 1:
            static_metadata.paged_kv_indptr_host[: batch_size + 1].copy_(
                cumulative_seq_lens
            )
            static_metadata.paged_kv_last_page_len_host.fill_(1)
        else:
            torch.add(
                entry.host_seq_lens,
                page_size - 1,
                out=entry.host_block_counts,
            )
            torch.div(
                entry.host_block_counts,
                page_size,
                rounding_mode="floor",
                out=entry.host_block_counts,
            )
            torch.cumsum(
                entry.host_block_counts,
                dim=0,
                out=static_metadata.paged_kv_indptr_host[1:],
            )
            torch.sub(
                entry.host_seq_lens,
                1,
                out=static_metadata.paged_kv_last_page_len_host,
            )
            torch.remainder(
                static_metadata.paged_kv_last_page_len_host,
                page_size,
                out=static_metadata.paged_kv_last_page_len_host,
            )
            torch.add(
                static_metadata.paged_kv_last_page_len_host,
                1,
                out=static_metadata.paged_kv_last_page_len_host,
            )
            if padded_batch_size > batch_size:
                static_metadata.paged_kv_last_page_len_host[
                    batch_size:padded_batch_size
                ].fill_(1)

        if page_size == 1 and padded_batch_size > batch_size:
            static_metadata.paged_kv_indptr_host[
                batch_size + 1 : padded_batch_size + 1
            ].fill_(int(cumulative_seq_lens[-1]))

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        context = ForwardContext(self.attention_backend, self.device)
        with forward_context(context):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self.model(entry.static_input_ids, entry.static_positions)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(self._stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            acl_graph=capture_context,
        )
        with forward_context(context):
            with torch.npu.graph(entry.graph, stream=self._stream):
                entry.static_output = self.model(
                    entry.static_input_ids, entry.static_positions
                )
        entry.graph_tasks = capture_context.tasks

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
