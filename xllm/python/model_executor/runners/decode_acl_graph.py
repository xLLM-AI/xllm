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

import os
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python import kernels
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
    q_seq_lens_host: torch.Tensor | None = None
    dsa_positions: torch.Tensor | None = None
    dsa_cos_sin: torch.Tensor | None = None
    dsa_c4_cos_sin: torch.Tensor | None = None
    dsa_c128_cos_sin: torch.Tensor | None = None
    dsa_graph_block_table_cols: int = 0
    dsa_graph_mode: bool = False
    paged_kv_indptr_host: torch.Tensor | None = None
    paged_kv_last_page_len_host: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    kv_seq_lens: torch.Tensor | None = None
    max_query_len: int = 1
    max_seq_len: int = 1
    # Mirrors ModelInputParams::meta.actual_num_sequences.  Graph tensors use
    # the bucket row count, while DSA metadata must zero rows that do not
    # belong to the current forward.
    actual_num_sequences: int = 0
    dsa_metadata: object | None = None
    multi_block_tables: tuple[torch.Tensor, ...] = ()
    dp_token_counts: tuple[int, ...] = ()
    linear_state_indices: torch.Tensor | None = None
    has_initial_state: torch.Tensor | None = None
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
        "stream",
        "update_stream",
        "replay_done_event",
    )


class _GraphSlot:
    __slots__ = (
        "graphs",
        "paged_kv_indices_buffer",
        "is_prepared",
        "prepared_batch_size",
        "prepared_actual_batch_size",
    )

    def __init__(self) -> None:
        self.graphs: dict[int, _DecodeGraphEntry] = {}
        self.paged_kv_indices_buffer: torch.Tensor | None = None
        self.is_prepared = False
        self.prepared_batch_size = 0
        self.prepared_actual_batch_size = 0


class DecodeAclGraphRunner(BaseRunner):
    """Decode graph runner for NPU (Ascend) using ACL graph capture/replay."""

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        enable_double_buffer: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.max_batch = max_batch
        self.max_model_len = max_model_len
        self._graph_slot_count = 2 if enable_double_buffer else 1
        self._graph_slots = [
            _GraphSlot() for _ in range(self._graph_slot_count)
        ]
        self._next_replay_slot = 0
        self._last_started_replay_slot = -1
        self._max_blocks_per_sequence: int = 0
        self._warmed_up = False

    def can_execute(
        self, input_ids: torch.Tensor, metadata: AttentionMetadata
    ) -> bool:
        return (
            not metadata.is_prefill
            and not metadata.is_chunked_prefill
            and _decode_bucket(input_ids.shape[0]) <= self.max_batch
        )

    def _take_replay_slot(
        self, padded_batch_size: int, actual_batch_size: int
    ) -> tuple[_GraphSlot, bool]:
        slot_idx = self._next_replay_slot
        self._next_replay_slot = (
            self._next_replay_slot + 1
        ) % self._graph_slot_count
        self._last_started_replay_slot = slot_idx
        slot = self._graph_slots[slot_idx]
        inputs_prepared = (
            slot.is_prepared
            and slot.prepared_batch_size == padded_batch_size
            and slot.prepared_actual_batch_size == actual_batch_size
        )
        slot.is_prepared = False
        return slot, inputs_prepared

    def _next_prepare_slot(self) -> _GraphSlot | None:
        if self._graph_slot_count <= 1 or self._last_started_replay_slot < 0:
            return None
        slot_idx = (
            self._last_started_replay_slot + 1
        ) % self._graph_slot_count
        return self._graph_slots[slot_idx]

    def warmup(self, device: torch.device, _dtype: torch.dtype) -> None:
        if self._warmed_up:
            return
        self._warmed_up = True

        # C++ AclGraphExecutorImpl captures each graph lazily from the first
        # real request in a slot. Double buffer therefore does not allocate two
        # complete sets of bucket graphs during service startup. Keep the same
        # lifecycle here; eager synthetic capture made the second DSV4 slot OOM
        # before the server became ready and also captured metadata that did
        # not belong to a real forward.
        if os.getenv("XLLM_ACLGRAPH_EAGER_WARMUP", "0") != "1":
            return

        bucket_override = os.getenv("XLLM_DSV4_GRAPH_WARMUP_BUCKETS", "")
        if bucket_override:
            buckets = sorted(
                {
                    int(value.strip())
                    for value in bucket_override.split(",")
                    if value.strip()
                }
            )
            if not buckets or any(
                bucket <= 0 or bucket > self.max_batch for bucket in buckets
            ):
                raise ValueError(
                    "XLLM_DSV4_GRAPH_WARMUP_BUCKETS must contain positive "
                    f"bucket sizes no greater than {self.max_batch}"
                )
        else:
            buckets = [
                size for size in (1, 2, 4, 8) if size <= self.max_batch
            ]
            buckets.extend(range(16, self.max_batch + 1, 16))
        page_size = self.attention_backend.page_size
        dummy_kv_len = max(
            int(getattr(self.attention_backend, "index_topk", 1)),
            int(getattr(self.attention_backend, "window_size", 1)),
            1,
        )
        pages_per_sequence = (
            dummy_kv_len + page_size - 1
        ) // page_size
        for batch_size in buckets:
            block_table = torch.arange(
                batch_size * pages_per_sequence,
                dtype=torch.int32,
                device=device,
            ).view(batch_size, pages_per_sequence)
            slot_base = block_table[:, -1].mul(page_size).add(
                (dummy_kv_len - 1) % page_size
            )
            manager_block_ids = torch.arange(
                batch_size, dtype=torch.int32, device=device
            )
            warmup_multi_tables = self._make_warmup_multi_block_tables(
                batch_size, manager_block_ids
            )
            metadata = _StaticAttentionMetadata(
                slot_mapping=slot_base,
                paged_kv_indptr=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=device
                ).mul_(pages_per_sequence),
                paged_kv_indices=block_table.reshape(-1),
                paged_kv_last_page_len=torch.full(
                    (batch_size,),
                    (dummy_kv_len - 1) % page_size + 1,
                    dtype=torch.int32,
                    device=device,
                ),
                kv_seq_lens_host=torch.full(
                    (batch_size,), dummy_kv_len,
                    dtype=torch.int32, device="cpu"
                ),
                q_seq_lens_host=torch.ones(
                    batch_size, dtype=torch.int32, device="cpu"
                ),
                kv_cu_seq_lens=torch.arange(
                    batch_size + 1, dtype=torch.int32, device=device
                ).mul_(dummy_kv_len),
                block_table=block_table,
                multi_block_tables=warmup_multi_tables,
                max_query_len=1,
                max_seq_len=dummy_kv_len,
                # C++ ACL graphs are captured lazily from a real request, so
                # capture metadata always has at least the triggering rows as
                # valid sequences.  Python eagerly captures every bucket; its
                # synthetic rows must therefore be valid dummy requests, not
                # an empty-DP rank (all-zero multi-row DSA metadata is rejected
                # by SparseAttnSharedkvMetadata).
                actual_num_sequences=batch_size,
            )
            for _ in range(self._graph_slot_count):
                self.execute(
                    torch.zeros(batch_size, dtype=torch.int32, device=device),
                    torch.full(
                        (batch_size,), dummy_kv_len - 1,
                        dtype=torch.int32, device=device,
                    ),
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

        slot, inputs_prepared = self._take_replay_slot(
            padded_batch_size, batch_size
        )

        entry = slot.graphs.get(padded_batch_size)
        first_capture = entry is None
        if first_capture:
            entry = self._allocate_entry(
                slot, padded_batch_size, input_ids, positions, metadata
            )
            slot.graphs[padded_batch_size] = entry

        if inputs_prepared:
            self._fill_tokens(entry, input_ids, batch_size)
        else:
            self._fill_entry(entry, input_ids, positions, metadata, batch_size)

        self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
        if not inputs_prepared:
            self._attach_graph_model_metadata(
                entry.static_positions, entry.static_metadata
            )
            self._prepare_graph_forward_metadata(entry.static_metadata)

        if (
            entry.graph is None
            and os.getenv("XLLM_DSV4_GRAPH_SKIP_CAPTURE", "0") == "1"
        ):
            hidden_size = int(self.model.cfg.hidden_size)
            model_dtype = next(self.model.parameters()).dtype
            return torch.zeros(
                batch_size,
                hidden_size,
                dtype=model_dtype,
                device=input_ids.device,
            )

        if first_capture:
            self._capture(entry)

        skip_replay = os.getenv("XLLM_DSV4_GRAPH_SKIP_REPLAY", "0") == "1"
        entry.stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(entry.stream):
            if not skip_replay:
                entry.graph.replay()
            output = entry.static_output[:batch_size].clone()

        # Match AclGraph::make_current_stream_wait_for_graph().  Replay is
        # enqueued on the graph stream; the caller may immediately launch an
        # eager prefill on the current stream.  Without this dependency that
        # prefill can race graph kernels and report their asynchronous failure
        # at an unrelated metadata synchronization point.
        torch.npu.current_stream().wait_stream(entry.stream)
        if os.getenv("XLLM_DSV4_GRAPH_REPLAY_SYNC", "0") == "1":
            torch.npu.synchronize()

        # A captured FIA task waits on its update event before execution.  This
        # lets replay run concurrently with the host-side updates for later
        # layers while preventing the graph from observing stale parameters.
        with torch.npu.stream(entry.update_stream):
            entry.update_stream.wait_event(entry.replay_done_event)
            self._update_graph_tasks(entry.update_stream, entry.graph_tasks)

        # The next update must not overwrite task parameters until this replay
        # has consumed them.
        entry.replay_done_event.record(entry.stream)

        torch.npu.current_stream().wait_stream(entry.stream)
        return output

    def prepare_graph_input(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        """Prepare the next graph slot on the producer stream."""
        slot = self._next_prepare_slot()
        if slot is None:
            return
        batch_size = input_ids.shape[0]
        padded_batch_size = _decode_bucket(batch_size)
        if padded_batch_size > self.max_batch:
            return

        if slot.is_prepared:
            return
        entry = slot.graphs.get(padded_batch_size)
        if entry is None:
            return

        self._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size,
            skip_token_update=True,
        )
        self.attention_backend.prepare(entry.static_metadata, graph_mode=True)
        self._attach_graph_model_metadata(
            entry.static_positions, entry.static_metadata
        )
        self._prepare_graph_forward_metadata(entry.static_metadata)
        slot.prepared_batch_size = padded_batch_size
        slot.prepared_actual_batch_size = batch_size
        slot.is_prepared = True

    def _allocate_entry(
        self,
        slot: _GraphSlot,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> _DecodeGraphEntry:
        device = input_ids.device
        if slot.paged_kv_indices_buffer is None:
            page_size = self.attention_backend.page_size
            max_blocks_per_sequence = (
                self.max_model_len + page_size - 1
            ) // page_size
            slot.paged_kv_indices_buffer = torch.zeros(
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
        entry.stream = torch.npu.Stream(device=device)
        entry.update_stream = torch.npu.Stream(device=device, priority=-1)
        entry.replay_done_event = torch.npu.Event()
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
            paged_kv_indices=slot.paged_kv_indices_buffer,
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
                padded_batch_size, dtype=torch.int32, device="cpu"
            ),
            q_seq_lens_host=torch.ones(
                padded_batch_size, dtype=torch.int32, device="cpu"
            ),
            block_table=static_block_table,
            multi_block_tables=self._allocate_multi_block_tables(
                metadata, padded_batch_size
            ),
            max_query_len=1,
            max_seq_len=max(int(getattr(metadata, "max_seq_len", 1)), 1),
            actual_num_sequences=0,
            dp_token_counts=tuple(getattr(metadata, "dp_token_counts", ())),
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
        *,
        skip_token_update: bool = False,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        if metadata.kv_cu_seq_lens is None:
            raise RuntimeError(
                "decode ACL graph requires device cumulative KV lengths"
            )
        graph_positions = positions.to(torch.int32).contiguous()
        if os.getenv("XLLM_DSV4_GRAPH_META_DEBUG", "0") == "1":
            def _describe(name: str, tensor: torch.Tensor) -> str:
                return (
                    f"{name}=shape{tuple(tensor.shape)}/numel={tensor.numel()}"
                    f"/{tensor.dtype}/{tensor.device}"
                )

            sources = (
                ("tokens", input_ids),
                ("positions", graph_positions),
                ("slot_mapping", metadata.slot_mapping),
                ("kv_cu_seq_lens", metadata.kv_cu_seq_lens),
                ("paged_kv_indptr", metadata.paged_kv_indptr),
                ("paged_kv_indices", metadata.paged_kv_indices),
                ("paged_kv_last_page_len", metadata.paged_kv_last_page_len),
            )
            destinations = (
                ("dst_tokens", entry.static_input_ids),
                ("dst_positions", entry.static_positions),
                ("dst_slot_mapping", static_metadata.slot_mapping),
                ("dst_kv_cu_seq_lens", static_metadata.kv_cu_seq_lens),
                ("dst_paged_kv_indptr", static_metadata.paged_kv_indptr),
                ("dst_paged_kv_indices", static_metadata.paged_kv_indices),
                (
                    "dst_paged_kv_last_page_len",
                    static_metadata.paged_kv_last_page_len,
                ),
            )
            print(
                "[ACLGRAPH META] "
                f"batch={batch_size} padded={padded_batch_size} "
                + " ".join(_describe(name, tensor) for name, tensor in sources)
                + " | "
                + " ".join(
                    _describe(name, tensor) for name, tensor in destinations
                ),
                flush=True,
            )
        kernels.update_decode_graph_metadata(
            entry.static_input_ids[:batch_size]
            if skip_token_update
            else input_ids,
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
        self._fill_multi_block_tables(
            static_metadata, metadata, batch_size, padded_batch_size
        )

        if static_metadata.block_table is not None:
            self._fill_block_table(
                static_metadata, metadata.block_table, batch_size
            )

    @staticmethod
    def _fill_tokens(
        entry: _DecodeGraphEntry,
        input_ids: torch.Tensor,
        batch_size: int,
    ) -> None:
        entry.static_input_ids.zero_()
        if batch_size > 0:
            entry.static_input_ids[:batch_size].copy_(input_ids)

    def _fill_host_metadata(
        self,
        entry: _DecodeGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        host_kv_lens = metadata.kv_seq_lens_host
        if host_kv_lens is None:
            raise RuntimeError("decode ACL graph requires host KV lengths")
        host_kv_lens = host_kv_lens.cpu()
        if host_kv_lens.numel() != batch_size:
            raise RuntimeError(
                "decode ACL graph requires per-sequence host KV lengths"
            )

        padded_batch_size = entry.batch_size
        actual_rows = int(
            getattr(metadata, "actual_num_sequences", batch_size)
        )
        actual_rows = min(max(actual_rows, 0), batch_size, padded_batch_size)
        static_metadata = entry.static_metadata
        static_metadata.actual_num_sequences = actual_rows
        entry.host_seq_lens[:batch_size].copy_(host_kv_lens)
        if actual_rows < padded_batch_size:
            entry.host_seq_lens[actual_rows:padded_batch_size].zero_()
        static_metadata.kv_seq_lens_host.copy_(entry.host_seq_lens)
        static_metadata.q_seq_lens_host.zero_()
        static_metadata.q_seq_lens_host[:actual_rows].fill_(1)
        static_metadata.max_query_len = 1
        static_metadata.max_seq_len = max(
            int(host_kv_lens.max().item()) if batch_size else 1, 1
        )
        cumulative_seq_lens = torch.zeros(
            padded_batch_size + 1, dtype=torch.int32, device="cpu"
        )
        torch.cumsum(
            entry.host_seq_lens,
            dim=0,
            out=cumulative_seq_lens[1:],
        )

        page_size = self.attention_backend.page_size
        if page_size == 1:
            static_metadata.paged_kv_indptr_host[: batch_size + 1].copy_(
                cumulative_seq_lens[: batch_size + 1]
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

    def _allocate_multi_block_tables(
        self, metadata: AttentionMetadata, padded_batch_size: int
    ) -> tuple[torch.Tensor, ...]:
        tables = getattr(metadata, "multi_block_tables", ())
        return tuple(
            torch.full(
                (padded_batch_size, self._max_blocks_per_sequence),
                -1,
                dtype=torch.int32,
                device=table.device,
            )
            for table in tables
        )

    def _make_warmup_multi_block_tables(
        self, batch_size: int, block_ids: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        group_infos = getattr(self.attention_backend, "group_infos", ())
        manager_count = max(len(group_infos), 1)
        return tuple(block_ids.unsqueeze(1).clone() for _ in range(manager_count))

    @staticmethod
    def _fill_block_table(
        static_metadata: _StaticAttentionMetadata,
        source: torch.Tensor | None,
        batch_size: int,
    ) -> None:
        dst = static_metadata.block_table
        if dst is None:
            raise RuntimeError("decode ACL graph block table is not allocated")
        # Match GraphPersistentParam::update(): restore the complete graph
        # buffer to its default even when this forward has no source table.
        dst.zero_()
        if source is None:
            return
        src = source.to(torch.int32)
        if src.dim() != 2 or dst.dim() != 2:
            raise RuntimeError("decode ACL graph block tables must be 2-D")
        if src.size(1) > dst.size(1):
            raise RuntimeError(
                "decode ACL graph block table exceeds capture capacity: "
                f"{src.size(1)} > {dst.size(1)}"
            )
        rows = min(batch_size, int(src.size(0)))
        if rows > 0 and src.size(1) > 0:
            dst[:rows, : src.size(1)].copy_(src[:rows])

    @staticmethod
    def _fill_multi_block_tables(
        static_metadata: _StaticAttentionMetadata,
        metadata: AttentionMetadata,
        batch_size: int,
        padded_batch_size: int,
    ) -> None:
        source_tables = tuple(getattr(metadata, "multi_block_tables", ()))
        if len(source_tables) != len(static_metadata.multi_block_tables):
            raise RuntimeError("DSA manager count changed within an ACL graph bucket")
        for dst, src in zip(static_metadata.multi_block_tables, source_tables):
            if src.dim() != 2 or dst.dim() != 2:
                raise RuntimeError("DSA ACL graph block tables must be 2-D")
            if src.size(1) > dst.size(1):
                raise RuntimeError(
                    "DSA ACL graph block table exceeds capture capacity: "
                    f"{src.size(1)} > {dst.size(1)}"
                )
            dst.fill_(-1)
            rows = min(batch_size, int(src.size(0)))
            cols = int(src.size(1))
            dst[:rows, :cols].copy_(src[:rows, :cols].to(torch.int32))
            if padded_batch_size > batch_size:
                dst[batch_size:].fill_(-1)

    def _prepare_graph_forward_metadata(
        self, metadata: _StaticAttentionMetadata
    ) -> None:
        prepare = getattr(
            self.attention_backend, "prepare_graph_forward_metadata", None
        )
        if prepare is not None:
            prepare(metadata)

    def _attach_graph_model_metadata(
        self,
        positions: torch.Tensor,
        metadata: _StaticAttentionMetadata,
    ) -> None:
        model = getattr(self.model, "model", self.model)
        attach = getattr(model, "attach_rope_tables_to_backend", None)
        if attach is not None:
            attach(
                self.attention_backend,
                positions,
                graph_bt_cols=self._max_blocks_per_sequence,
                metadata=metadata,
            )

    def _capture(self, entry: _DecodeGraphEntry) -> None:
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
        )
        with forward_context(context):
            for _ in range(_CAPTURE_WARMUP_STEPS):
                self.model(entry.static_input_ids, entry.static_positions)
        torch.npu.synchronize()
        entry.graph = torch.npu.NPUGraph()
        capture_context = AclGraphCaptureContext(entry.stream, [])
        context = ForwardContext(
            self.attention_backend,
            self.device,
            entry.static_metadata,
            self.layer_caches,
            acl_graph=capture_context,
        )
        with forward_context(context):
            with torch.npu.graph(entry.graph, stream=entry.stream):
                entry.static_output = self.model(
                    entry.static_input_ids, entry.static_positions
                )
        entry.graph_tasks = capture_context.tasks
        # Match AclGraph::capture(): complete capture-stream work before the
        # first test replay.  Without this boundary, an asynchronous capture
        # failure is reported by an unrelated metadata call in the next
        # request/bucket and is incorrectly attributed to that producer.
        entry.stream.synchronize()

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
