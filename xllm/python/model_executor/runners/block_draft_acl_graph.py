# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ACL graphs for fixed-width, non-causal DFlash/DSpark draft blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.attention.block_draft_window import block_draft_model_window_context
from xllm.python.model_executor.forward_context import AclGraphExecutionState, EplbRuntimeState
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.acl_graph import (
    AclGraphEntry,
    StaticGraphAttentionMetadata,
)
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner
from xllm.python.model_executor.runners.decode_cuda_graph import _decode_bucket
from xllm.python.model_executor.runners.kv_capacity import block_draft_kv_capacity_buckets

_BlockGraphKey = tuple[object, ...]


class BlockDraftAclGraphRunner(DecodeAclGraphRunner):
    """Capture DSpark graphs by sequence count, query width, and page capacity.

    DSpark presents ``N`` non-causal query rows per request.  The regular ACL
    decode runner intentionally rejects this packed layout because its graph
    metadata has one row per sequence.  This runner keeps one row per request
    in the paged attention metadata and uses device KV lengths to mask the
    changing accepted prefix during replay. Full-attention page capacity grows
    in powers of two. Window-only drafts keep one full logical source table;
    FIA receives a fixed small window table selected on Device instead.

    Under DP, all ranks pad to the same whole-request bucket and coordinate
    capture/cache presence. The paired key retains each rank's page-capacity
    variant; padded rows use invalid cache slots and their outputs are sliced
    off by the decode runner after replay.
    """

    # GLM-5.2's target KDA path can run a vendor recurrent kernel on an
    # auxiliary stream while the block draft is being captured.  Keep those
    # independent target-side launches out of this draft graph's capture
    # scope; the graph stream itself is still ordered by the explicit waits in
    # the base runner.
    _capture_error_mode = "thread_local"
    _WARMUP_PAGE_COUNT = 16

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        block_draft_graph_warmup_max_kv_len: int = 0,
    ) -> None:
        # Like the decode runner, max_batch is a global capacity; here it is
        # measured in requests rather than flattened query rows.
        super().__init__(model, attention_backend, device, max_batch, max_model_len, dp_size, dp_rank)
        if type(block_draft_graph_warmup_max_kv_len) is not int or block_draft_graph_warmup_max_kv_len < -1:
            raise ValueError("Block graph warmup KV limit must be -1, zero or a positive integer")
        self._block_draft_graph_warmup_max_kv_len = block_draft_graph_warmup_max_kv_len
        self._windowed_block_draft = (
            not getattr(attention_backend, "is_mla", False) and block_draft_model_window_context(model) is not None
        )

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> bool:
        if input_ids.dim() != 1:
            return False
        if not metadata.is_chunked_prefill or metadata.is_prefill or metadata.is_spec_verify:
            return False
        if input_embedding is not None or mtp_topk_indices is not None:
            return False
        q_seq_lens = metadata.q_seq_lens
        block_table = metadata.block_table
        slot_mapping = metadata.slot_mapping
        kv_seq_lens = metadata.kv_seq_lens
        if q_seq_lens is None or block_table is None or slot_mapping is None or kv_seq_lens is None:
            return False
        if q_seq_lens.dim() != 1 or q_seq_lens.numel() == 0:
            return False
        query_widths = q_seq_lens.to(torch.int64).tolist()
        query_width = query_widths[0]
        if query_width <= 0 or any(width != query_width for width in query_widths):
            return False
        sequence_count = len(query_widths)
        if sequence_count > self.max_batch:
            return False
        if block_table.dim() != 2 or block_table.shape[0] != sequence_count:
            return False
        if input_ids.numel() != sequence_count * query_width:
            return False
        if slot_mapping.dim() != 1 or slot_mapping.numel() != input_ids.numel():
            return False
        if kv_seq_lens.dtype != torch.int32 or kv_seq_lens.shape != (sequence_count,):
            return False
        host_kv_lens = getattr(metadata, "kv_seq_lens_host_values", None)
        if host_kv_lens is None or len(host_kv_lens) != sequence_count:
            return False
        if any(int(length) <= 0 for length in host_kv_lens):
            return False
        if not self._valid_paged_metadata(metadata, sequence_count):
            return False
        query_ends = self._query_ends(metadata, query_widths)
        if query_ends is None or query_ends[-1] != input_ids.numel():
            return False
        return (
            block_table.shape[1] > 0 and self._padded_sequence_count(metadata, sequence_count, query_width) is not None
        )

    def _padded_sequence_count(
        self,
        metadata: AttentionMetadata,
        sequence_count: int,
        query_width: int,
    ) -> int | None:
        if self.dp_size <= 1:
            return sequence_count if sequence_count <= self.max_batch else None
        counts = tuple(getattr(metadata, "dp_execution_token_counts", ()))
        phases = tuple(getattr(metadata, "dp_is_decode", ()))
        if len(counts) != self.dp_size or len(phases) != self.dp_size:
            raise ValueError("DSpark DP graphs require execution counts and phases for every DP rank")
        if any(count <= 0 or count % query_width for count in counts):
            raise ValueError("DSpark DP execution counts must contain complete query blocks")
        if counts[self.dp_rank] != sequence_count * query_width:
            raise ValueError("DSpark DP local execution count does not match the query rows")
        if not all(phases):
            return None
        group_sequences = max(counts) // query_width
        if group_sequences > self.max_batch:
            return None
        return min(_decode_bucket(group_sequences), self.max_batch)

    @staticmethod
    def _query_ends(metadata: AttentionMetadata, query_widths: list[int]) -> list[int] | None:
        host_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
        if host_ends is None:
            return None
        host_ends = list(host_ends)
        if len(host_ends) == len(query_widths) + 1 and host_ends[0] == 0:
            host_ends = host_ends[1:]
        if len(host_ends) != len(query_widths):
            return None
        expected = []
        total = 0
        for width in query_widths:
            total += width
            expected.append(total)
        return host_ends if host_ends == expected else None

    @staticmethod
    def _valid_paged_metadata(metadata: AttentionMetadata, sequence_count: int) -> bool:
        paged_kv_indptr = getattr(metadata, "paged_kv_indptr", None)
        paged_kv_indices = getattr(metadata, "paged_kv_indices", None)
        paged_kv_last_page_len = getattr(metadata, "paged_kv_last_page_len", None)
        paged_metadata = (paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)
        if all(value is None for value in paged_metadata):
            return True
        if any(value is None for value in paged_metadata):
            return False
        if (
            paged_kv_indptr.dim() != 1
            or paged_kv_indptr.numel() != sequence_count + 1
            or paged_kv_indices.dim() != 1
            or paged_kv_last_page_len.dim() != 1
            or paged_kv_last_page_len.numel() != sequence_count
        ):
            return False
        # The scheduler normally keeps these tensors on the NPU.  Do not
        # call ``item``/``all`` on device tensors here: this method runs on
        # every graph candidate and such checks would synchronize the device
        # before the graph update stream can start.  CPU metadata is cheap to
        # validate eagerly; device metadata is consumed by the attention path
        # after the shape checks above.
        if paged_kv_indptr.device.type != "cpu":
            return True
        if int(paged_kv_indptr[0].item()) != 0:
            return False
        if not bool(torch.all(paged_kv_indptr[1:] >= paged_kv_indptr[:-1]).item()):
            return False
        return int(paged_kv_indptr[-1].item()) == paged_kv_indices.numel()

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        pre_capture_kv_buckets: bool = False,
        input_batch: InputBatch | None = None,
    ) -> _BlockGraphKey:
        self._validate_inputs(input_ids, positions, metadata)
        sequence_count = metadata.q_seq_lens.numel()
        query_width = int(metadata.q_seq_lens[0].item())
        padded_sequences = self._padded_sequence_count(metadata, sequence_count, query_width)
        assert padded_sequences is not None
        local_key = self._graph_key(input_ids, metadata, padded_sequences, query_width)
        local_pages = local_key[2]
        tiers: tuple[int, ...] = ()
        if pre_capture_kv_buckets and not self._windowed_block_draft:
            page_size = self._logical_page_size
            # The owned-table runner pads its last capacity to a power of two,
            # unlike the prepared runner's clamped source-table view. Include
            # the existing short-model lookahead when matching runtime keys.
            model_pages = (self.max_model_len + page_size - 1) // page_size
            warmup_length = min(self.max_model_len, page_size * self._WARMUP_PAGE_COUNT)
            warmup_pages = (warmup_length + query_width + page_size - 1) // page_size
            max_pages = 1 << (max(model_pages, warmup_pages) - 1).bit_length()
            tiers = block_draft_kv_capacity_buckets(
                page_size,
                max_pages,
                query_width,
                self._block_draft_graph_warmup_max_kv_len,
            )
            # Model-wide tiers have the same order/count on every DP rank.
            # Keep the current local capacity as a separate final step: a
            # rank-local union would mismatch collectives for unequal shards.
        for pages in (*tiers, local_pages):
            capacity_key = (*local_key[:2], pages, *local_key[3:])
            # Peer variants belong to the complete capacity-specific key.
            graph_key = self._synchronize_dp_graph_key(capacity_key, input_ids)
            if self._synchronize_dp_graph_presence(graph_key, input_ids):
                self._graphs.pop(graph_key, None)
                self._prepare_graph_entry(
                    input_ids, positions, metadata, graph_key=graph_key, eplb=eplb, input_batch=input_batch
                )
        return graph_key

    def _graph_key(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        sequence_count: int,
        query_width: int,
    ) -> _BlockGraphKey:
        return (
            sequence_count,
            query_width,
            self._page_table_capacity(metadata),
            input_ids.dtype,
            input_ids.device,
            (),
        )

    def _page_table_capacity(self, metadata: AttentionMetadata) -> int:
        host_kv_lens = getattr(metadata, "kv_seq_lens_host_values", None)
        sequence_count = int(metadata.q_seq_lens.numel())
        page_size = self._logical_page_size
        if page_size <= 0:
            raise ValueError("DSpark ACL graph page size must be positive")
        if self._windowed_block_draft:
            # Only the logical source table spans the model. The attention
            # operator receives a small Device-selected window on each replay.
            return max(1, (self.max_model_len + page_size - 1) // page_size)
        if host_kv_lens is None or len(host_kv_lens) != sequence_count:
            required_cols = int(metadata.block_table.shape[1])
        else:
            max_kv_len = max(int(length) for length in host_kv_lens)
            required_cols = max(1, (max_kv_len + page_size - 1) // page_size)
        if required_cols <= 0:
            raise ValueError("DSpark ACL graph page-table width must be positive")
        required_capacity = 1 << (required_cols - 1).bit_length()
        query_width = int(metadata.q_seq_lens[0].item())
        warmup_context_len = min(self.max_model_len, page_size * self._WARMUP_PAGE_COUNT)
        warmup_cols = max(1, (warmup_context_len + query_width + page_size - 1) // page_size)
        warmup_capacity = 1 << (warmup_cols - 1).bit_length()
        # The C++ profile warmup captures a 16-page block-diffusion shape.
        # Keep real requests within that bucket whenever possible.  Capturing
        # a smaller lazy bucket during the first request leaves FIA's graph
        # task/workspace state different from the startup capture and can
        # deadlock on the next replay.  Larger requests still get their own
        # power-of-two bucket.
        return max(required_capacity, warmup_capacity)

    def _validate_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        if not self.can_execute(input_ids, metadata):
            raise ValueError("DSpark ACL graph requires fixed-width non-causal block metadata")
        if positions.dim() != 1 or positions.numel() != input_ids.numel():
            raise ValueError("DSpark ACL graph positions must match input_ids")

    def _prepare_graph_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _BlockGraphKey,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> AclGraphEntry:
        if input_embedding is not None or mtp_topk_indices is not None:
            raise ValueError("DSpark ACL graph does not accept embedding or MTP graph inputs")
        page_table_capacity = int(graph_key[2])
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        if entry is None:
            logger.info(
                "DSpark ACL graph page-table bucket: capacity=%d source_cols=%d windowed_draft=%s",
                page_table_capacity,
                metadata.block_table.shape[1],
                self._windowed_block_draft,
            )
            entry = self._allocate_entry(
                input_ids,
                positions,
                metadata,
                sequence_count=int(graph_key[0]),
                page_table_capacity=page_table_capacity,
            )
        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)
            self._initialize_task_updates()
            self._update_done_event = torch.npu.Event()
        if self._replay_done_event is not None:
            torch.npu.current_stream().wait_event(self._replay_done_event)
        if self._update_done_recorded:
            assert self._update_done_event is not None
            torch.npu.current_stream().wait_event(self._update_done_event)
        self._prepare_graph_eplb_state(entry, eplb, metadata, first_capture=first_capture)
        self._fill_entry(entry, input_ids, positions, metadata)
        self._bind_graph_input_batch(entry, input_batch)
        self._prepare_attention(entry, entry.static_metadata)
        if first_capture:
            self._capture(entry, self._stream)
            self._graphs[graph_key] = entry
        return entry

    def _allocate_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        *,
        sequence_count: int,
        page_table_capacity: int,
    ) -> AclGraphEntry:
        del positions
        query_width = int(metadata.q_seq_lens[0].item())
        token_count = sequence_count * query_width
        device = input_ids.device
        page_size = self._logical_page_size
        if page_size <= 0:
            raise ValueError("DSpark ACL graph page size must be positive")
        kv_capacity = page_table_capacity * page_size
        if self._query_ends(metadata, [query_width] * metadata.q_seq_lens.numel()) is None:
            raise ValueError("DSpark ACL graph requires canonical query ends")
        query_ends = [query_width * (index + 1) for index in range(sequence_count)]
        static_block_table = torch.zeros(
            sequence_count,
            page_table_capacity,
            dtype=torch.int32,
            device=device,
        )
        static_kv_lens = torch.empty(sequence_count, dtype=torch.int32, device=device)
        static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=torch.full((token_count,), -1, dtype=metadata.slot_mapping.dtype, device=device),
            paged_kv_indptr=torch.zeros(sequence_count + 1, dtype=torch.int32, device=device),
            paged_kv_indices=torch.zeros(sequence_count * page_table_capacity, dtype=torch.int32, device=device),
            paged_kv_last_page_len=torch.ones(sequence_count, dtype=torch.int32, device=device),
            q_cu_seq_lens=torch.tensor([0, *query_ends], dtype=torch.int32, device=device),
            q_cu_seq_lens_host_values=list(query_ends),
            q_seq_lens=torch.full(
                (sequence_count,),
                query_width,
                dtype=torch.int32,
                device=device,
            ),
            q_seq_lens_host=(
                torch.full((sequence_count,), query_width, dtype=torch.int32)
                if getattr(metadata, "q_seq_lens_host", None) is not None
                else None
            ),
            kv_cu_seq_lens=(
                torch.zeros(sequence_count + 1, dtype=torch.int32, device=device)
                if metadata.kv_cu_seq_lens is not None
                else None
            ),
            kv_seq_lens_host_values=[kv_capacity] * sequence_count,
            block_table=static_block_table,
            kv_seq_lens=static_kv_lens,
            is_chunked_prefill=True,
            dp_execution_token_counts=(token_count,) * self.dp_size if self.dp_size > 1 else (),
            dp_is_decode=(1,) * self.dp_size if self.dp_size > 1 else (),
        )
        static_metadata.prepared_attention_state = self.attention_backend.prepare_metadata(
            static_metadata, device_kv_lengths=True
        )
        entry = AclGraphEntry()
        entry.batch_size = token_count
        entry.graph = None
        entry.static_output = None
        entry.static_input_ids = torch.zeros(token_count, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(token_count, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        entry.static_mtp_topk_indices = None
        entry.static_metadata = static_metadata
        entry.kv_seq_lens_delta = static_kv_lens
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        return entry

    def _fill_entry(
        self,
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> None:
        static = entry.static_metadata
        block_table = metadata.block_table.to(torch.int32)
        if block_table.dim() != 2 or not 0 < block_table.shape[0] <= static.block_table.shape[0]:
            raise ValueError("DSpark ACL graph block_table shape changed within a graph bucket")
        sequence_count = block_table.shape[0]
        token_count = input_ids.numel()
        query_width = int(static.q_cu_seq_lens_host_values[0])
        if (
            input_ids.dim() != 1
            or token_count != sequence_count * query_width
            or positions.shape != input_ids.shape
            or metadata.slot_mapping.shape != input_ids.shape
            or metadata.kv_seq_lens.shape != (sequence_count,)
            or self._query_ends(metadata, [query_width] * sequence_count) is None
        ):
            raise ValueError("DSpark ACL graph query layout changed within a graph bucket")
        if not self._valid_paged_metadata(metadata, block_table.shape[0]):
            raise ValueError("DSpark ACL graph paged KV metadata is inconsistent")
        page_counts = self._page_counts(metadata, static.block_table.shape[1])
        if page_counts is None:
            raise ValueError("DSpark ACL graph requires one Host KV length per sequence")
        if max(page_counts, default=0) > block_table.shape[1]:
            raise RuntimeError("DSpark ACL graph Host KV lengths exceed the source block table")
        if max(page_counts, default=0) > static.block_table.shape[1]:
            raise RuntimeError(
                "DSpark ACL graph block_table exceeds the captured page-table capacity: "
                f"live={max(page_counts)}, capacity={static.block_table.shape[1]}"
            )
        kv_host = getattr(metadata, "kv_seq_lens_host_values", None)
        if kv_host is None or len(kv_host) != block_table.shape[0]:
            raise ValueError("DSpark ACL graph requires one Host KV length per sequence")
        kv_host = [int(length) for length in kv_host]
        prepared = static.prepared_attention_state
        if prepared is None:
            raise RuntimeError("DSpark ACL graph prepared attention state is missing")

        entry.static_input_ids[:token_count].copy_(input_ids)
        entry.static_positions[:token_count].copy_(positions.to(torch.int32))
        entry.static_input_ids[token_count:].zero_()
        entry.static_positions[token_count:].zero_()
        static.slot_mapping[:token_count].copy_(metadata.slot_mapping)
        static.slot_mapping[token_count:].fill_(-1)
        static.block_table.zero_()
        static.paged_kv_indices.zero_()
        static.paged_kv_last_page_len.fill_(1)
        page_offset = 0
        for row, page_count in enumerate(page_counts):
            static.block_table[row, :page_count].copy_(block_table[row, :page_count])
            # The scheduler's paging tensors may include reserved future
            # pages. Rebuild the graph-owned compact layout from the live KV
            # lengths instead of copying that unrelated allocation capacity.
            static.paged_kv_indptr[row].fill_(page_offset)
            static.paged_kv_indices[page_offset : page_offset + page_count].copy_(block_table[row, :page_count])
            static.paged_kv_last_page_len[row].fill_((kv_host[row] - 1) % self._logical_page_size + 1)
            page_offset += page_count
        static.paged_kv_indptr[sequence_count:].fill_(page_offset)
        static.kv_seq_lens[:sequence_count].copy_(metadata.kv_seq_lens)
        static.kv_seq_lens[sequence_count:].fill_(1)
        if static.kv_cu_seq_lens is not None:
            static.kv_cu_seq_lens[0].zero_()
            torch.cumsum(static.kv_seq_lens, dim=0, out=static.kv_cu_seq_lens[1:])
        static.kv_seq_lens_host_values[:] = kv_host + [1] * (static.block_table.shape[0] - sequence_count)

        if not self.attention_backend.is_mla:
            prepared.actual_seq_q[:] = list(static.q_cu_seq_lens_host_values)
        # BNSD FIA captures Host KV lengths by value, not by list address.
        # Keep the allocation-time capacity bound even during first capture;
        # the captured Device-length mask controls the live prefix on replay.
        if prepared.query_ends is not None:
            prepared.query_ends[:] = list(static.q_cu_seq_lens_host_values)

    def _page_counts(self, metadata: AttentionMetadata, capacity: int) -> list[int] | None:
        host_kv_lens = getattr(metadata, "kv_seq_lens_host_values", None)
        sequence_count = int(metadata.q_seq_lens.numel())
        if host_kv_lens is None or len(host_kv_lens) != sequence_count:
            return None
        page_size = self._logical_page_size
        del capacity
        if page_size <= 0 or any(int(length) <= 0 for length in host_kv_lens):
            raise ValueError("DSpark ACL graph Host KV lengths must be positive")
        return [max(1, (int(kv_len) + page_size - 1) // page_size) for kv_len in host_kv_lens]
