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

"""Decode graphs bound to final Slot views; never allocates model inputs."""

from __future__ import annotations

import torch
import torch.nn as nn

from scripts.logger import logger
from xllm.python.attention.backend import AttentionBackend, AttentionMetadata
from xllm.python.model_executor.dspark_graph_input import DSparkAdaptiveGraphInputs
from xllm.python.model_executor.forward_context import AclGraphExecutionState, EplbRuntimeState, LayerSynchronizer
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.kv_capacity import kv_capacity_bucket, kv_capacity_buckets


class _PreparedBlockGraphEntry(AclGraphEntry):
    """Retain the Slot table source for an in-graph compact metadata copy."""

    __slots__ = ("source_block_table",)

    source_block_table: torch.Tensor


class _PreparedDSparkGraphEntry(AclGraphEntry):
    """Capture adaptive packing into the target's borrowed input views."""

    __slots__ = ("dspark_adaptive",)

    dspark_adaptive: DSparkAdaptiveGraphInputs


def _dspark_graph_tensors(inputs: DSparkAdaptiveGraphInputs) -> tuple[torch.Tensor, ...]:
    return (
        inputs.arena,
        inputs.control,
        inputs.anchors,
        inputs.base_positions,
        inputs.base_kv_lengths,
        inputs.base_block_table,
        inputs.offsets,
        inputs.owners,
        inputs.columns,
    )


class PreparedAclGraphRunner(AclGraphRunner):
    """Owns captured graphs; input storage is borrowed from native Slots.

    Regular warmup Tasks populate the cache before serving admits a bucket.
    Native Slots own bucket selection and padding. Launch looks up the captured
    binding, uses the caller's task stream and returns a borrowed output whose
    readers must retire before reuse. A missing binding never captures at runtime.

    Block drafts contain one complete equal-width query per request row.
    Native padding must use reserved cache rows with valid device KV lengths;
    zero-length dummy requests are not an attention padding representation.
    A strided multi-request page-table slice uses a small contiguous table
    allocated at warmup and refreshed inside the captured forward.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        enable_mega_moe_token_mask: bool = False,
        block_draft: bool = False,
        spec_verify_width: int = 1,
        device_kv_lengths: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self._enable_mega_moe_token_mask = enable_mega_moe_token_mask
        self._block_draft = block_draft
        if not 1 <= spec_verify_width <= 17 or (block_draft and spec_verify_width != 1):
            raise ValueError("Speculative verify graph width must be in [1, 17], with width one for block drafts")
        self._spec_verify_width = spec_verify_width
        if device_kv_lengths and dp_size != 1:
            raise ValueError("Device-length ACL graphs require DP size one")
        self._device_kv_lengths = device_kv_lengths
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self._prepared_graphs: dict[tuple[object, ...], AclGraphEntry] = {}
        self.prepared_replays = 0

    @staticmethod
    def _prepared_binding(
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None = None,
    ) -> tuple[object, ...]:
        tensors = (
            input_ids,
            positions,
            metadata.slot_mapping,
            metadata.block_table,
            metadata.q_seq_lens,
            getattr(metadata, "q_cu_seq_lens", None),
            metadata.kv_seq_lens,
            getattr(metadata, "linear_state_indices", None),
            getattr(metadata, "linear_state_read_indices", None),
            getattr(metadata, "linear_state_write_indices", None),
            getattr(metadata, "has_initial_state", None),
            getattr(metadata, "num_accepted_tokens", None),
            input_embedding,
            mtp_topk_indices,
        )
        if dspark_adaptive is not None:
            tensors += _dspark_graph_tensors(dspark_adaptive)
        addresses = tuple(
            None
            if tensor is None
            else (
                tensor.data_ptr(),
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.dtype,
                tensor.device,
            )
            for tensor in tensors
        )

        counts = tuple(getattr(metadata, "dp_execution_token_counts", ()))
        phases = tuple(getattr(metadata, "dp_is_decode", ()))
        dspark_binding = (
            ()
            if dspark_adaptive is None
            else ("adaptive_dspark", dspark_adaptive.block_size, dspark_adaptive.draft_width)
        )
        return addresses + dspark_binding + ((counts, phases) if len(counts) > 1 else ())

    def _validate_dspark_graph_inputs(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        inputs: DSparkAdaptiveGraphInputs,
    ) -> None:
        if not isinstance(inputs, DSparkAdaptiveGraphInputs):
            raise ValueError("adaptive DSpark requires a DSparkAdaptiveGraphInputs descriptor")
        if any(not isinstance(tensor, torch.Tensor) for tensor in _dspark_graph_tensors(inputs)):
            raise ValueError("adaptive DSpark descriptor fields must be tensors")
        if input_ids.device != self.device:
            raise ValueError("adaptive DSpark inputs must use the graph runner device")
        if inputs.anchors.ndim != 1 or not 1 <= inputs.anchors.numel() <= 128:
            raise ValueError("adaptive DSpark requires request capacity in [1, 128]")
        if not isinstance(inputs.draft_width, int) or inputs.draft_width not in (5, 7):
            raise ValueError("adaptive DSpark draft_width must be 5 or 7")
        capacity = inputs.anchors.numel()
        graph_rows = input_ids.numel()
        if graph_rows > 128 * (inputs.draft_width + 1):
            raise ValueError("adaptive DSpark graph capacity exceeds the compiled draft-width limit")
        if inputs.base_block_table.ndim != 2 or inputs.base_block_table.shape[0] != capacity:
            raise ValueError("adaptive DSpark requires a block table per request-capacity row")
        table_width = inputs.base_block_table.shape[1]
        if (
            table_width == 0
            or not isinstance(inputs.block_size, int)
            or isinstance(inputs.block_size, bool)
            or inputs.block_size <= 0
            or table_width * inputs.block_size > 2**31 - 1
            or max(capacity, graph_rows) * table_width > 2**31 - 1
        ):
            raise ValueError("adaptive DSpark requires positive INT32 cache-page geometry")
        if inputs.block_size != self.attention_backend.logical_page_size:
            raise ValueError("adaptive DSpark block size must match the target attention backend")
        for name, tensor, shape, dtype in (
            ("arena", inputs.arena, ((12 * inputs.draft_width + 4) * capacity,), torch.uint8),
            ("control", inputs.control, (1,), torch.int64),
            ("anchors", inputs.anchors, (capacity,), torch.int64),
            ("base positions", inputs.base_positions, (capacity,), torch.int32),
            ("base KV lengths", inputs.base_kv_lengths, (capacity,), torch.int32),
            ("base block table", inputs.base_block_table, (capacity, table_width), torch.int32),
            ("offsets", inputs.offsets, (capacity + 1,), torch.int32),
            ("owners", inputs.owners, (graph_rows,), torch.int32),
            ("columns", inputs.columns, (graph_rows,), torch.int32),
            ("tokens", input_ids, (graph_rows,), torch.int32),
            ("positions", positions, (graph_rows,), torch.int32),
            ("KV lengths", metadata.kv_seq_lens, (graph_rows,), torch.int32),
            ("cache slots", metadata.slot_mapping, (graph_rows,), torch.int32),
            ("block table", metadata.block_table, (graph_rows, table_width), torch.int32),
            ("query lengths", metadata.q_seq_lens, (graph_rows,), torch.int32),
            ("query ends", getattr(metadata, "q_cu_seq_lens", None), (graph_rows,), torch.int32),
        ):
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != input_ids.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"adaptive DSpark requires contiguous {dtype} {name} with shape {shape} on the input device"
                )

    def _validate_prepared_dp(self, input_ids: torch.Tensor, metadata: AttentionMetadata) -> None:
        if self.dp_size == 1:
            return
        counts = tuple(metadata.dp_execution_token_counts)
        phases = tuple(metadata.dp_is_decode)
        if len(counts) != self.dp_size or len(phases) != self.dp_size or any(count <= 0 for count in counts):
            raise ValueError("prepared DP requires complete positive execution counts and phases")
        if counts[self.dp_rank] != input_ids.numel():
            raise ValueError("prepared DP execution counts do not match local physical rows")
        if not all(phases) or len(set(counts)) != 1:
            raise ValueError("prepared DP graphs require the same decode batch size on every rank")

    def _validate_prepared_input(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None = None,
    ) -> None:
        batch_size = input_ids.numel()
        if not 0 < batch_size <= self.max_batch:
            raise ValueError("prepared ACL graph batch size exceeds capacity or is empty")
        if dspark_adaptive is not None and (
            self._block_draft
            or not self.attention_backend.is_mla
            or input_embedding is not None
            or mtp_topk_indices is not None
        ):
            raise ValueError("adaptive DSpark requires an ordinary MLA target graph with token-only inputs")
        self._validate_prepared_dp(input_ids, metadata)
        if self._block_draft:
            if metadata.is_prefill or not metadata.is_chunked_prefill or metadata.is_spec_verify:
                raise ValueError("prepared block graph requires non-causal chunked-prefill queries")
            if input_embedding is not None or mtp_topk_indices is not None:
                raise ValueError("prepared block graph requires ordinary token inputs")
        elif self._spec_verify_width > 1:
            if metadata.is_prefill or metadata.is_chunked_prefill or not metadata.is_spec_verify:
                raise ValueError("prepared speculative verify graph requires speculative verify metadata")
            if batch_size % self._spec_verify_width:
                raise ValueError("prepared speculative verify graph requires complete fixed-width sequences")
        elif metadata.is_prefill or metadata.is_chunked_prefill or metadata.is_spec_verify:
            raise ValueError("prepared ACL graph requires ordinary decode")
        if getattr(metadata, "prepared_attention_state", None) is None:
            raise ValueError("prepared ACL graph requires prepared metadata")
        block_table = metadata.block_table
        if block_table is None or block_table.dim() != 2:
            raise ValueError("prepared ACL graph requires a block table per metadata row")
        if self._block_draft:
            request_count = block_table.shape[0]
            query_width = batch_size // request_count if request_count else 0
            if (
                input_ids.dim() != 1
                or positions.shape != input_ids.shape
                or metadata.slot_mapping.shape != input_ids.shape
                or query_width == 0
                or batch_size != request_count * query_width
                or block_table.shape[1] == 0
                or not block_table.is_contiguous()
                or block_table.dtype != torch.int32
                or block_table.device != input_ids.device
                or metadata.q_cu_seq_lens_host_values != [query_width * (index + 1) for index in range(request_count)]
            ):
                raise ValueError(
                    "prepared block graph requires equal positive query widths and contiguous request rows"
                )
            kv_bounds = metadata.kv_seq_lens_host_values
            if kv_bounds is None or len(kv_bounds) != request_count or any(bound < query_width for bound in kv_bounds):
                raise ValueError("prepared block graph requires a conservative KV bound per request")
            if max(kv_bounds) > block_table.shape[1] * self.attention_backend.page_size:
                raise ValueError("prepared block graph KV bounds exceed block-table capacity")
            for name, tensor, rows in (
                ("positions", positions, batch_size),
                ("cache slots", metadata.slot_mapping, batch_size),
                ("query ends", metadata.q_cu_seq_lens, request_count),
                ("KV lengths", metadata.kv_seq_lens, request_count),
            ):
                if (
                    tensor is None
                    or tensor.ndim != 1
                    or tensor.numel() != rows
                    or not tensor.is_contiguous()
                    or tensor.dtype != torch.int32
                    or tensor.device != input_ids.device
                ):
                    raise ValueError(f"prepared block graph requires contiguous int32 {name} on the input device")
        else:
            self._validate_decode_token_layout(input_ids, positions, metadata.slot_mapping, block_table.shape[0])
        expected_rows = block_table.shape[0] if self._block_draft else batch_size
        if metadata.q_seq_lens.numel() != expected_rows or metadata.kv_seq_lens.numel() != expected_rows:
            raise ValueError("prepared ACL graph requires query and KV lengths per metadata row")
        state_rows = batch_size // self._spec_verify_width
        for name in ("linear_state_indices", "linear_state_read_indices", "linear_state_write_indices"):
            indices = getattr(metadata, name, None)
            if indices is not None and (
                indices.ndim != 1
                or indices.numel() != state_rows
                or indices.dtype != torch.int32
                or indices.device != input_ids.device
                or not indices.is_contiguous()
            ):
                raise ValueError(
                    f"prepared ACL graph requires contiguous int32 {name} with one state slot per sequence"
                )
        if self._spec_verify_width > 1:
            accepted = getattr(metadata, "num_accepted_tokens", None)
            state_indices = getattr(metadata, "linear_state_write_indices", None)
            if state_indices is None:
                state_indices = getattr(metadata, "linear_state_indices", None)
            if (
                state_indices is None
                or accepted is None
                or accepted.ndim != 1
                or accepted.numel() != state_rows
                or accepted.dtype != torch.int32
                or accepted.device != input_ids.device
                or not accepted.is_contiguous()
            ):
                raise ValueError("prepared speculative verify graph requires final int32 accepted counts per sequence")
        read_indices = getattr(metadata, "linear_state_read_indices", None)
        write_indices = getattr(metadata, "linear_state_write_indices", None)
        if write_indices is None:
            write_indices = getattr(metadata, "linear_state_indices", None)
        if read_indices is not None and (write_indices is None or read_indices.data_ptr() != write_indices.data_ptr()):
            raise ValueError("prepared ACL decode requires one shared linear-state read/write view")
        for name, tensor in (("hidden state", input_embedding), ("MTP top-k", mtp_topk_indices)):
            if tensor is not None and (
                tensor.dim() < 2 or tensor.shape[0] != batch_size or tensor.device != input_ids.device
            ):
                raise ValueError(f"prepared ACL graph requires {name} on the input device with one row per token")
        if mtp_topk_indices is not None and input_embedding is None:
            raise ValueError("prepared MTP top-k requires hidden state")
        if dspark_adaptive is not None:
            self._validate_dspark_graph_inputs(input_ids, positions, metadata, dspark_adaptive)

    @torch.inference_mode()
    def warmup_prepared(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None = None,
    ) -> None:
        self._validate_prepared_input(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, dspark_adaptive=dspark_adaptive
        )
        binding = self._prepared_binding(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, dspark_adaptive=dspark_adaptive
        )
        capacity_pages = kv_capacity_buckets(metadata.block_table.shape[1]) if self._device_kv_lengths else (0,)
        previous_count = len(self._prepared_graphs)
        for pages in capacity_pages:
            self._warmup_prepared_entry(
                input_ids,
                positions,
                metadata,
                input_embedding,
                mtp_topk_indices,
                binding=(*binding, pages) if self._device_kv_lengths else binding,
                capacity_pages=pages,
                eplb=eplb,
                dspark_adaptive=dspark_adaptive,
            )
        if self._device_kv_lengths and len(self._prepared_graphs) != previous_count:
            logger.info(
                "Python KV capacity warmup: runner=prepared model=%s rows=%d capacity_pages=%s "
                "capacity_tokens=%s binding_graphs=%d total_graphs=%d",
                type(self.model).__name__,
                input_ids.numel(),
                capacity_pages,
                tuple(pages * self.attention_backend.page_size for pages in capacity_pages),
                len(capacity_pages),
                len(self._prepared_graphs),
            )

    def _warmup_prepared_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None,
        *,
        binding: tuple[object, ...],
        capacity_pages: int,
        eplb: EplbRuntimeState | None,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None,
    ) -> None:
        if binding in self._prepared_graphs:
            return
        self._initialize_task_updates()
        if dspark_adaptive is not None:
            entry = _PreparedDSparkGraphEntry()
            entry.dspark_adaptive = dspark_adaptive
        else:
            entry = _PreparedBlockGraphEntry() if self._block_draft or self._device_kv_lengths else AclGraphEntry()
        entry.kv_capacity_pages = capacity_pages
        entry.batch_size = input_ids.numel()
        entry.graph = None
        entry.static_output = None
        entry.static_input_ids = input_ids
        entry.static_positions = positions
        entry.static_input_embedding = input_embedding
        entry.static_mtp_topk_indices = mtp_topk_indices
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({})
        entry.replay_logged = False
        mega_moe_token_mask = None
        if self._enable_mega_moe_token_mask:
            entry.mega_moe_token_counts = (entry.batch_size,) * self.dp_size
            # The borrowed-input runner also supports direct replay outside inference mode.
            with torch.inference_mode(False):
                mega_moe_token_mask = torch.ones(entry.batch_size * self.dp_size, dtype=torch.int8, device=self.device)
        # Retain the exact views independently of the mutable native Slot
        # metadata. Only the original model's inputs are captured here.
        entry.static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=metadata.slot_mapping,
            paged_kv_indptr=metadata.paged_kv_indptr,
            paged_kv_indices=metadata.paged_kv_indices,
            paged_kv_last_page_len=metadata.paged_kv_last_page_len,
            block_table=metadata.block_table,
            q_seq_lens=metadata.q_seq_lens,
            q_cu_seq_lens=metadata.q_cu_seq_lens,
            kv_seq_lens=metadata.kv_seq_lens,
            linear_state_indices=getattr(metadata, "linear_state_indices", None),
            linear_state_read_indices=getattr(metadata, "linear_state_read_indices", None),
            linear_state_write_indices=getattr(metadata, "linear_state_write_indices", None),
            has_initial_state=getattr(metadata, "has_initial_state", None),
            num_accepted_tokens=getattr(metadata, "num_accepted_tokens", None),
            is_spec_verify=self._spec_verify_width > 1,
            kv_seq_lens_host_values=list(metadata.kv_seq_lens_host_values),
            dp_execution_token_counts=tuple(getattr(metadata, "dp_execution_token_counts", ())),
            dp_is_decode=tuple(getattr(metadata, "dp_is_decode", ())),
            mega_moe_token_mask=mega_moe_token_mask,
            prepared_attention_state=(
                metadata.prepared_attention_state
                if getattr(self.attention_backend, "is_mla", False) or self._block_draft or self._device_kv_lengths
                else None
            ),
        )
        if entry.static_metadata.linear_state_read_indices is not None:
            # Decode reads and writes the same state. Keep one Python object
            # so metadata builders do not compare Device values on the Host.
            entry.static_metadata.linear_state_read_indices = (
                entry.static_metadata.linear_state_write_indices
                if entry.static_metadata.linear_state_write_indices is not None
                else entry.static_metadata.linear_state_indices
            )
        if self._block_draft or self._device_kv_lengths:
            # Device-length graphs capture every declared capacity tier. DSpark retains its
            # startup context bound. Live device lengths mask the valid prefix
            # within each graph's fixed FIA capacity.
            page_size = self.attention_backend.page_size
            request_count = metadata.block_table.shape[0]
            query_width = entry.batch_size // request_count
            if self._device_kv_lengths:
                pages = capacity_pages
            else:
                required_pages = (max(metadata.kv_seq_lens_host_values) + page_size - 1) // page_size
                minimum_pages = 16 + (query_width + page_size - 1) // page_size
                pages = min(1 << (max(required_pages, minimum_pages) - 1).bit_length(), metadata.block_table.shape[1])
            static = entry.static_metadata
            # Multiple narrowed rows are strided. Allocate only this small
            # metadata table at warmup; refresh it inside the captured forward
            # so replay has no extra Host launch and never scans full-context
            # padding just to preserve contiguity.
            entry.source_block_table = metadata.block_table[:, :pages]
            static.block_table = (
                entry.source_block_table.clone(memory_format=torch.contiguous_format)
                if self._device_kv_lengths
                else entry.source_block_table.contiguous()
            )
            static.q_cu_seq_lens_host_values = list(metadata.q_cu_seq_lens_host_values)
            static.kv_seq_lens_host_values = [pages * page_size] * request_count
            static.is_chunked_prefill = self._block_draft
            static.prepared_attention_state = self.attention_backend.prepare_metadata(static, device_kv_lengths=True)
        if mega_moe_token_mask is not None:
            self._fill_mega_moe_token_mask(entry, metadata.raw_dp_execution_token_counts)
        self._prepare_graph_eplb_state(entry, eplb, metadata, first_capture=True)
        self._prepare_attention(entry, entry.static_metadata)
        self._capture(entry, torch.npu.current_stream(self.device))
        self._prepared_graphs[binding] = entry

    def _resolve_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None = None,
    ) -> AclGraphEntry:
        self._validate_prepared_input(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, dspark_adaptive=dspark_adaptive
        )
        binding = self._prepared_binding(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, dspark_adaptive=dspark_adaptive
        )
        if self._device_kv_lengths:
            pages = kv_capacity_bucket(
                metadata.kv_seq_lens_host_values,
                self.attention_backend.page_size,
                metadata.block_table.shape[1],
            )
            binding = (*binding, pages)
        entry = self._prepared_graphs.get(binding)
        if entry is None:
            # Native admission already selected a captured bucket for every
            # rank. A missing Slot binding is an error, never a local fallback.
            raise RuntimeError("prepared ACL replay requires a warmed Slot binding")
        return entry

    def _forward_static(self, entry: AclGraphEntry) -> ModelExecutionOutput:
        if isinstance(entry, _PreparedDSparkGraphEntry):
            inputs = entry.dspark_adaptive
            metadata = entry.static_metadata
            torch.ops.xllm_ops.dspark_allocate_verify_prefixes_graph(
                inputs.arena, inputs.control, entry.batch_size, inputs.draft_width
            )
            torch.ops.xllm_ops.dspark_pack_verify_inputs_graph(
                inputs.anchors,
                inputs.arena,
                inputs.base_positions,
                inputs.base_kv_lengths,
                inputs.base_block_table,
                inputs.control,
                inputs.offsets,
                entry.static_input_ids,
                entry.static_positions,
                metadata.kv_seq_lens,
                metadata.slot_mapping,
                metadata.block_table,
                metadata.q_seq_lens,
                inputs.owners,
                inputs.columns,
                inputs.block_size,
                inputs.draft_width,
            )
        if isinstance(entry, _PreparedBlockGraphEntry):
            table = entry.static_metadata.block_table
            if table.data_ptr() != entry.source_block_table.data_ptr():
                table.copy_(entry.source_block_table)
        return super()._forward_static(entry)

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        dspark_adaptive: DSparkAdaptiveGraphInputs | None = None,
    ) -> ModelExecutionOutput:
        if layer_synchronizer is not None:
            raise ValueError("prepared ACL replay does not support layer synchronization")
        entry = self._resolve_entry(
            input_ids, positions, metadata, input_embedding, mtp_topk_indices, dspark_adaptive=dspark_adaptive
        )
        return self._execute_entry(entry, input_ids, metadata, input_embedding, mtp_topk_indices, eplb=eplb)

    def execute_block(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
    ) -> ModelExecutionOutput | None:
        """Return None only when the query exceeds the captured block capacity."""
        if layer_synchronizer is not None:
            raise ValueError("prepared ACL replay does not support layer synchronization")
        entry = self._resolve_entry(input_ids, positions, metadata, input_embedding, mtp_topk_indices)
        if max(metadata.kv_seq_lens_host_values) > entry.static_metadata.kv_seq_lens_host_values[0]:
            return None
        return self._execute_entry(entry, input_ids, metadata, input_embedding, mtp_topk_indices, eplb=eplb)

    def _execute_entry(
        self,
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None,
        *,
        eplb: EplbRuntimeState | None,
    ) -> ModelExecutionOutput:
        # Only serialized Launch installs shared backend state. This path
        # selects entry-owned buffers and refreshes its captured Host lists.
        if entry.static_metadata.mega_moe_token_mask is not None:
            # Counts precede native Slot padding; execution counts are physical.
            self._fill_mega_moe_token_mask(entry, metadata.raw_dp_execution_token_counts)
        self._prepare_graph_eplb_state(entry, eplb, metadata, first_capture=False)
        self._prepare_attention(
            entry, entry.static_metadata if self._block_draft or self._device_kv_lengths else metadata, replay=True
        )
        stream = torch.npu.current_stream(self.device)
        entry.graph.replay()
        self._update_after_replay(entry, stream)
        self.prepared_replays += 1
        if not entry.replay_logged:
            logger.info(
                "Python prepared ACL graph first replay: model=%s bucket=%d physical_rows=%d "
                "mtp_topk=%s hidden_shape=%s topk_shape=%s adaptive_dspark=%s capacity_pages=%d capacity_tokens=%d",
                type(self.model).__name__,
                entry.batch_size,
                input_ids.numel(),
                mtp_topk_indices is not None,
                None if input_embedding is None else tuple(input_embedding.shape),
                None if mtp_topk_indices is None else tuple(mtp_topk_indices.shape),
                isinstance(entry, _PreparedDSparkGraphEntry),
                entry.kv_capacity_pages,
                entry.kv_capacity_pages * self.attention_backend.page_size if entry.kv_capacity_pages else 0,
            )
            entry.replay_logged = True
        return entry.static_output

    def can_execute_block(self, input_ids: torch.Tensor, positions: torch.Tensor, metadata: AttentionMetadata) -> bool:
        """Only declared out-of-capacity queries use eager; missing bindings fail."""
        entry = self._resolve_entry(input_ids, positions, metadata)
        return max(metadata.kv_seq_lens_host_values) <= entry.static_metadata.kv_seq_lens_host_values[0]
