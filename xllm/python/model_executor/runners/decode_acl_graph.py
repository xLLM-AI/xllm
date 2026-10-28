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
* ``torch.npu.graph_task_update_begin/end`` to refresh FIA host params after
  replay is queued; captured external events gate each FIA task until its
  current parameters are ready.
* Static ``block_table`` and ``slot_mapping`` tensors so the graph records
  fixed addresses whose *contents* are updated via ``_fill_entry`` each step.
* C++ ACLNN ops (RMSNorm, SiLU, reshape_paged_cache) are used in both eager
  and capture modes — no PyTorch fallbacks needed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.attention.expanded_decode_metadata import (
    ExpandedDecodeMetadata,
    resolve_expanded_decode_metadata,
)
from xllm.python.attention.mla_index_context import is_compact_kpool_tail
from xllm.python.attention.paged_kv_metadata import (
    build_row_aligned_paged_kv_metadata,
    is_row_aligned_paged_kv,
)
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    EplbRuntimeState,
    LayerSynchronizer,
)
from xllm.python.model_executor.runners.acl_graph import AclGraphEntry, AclGraphRunner, StaticGraphAttentionMetadata
from xllm.python.model_executor.runners.base import ModelExecutionOutput
from xllm.python.model_executor.runners.decode_cuda_graph import (
    _decode_bucket,
)
from xllm.python.model_executor.runners.kv_capacity import kv_capacity_bucket, kv_capacity_buckets

_GraphKey = tuple[object, ...]


@dataclass(slots=True)
class DecodeAclGraphMetadata:
    """Resolved per-row KV and paging inputs shared within one execution."""

    block_table: torch.Tensor
    kv_seq_lens: torch.Tensor
    kv_seq_lens_host_values: list[int] | None
    paged_kv_indptr: torch.Tensor
    paged_kv_indices: torch.Tensor
    paged_kv_last_page_len: torch.Tensor


def _same_npu_stream(lhs: object, rhs: object) -> bool:
    """Compare the underlying stream, not just its Python wrapper."""
    if lhs is rhs:
        return True
    lhs_id = getattr(lhs, "npu_stream", None)
    rhs_id = getattr(rhs, "npu_stream", None)
    return lhs_id is not None and rhs_id is not None and lhs_id == rhs_id


def _uniform_span_width(spans: tuple[int, ...]) -> int | None:
    """Return the single span length when every entry matches, else None."""
    if not spans or any(length != spans[0] for length in spans):
        return None
    return spans[0]


class DecodeAclGraphRunner(AclGraphRunner):
    """Owns static inputs and lazily captures decode graphs on a private stream.

    Inputs are copied into entry-owned storage; MTP outputs are detached from
    replay buffers. Taskless graphs replay on the caller's stream.
    Slot-bound execution uses PreparedAclGraphRunner.
    """

    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
        max_batch: int,
        max_model_len: int,
        dp_size: int = 1,
        dp_rank: int = 0,
        num_decoding_tokens: int = 1,
        enable_mega_moe_token_mask: bool = False,
        *,
        is_spec_draft: bool = False,
        enable_graph_mode_decode_no_padding: bool = False,
        spec_verify_width: int = 1,
        device_kv_lengths: bool = False,
    ) -> None:
        super().__init__(model, attention_backend, device)
        self.dp_size = dp_size
        self.dp_rank = dp_rank
        self.max_batch = (max_batch + dp_size - 1) // dp_size
        self.max_model_len = max_model_len
        self.num_decoding_tokens = max(1, int(num_decoding_tokens))
        self._is_spec_draft = is_spec_draft
        self._enable_decode_no_padding = enable_graph_mode_decode_no_padding
        # New exact shapes can arrive while native input preparation copies
        # the next batch on another thread. Only this thread captures the
        # runner's owned static buffers, as in the unified MTP graph runner.
        if enable_graph_mode_decode_no_padding:
            self._capture_error_mode = "thread_local"
        self._serial_workspace_pool: list[torch.Tensor] | None = [] if enable_graph_mode_decode_no_padding else None
        if not 1 <= spec_verify_width <= 17 or (is_spec_draft and spec_verify_width != 1):
            raise ValueError("Speculative verify graph width must be in [1, 17], with width one for draft models")
        self._spec_verify_width = spec_verify_width
        if device_kv_lengths and dp_size != 1:
            raise ValueError("Device-length ACL graphs require DP size one")
        self._device_kv_lengths = device_kv_lengths
        self._has_index_cache = False
        self._has_linear_cache = False
        self._model_owns_linear_spec_verify = any(
            getattr(builder, "supports_linear_spec_verify_graph", False) for builder in self.execution_metadata_builders
        )
        self._uses_compressed_kpool_tail = False
        self._graphs: dict[_GraphKey, AclGraphEntry] = {}
        self._dp_graph_variants: dict[_GraphKey, int] = {}
        self._enable_mega_moe_token_mask = enable_mega_moe_token_mask
        self._paged_kv_indices_buffer: torch.Tensor | None = None
        self._max_blocks_per_sequence: int = 0
        self._stream: torch.npu.Stream | None = None
        self._update_done_event: torch.npu.Event | None = None
        self._update_done_recorded = False
        self._taskless_replay_stream: torch.npu.Stream | None = None

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        super().bind_layer_caches(layer_caches)
        self._has_index_cache = any(cache.index is not None for cache in layer_caches)
        self._has_linear_cache = any(cache.conv is not None or cache.ssm is not None for cache in layer_caches)
        # Compressed-tail signature is fixed after bind; cache it once instead
        # of rescanning every layer cache on each decode step.
        self._uses_compressed_kpool_tail = any(is_compact_kpool_tail(cache.kpool_tail) for cache in layer_caches)

    def can_execute(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> bool:
        return self.prepare_execution(input_ids, metadata, input_embedding, mtp_topk_indices) is not None

    def prepare_execution(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
    ) -> DecodeAclGraphMetadata | None:
        """Admit a step and return metadata reusable until its replay is queued.

        The caller owns this result for one execution only. Keep it local to
        that call so scheduler updates are decoded again on the next step.
        """
        if input_ids.dim() != 1:
            return None
        try:
            expanded_view = self._expanded_verify_view(metadata)
        except RuntimeError:
            return None
        is_expanded_spec_verify = expanded_view is not None
        if (metadata.is_prefill or metadata.is_chunked_prefill) and not is_expanded_spec_verify:
            return None
        if is_expanded_spec_verify and self._model_owns_linear_spec_verify:
            query_lens = self._expanded_query_lens(metadata)
            if _uniform_span_width(query_lens) != self.num_decoding_tokens:
                return None

        # Linear verify requires a backend recurrent-state protocol.
        linear_indices = getattr(metadata, "linear_state_indices", None)
        if (
            is_expanded_spec_verify
            and not self.attention_backend.is_mla
            and linear_indices is not None
            and linear_indices.numel() > 0
            and self._spec_verify_width == 1
            and not self.attention_backend.supports_linear_spec_verify_graph
            and not self._model_owns_linear_spec_verify
        ):
            return None

        batch_size = input_ids.numel()
        if self.dp_size > 1:
            execution_counts = getattr(metadata, "dp_execution_token_counts", None)
            if execution_counts is None or len(execution_counts) != self.dp_size:
                raise RuntimeError(
                    "DP decode step requires valid dp_execution_token_counts "
                    f"(got {execution_counts!r}, expected length {self.dp_size})."
                )
            if any(count <= 0 for count in execution_counts):
                raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")
            dp_is_decode = getattr(metadata, "dp_is_decode", None)
            if dp_is_decode is not None and not all(dp_is_decode):
                return None
            if max(execution_counts) > self.max_batch:
                return None
            padded_batch_size = self._padded_batch_size(batch_size, metadata)
        else:
            execution_counts = ()
            if batch_size > self.max_batch:
                return None
            padded_batch_size = self._padded_batch_size(batch_size, metadata)

        def reject_layout(reason: str) -> None:
            if self.dp_size > 1:
                raise RuntimeError(f"DP decode graph layout is incompatible: {reason}")
            return None

        if is_expanded_spec_verify:
            try:
                expanded_query_lens = self._expanded_query_lens(metadata)
            except RuntimeError:
                return reject_layout("expanded verify query lengths are not host-resident")
            if not expanded_query_lens:
                return reject_layout("expanded verify query lengths are missing")
            host_query_lens = getattr(metadata, "q_seq_lens_host", None)
            host_query_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
            if (host_query_lens is None or host_query_lens.numel() == 0) and not host_query_ends:
                return reject_layout("expanded verify host query lengths are missing")
            query_width = _uniform_span_width(expanded_query_lens)
            if query_width is None:
                return reject_layout("expanded verify query lengths are not uniform")
            if sum(expanded_query_lens) != batch_size:
                return reject_layout("expanded verify query lengths do not cover the decode batch")
            if padded_batch_size % query_width != 0:
                return reject_layout(
                    f"expanded verify bucket {padded_batch_size} is not divisible by width {query_width}"
                )
        if mtp_topk_indices is not None:
            if mtp_topk_indices.dim() < 2 or mtp_topk_indices.shape[0] != batch_size:
                raise ValueError(
                    "MTP top-k indices must have one leading row per decode token "
                    f"(got shape={tuple(mtp_topk_indices.shape)}, rows={batch_size})"
                )
        try:
            decode_metadata = self._decode_metadata(metadata, expanded_view)
            compatible = self._has_compatible_decode_metadata(
                input_ids, metadata, expanded_view, decode_metadata=decode_metadata
            )
        except RuntimeError:
            if self.dp_size > 1:
                raise
            return None
        if not (compatible and (input_embedding is None or input_embedding.shape[0] == batch_size)):
            return None
        # MTP spec-verify packs (num_speculative_tokens+1) token-rows per
        # sequence, so batch_size (input_ids.numel()) is in TOKENS while
        # self.max_batch (derived from max_seqs_per_batch) is in SEQS. Comparing
        # tokens to seqs wrongly rejects an 8-seq × 4-row verify (32 tokens,
        # 8 seqs ≤ 16) as bucket-overflow, forcing eager (~10x slower) under
        # concurrency. For expanded verify, gate on the actual SEQUENCE count
        # (linear_state_indices is per-seq) — the graph still captures/replays
        # at the full token bucket (keyed by padded_batch_size below), this
        # only fixes the graph-vs-eager admission decision.
        if is_expanded_spec_verify:
            lsi = getattr(metadata, "linear_state_indices", None)
            # The captured bucket is a power of two (1/2/4/8/16k) but the
            # static expanded-verify metadata (per-row q_cu + per-group
            # q_seq_lens) is only laid out when the PADDED bucket divides by
            # the spec width. Widths that are not powers of two (w=3 for
            # num_spec=2, w=5, ...) would capture an entry without the group
            # layout and silently degrade the in-graph verify semantics —
            # refuse graph admission there and fall back to eager instead.
            if lsi is not None and lsi.numel() > 0:
                seq_count = lsi.numel()
                if batch_size % seq_count == 0:
                    width = batch_size // seq_count
                    if padded_batch_size % width != 0:
                        return reject_layout(
                            f"expanded verify bucket {padded_batch_size} is not divisible by width {width}"
                        )
        if getattr(metadata, "kpool_query_lens", ()):
            # Same invariants as the capture path in _padded_kpool_query_lens;
            # reject instead of raising so admission can fall back to eager.
            try:
                self._padded_kpool_query_lens(metadata, batch_size, padded_batch_size)
            except RuntimeError as error:
                return reject_layout(str(error))
        if self.dp_size > 1:
            # DP ranks share one graph shape, so a missing or malformed
            # Execution counts cannot silently fall back to eager: divergent
            # execution paths across ranks would deadlock HCCL collectives.
            if execution_counts[self.dp_rank] != input_ids.shape[0]:
                raise RuntimeError(
                    "DP execution token count does not match the local input: "
                    f"rank={self.dp_rank}, rows={input_ids.shape[0]}, "
                    f"counts={execution_counts}"
                )
        return decode_metadata

    def _has_compatible_index_history(self, metadata: AttentionMetadata, block_table: torch.Tensor) -> bool:
        if not self._has_index_cache:
            return True
        max_kv_cap = getattr(self.attention_backend, "graph_index_history_max_kv", None)
        if max_kv_cap is None:
            return True
        page_size = self.attention_backend.page_size
        if max_kv_cap <= 0 or page_size <= 0 or max_kv_cap % page_size:
            return False
        if self.dp_size > 1:
            # Draft iterations advance KV without refreshing the shared host plan.
            if self._is_spec_draft:
                return False
            global_kv_lengths = getattr(metadata, "dp_global_kv_max_seq_lens", None)
            if global_kv_lengths is None:
                return False
            if not isinstance(global_kv_lengths, Sequence):
                raise RuntimeError("DP index-history lengths must be a host sequence")
            if not global_kv_lengths:
                return False
            if len(global_kv_lengths) != self.dp_size:
                raise RuntimeError(f"DP index-history lengths must contain {self.dp_size} entries")
            if any(
                not isinstance(length, int) or isinstance(length, bool) or length < 0 for length in global_kv_lengths
            ):
                raise RuntimeError("DP index-history lengths must be nonnegative host integers")
            # All ranks, including dummy ranks, decide from the same logical history.
            # Target verify appends rows after the scheduler publishes its plan.
            max_verify_kv = max(global_kv_lengths) + self.num_decoding_tokens - 1
            return max_verify_kv <= max_kv_cap

        return block_table.shape[1] * self._logical_page_size <= max_kv_cap

    def _padded_batch_size(self, batch_size: int, metadata: AttentionMetadata) -> int:
        # All DP ranks must use the same shape for captured MoE collectives.
        if self.dp_size > 1:
            execution_counts = metadata.dp_execution_token_counts
            if any(count <= 0 for count in execution_counts):
                raise RuntimeError(f"DP execution token counts must be positive, got {execution_counts}")
            batch_size = max(execution_counts)
        if batch_size > self.max_batch:
            raise ValueError("decode batch exceeds ACL graph capacity")
        if (
            (self._has_linear_cache or self._model_owns_linear_spec_verify) and metadata.is_spec_verify
        ) or self._spec_verify_width > 1:
            # Recurrent verify buckets must contain complete requests.
            width = self._spec_verify_width if self._spec_verify_width > 1 else self.num_decoding_tokens
            if batch_size % width:
                raise ValueError("linear verify graph requires complete fixed-width requests")
            if self._enable_decode_no_padding:
                return batch_size
            return min(_decode_bucket(batch_size // width), self.max_batch // width) * width
        if self._enable_decode_no_padding:
            return batch_size
        # Keep a final partial bucket instead of falling back to eager or
        # padding beyond the backend's declared row capacity.
        return min(_decode_bucket(batch_size), self.max_batch)

    @property
    def _logical_page_size(self) -> int:
        return int(getattr(self.attention_backend, "logical_page_size", self.attention_backend.page_size))

    def _decode_metadata(
        self,
        metadata: AttentionMetadata,
        expanded: ExpandedDecodeMetadata | None = None,
    ) -> DecodeAclGraphMetadata:
        """Return per-row KV and paging metadata for decode graph replay."""
        block_table = expanded.block_table if expanded is not None else metadata.block_table
        if block_table is None:
            multi_block_tables = getattr(metadata, "multi_block_tables", ()) or ()
            block_table = multi_block_tables[0] if multi_block_tables else None
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )
        if block_table is None or kv_seq_lens is None:
            raise RuntimeError("decode graph requires block and KV metadata")
        block_table = block_table.to(device=kv_seq_lens.device, dtype=torch.int32).contiguous()
        kv_seq_lens = kv_seq_lens.to(torch.int32)
        if getattr(self.attention_backend, "is_mla", False) and not getattr(
            self.attention_backend, "requires_host_kv_lengths", False
        ):
            # The C++ engine sizes kv_seq_lens_host_values by the global DP
            # batch, while block_table holds only this rank's local rows. When
            # the backend does not consume host KV lengths (e.g. sparse MLA),
            # drop them so the mismatched global length never reaches shape
            # validation.
            kv_seq_lens_host_values = None
        else:
            if not kv_seq_lens_host_values:
                raise RuntimeError("decode graph requires scheduler-provided host KV lengths")

        paged_kv_indptr = expanded.paged_kv_indptr if expanded is not None else metadata.paged_kv_indptr
        paged_kv_indices = expanded.paged_kv_indices if expanded is not None else metadata.paged_kv_indices
        paged_kv_last_page_len = (
            expanded.paged_kv_last_page_len if expanded is not None else metadata.paged_kv_last_page_len
        )
        # Speculative row builders provide row-aligned paging through the
        # regular attention metadata. Keep the fallback for older producers,
        # but steady-state replay reuses the packed tensors without Python
        # loops or extra H2D copies.
        _paged_missing = paged_kv_indptr is None or paged_kv_indices is None or paged_kv_last_page_len is None
        if _paged_missing and expanded is None:
            raise RuntimeError("decode graph requires paged KV metadata")
        if expanded is None and not is_row_aligned_paged_kv(
            paged_kv_indptr, paged_kv_last_page_len, block_table.shape[0]
        ):
            (
                paged_kv_indptr,
                paged_kv_indices,
                paged_kv_last_page_len,
            ) = build_row_aligned_paged_kv_metadata(
                block_table,
                kv_seq_lens_host_values,
                self._logical_page_size,
            )
        self._validate_decode_metadata_shapes(
            block_table,
            kv_seq_lens,
            kv_seq_lens_host_values,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )
        return DecodeAclGraphMetadata(
            block_table=block_table,
            kv_seq_lens=kv_seq_lens,
            kv_seq_lens_host_values=kv_seq_lens_host_values,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            paged_kv_last_page_len=paged_kv_last_page_len,
        )

    def _expanded_verify_view(self, metadata: AttentionMetadata) -> ExpandedDecodeMetadata | None:
        return resolve_expanded_decode_metadata(
            metadata,
            block_size=self._logical_page_size,
            allow_untyped_spec_verify=not self.attention_backend.is_mla or self._has_linear_cache,
        )

    @staticmethod
    def _expanded_query_lens(metadata: AttentionMetadata) -> tuple[int, ...]:
        """Return the logical query groups used by an expanded verify graph."""
        query_lens = getattr(metadata, "q_seq_lens_host", None)
        if query_lens is None or query_lens.numel() == 0:
            query_lens = getattr(metadata, "q_seq_lens", None)
        if query_lens is None or query_lens.numel() == 0:
            return ()
        if query_lens.device.type != "cpu":
            raise RuntimeError("expanded decode query lengths must be host-resident")
        return tuple(int(length) for length in query_lens.tolist())

    @staticmethod
    def _padded_expanded_query_lens(query_lens: tuple[int, ...], padded_batch_size: int) -> tuple[int, ...]:
        """Return a fixed group layout for one expanded graph bucket."""
        query_width = _uniform_span_width(query_lens)
        if query_width is None or query_width <= 0 or padded_batch_size % query_width != 0:
            return query_lens
        return (query_width,) * (padded_batch_size // query_width)

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
        row_count = block_table.shape[0]
        per_row_tensors = (
            ("kv_seq_lens", kv_seq_lens),
            ("paged_kv_last_page_len", paged_kv_last_page_len),
        )
        for name, tensor in per_row_tensors:
            if tensor.dim() != 1 or tensor.numel() != row_count:
                raise RuntimeError(f"decode {name} must contain one value per metadata row")
        if kv_seq_lens_host_values is not None and len(kv_seq_lens_host_values) != row_count:
            raise RuntimeError("decode kv_seq_lens_host_values must contain one value per metadata row")
        if paged_kv_indptr.dim() != 1 or paged_kv_indptr.numel() != row_count + 1:
            raise RuntimeError(
                "decode paged_kv_indptr must contain one offset per metadata row plus the terminal offset"
            )
        if paged_kv_indices.dim() != 1 or paged_kv_indices.numel() == 0:
            raise RuntimeError("decode paged_kv_indices must be a non-empty flat page list")

    def _has_compatible_decode_metadata(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        expanded: ExpandedDecodeMetadata | None = None,
        *,
        decode_metadata: DecodeAclGraphMetadata | None = None,
    ) -> bool:
        """Check that token tensors and decode metadata use the same row layout."""
        if decode_metadata is None:
            decode_metadata = self._decode_metadata(metadata, expanded)
        block_table = decode_metadata.block_table
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is None or slot_mapping.numel() == 0:
            # Admission only needs the host layout; defer slot upload until fill.
            host_slots = getattr(metadata, "new_cache_slots_host_values", None)
            if (
                input_ids.dim() != 1
                or input_ids.numel() != block_table.shape[0]
                or host_slots is None
                or len(host_slots) != input_ids.numel()
            ):
                return False
        else:
            self._validate_decode_token_layout(input_ids, None, slot_mapping, block_table.shape[0])
        batch_size = input_ids.numel()
        is_expanded = expanded is not None
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
        # Pure MLA does not read recurrent-state metadata. The scheduler keeps
        # one placeholder per request even when speculative verify expands the
        # attention inputs to N+1 rows. Preserve these checks for hybrid models
        # that actually bind a recurrent cache and for non-MLA backends.
        if self._has_linear_cache or not self.attention_backend.is_mla:
            has_initial_state = getattr(metadata, "has_initial_state", None)
            if has_initial_state is not None:
                state_count = has_initial_state.numel()
                if state_count != batch_size and not (
                    is_expanded and state_count > 0 and batch_size % state_count == 0
                ):
                    return False
            linear_idx = getattr(metadata, "linear_state_indices", None)
            if (
                linear_idx is not None
                and linear_idx.numel() != batch_size
                and not (is_expanded and linear_idx.numel() > 0 and batch_size % linear_idx.numel() == 0)
            ):
                return False
        return self._has_compatible_index_history(metadata, block_table)

    @staticmethod
    def _cumulative_lengths(
        sequence_lengths: torch.Tensor,
        cumulative_lengths: torch.Tensor | None,
    ) -> torch.Tensor:
        """Normalize NPU sequence ends to a cumulative tensor with a zero."""
        batch_size = sequence_lengths.numel()
        if cumulative_lengths is None:
            cumulative_lengths = torch.cumsum(sequence_lengths, dim=0, dtype=torch.int32)
        else:
            cumulative_lengths = cumulative_lengths.to(torch.int32)
        if cumulative_lengths.numel() not in (batch_size, batch_size + 1):
            raise RuntimeError(
                "cumulative sequence lengths must contain either one value per "
                "sequence or a leading zero plus one value per sequence"
            )
        if cumulative_lengths.numel() == batch_size:
            cumulative_lengths = torch.cat(
                (torch.zeros(1, dtype=torch.int32, device=cumulative_lengths.device), cumulative_lengths)
            )
        return cumulative_lengths

    def _resolve_graph_key(
        self,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        padded_batch_size: int,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None,
        expanded: ExpandedDecodeMetadata | None,
        *,
        kv_capacity_pages: int = 0,
        decode_metadata: DecodeAclGraphMetadata | None = None,
    ) -> tuple[_GraphKey, tuple[int, ...], tuple[int, ...]]:
        """Build the graph key plus the span layouts it was derived from."""
        batch_size = input_ids.shape[0]
        is_expanded = expanded is not None
        kpool_query_lens = self._padded_kpool_query_lens(metadata, batch_size, padded_batch_size)
        expanded_query_lens = (
            self._padded_expanded_query_lens(self._expanded_query_lens(metadata), padded_batch_size)
            if is_expanded
            else ()
        )
        graph_key = self._graph_key(
            padded_batch_size,
            is_expanded,
            input_embedding,
            mtp_topk_indices,
            kpool_query_lens=kpool_query_lens,
            expanded_query_lens=expanded_query_lens,
        )
        if self._device_kv_lengths:
            if not kv_capacity_pages:
                kv_capacity_pages = self._capacity_pages(
                    self._decode_metadata(metadata, expanded) if decode_metadata is None else decode_metadata
                )
            graph_key = (*graph_key, kv_capacity_pages)
        return self._synchronize_dp_graph_key(graph_key, input_ids), kpool_query_lens, expanded_query_lens

    def _capacity_page_limit(self) -> int:
        return (self.max_model_len + self._logical_page_size - 1) // self._logical_page_size + 1

    def _capacity_pages(self, metadata: DecodeAclGraphMetadata) -> int:
        pages = kv_capacity_bucket(
            metadata.kv_seq_lens_host_values, self._logical_page_size, self._capacity_page_limit()
        )
        if max(metadata.kv_seq_lens_host_values) > metadata.block_table.shape[1] * self._logical_page_size:
            raise ValueError("Host KV upper bound exceeds the source block table")
        return pages

    def warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        eplb: EplbRuntimeState | None = None,
        decode_metadata: DecodeAclGraphMetadata | None = None,
        is_graph_warmup: bool = False,
    ) -> _GraphKey:
        batch_size = input_ids.shape[0]
        padded_batch_size = self._padded_batch_size(batch_size, metadata)

        expanded = self._expanded_verify_view(metadata)
        graph_key, _, _ = self._resolve_graph_key(
            input_ids,
            metadata,
            padded_batch_size,
            input_embedding,
            mtp_topk_indices,
            expanded,
            decode_metadata=decode_metadata,
        )
        if self._device_kv_lengths:
            if not is_graph_warmup:
                if graph_key not in self._graphs:
                    raise RuntimeError("KV capacity graph was not captured during startup warmup")
                return graph_key
            capacity_pages = kv_capacity_buckets(self._capacity_page_limit())
            previous_count = len(self._graphs)
            for pages in capacity_pages:
                capacity_key, _, _ = self._resolve_graph_key(
                    input_ids,
                    metadata,
                    padded_batch_size,
                    input_embedding,
                    mtp_topk_indices,
                    expanded,
                    kv_capacity_pages=pages,
                    decode_metadata=decode_metadata,
                )
                if capacity_key in self._graphs:
                    continue
                self._prepare_graph_entry(
                    input_ids,
                    positions,
                    metadata,
                    input_embedding,
                    mtp_topk_indices,
                    graph_key=capacity_key,
                    eplb=eplb,
                    expanded=expanded,
                    decode_metadata=decode_metadata,
                    kv_capacity_pages=pages,
                    allow_capacity_capture=True,
                )
            if len(self._graphs) != previous_count:
                logger.info(
                    "Python KV capacity warmup: runner=ordinary model=%s rows=%d capacity_pages=%s "
                    "capacity_tokens=%s binding_graphs=%d total_graphs=%d",
                    type(self.model).__name__,
                    padded_batch_size,
                    capacity_pages,
                    tuple(pages * self._logical_page_size for pages in capacity_pages),
                    len(capacity_pages),
                    len(self._graphs),
                )
            return graph_key
        if not self._synchronize_dp_graph_presence(graph_key, input_ids):
            return graph_key

        self._graphs.pop(graph_key, None)

        decode_kwargs = {} if decode_metadata is None else {"decode_metadata": decode_metadata}
        self._prepare_graph_entry(
            input_ids,
            positions,
            metadata,
            input_embedding,
            mtp_topk_indices,
            graph_key=graph_key,
            eplb=eplb,
            expanded=expanded,
            **decode_kwargs,
        )
        return graph_key

    def _synchronize_dp_graph_key(
        self,
        local_key: _GraphKey,
        input_ids: torch.Tensor,
    ) -> _GraphKey:
        """Keep graphs captured with different peer variants in separate entries.

        Empty DP ranks omit MTP embeddings and top-k inputs. A local variant
        can therefore participate in several distinct collective captures.
        Local cache presence alone cannot establish that the peers are about
        to replay graphs captured together: both entries can exist but belong
        to different captures. Include every DP rank's variant in the key.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return local_key
        variant = self._dp_graph_variants.setdefault(local_key, len(self._dp_graph_variants) + 1)
        local_variant = torch.tensor([variant], dtype=torch.int32, device=input_ids.device)
        from xllm.python import distributed

        variants = distributed.all_gather(local_variant, dim=0, world_size=self.dp_size, group_name="dp")
        return (*local_key, tuple(variants.cpu().tolist()))

    def _synchronize_dp_graph_presence(
        self,
        graph_key: _GraphKey,
        input_ids: torch.Tensor,
    ) -> bool:
        """Make DP ranks capture their local graph variants together.

        A rank can enter a request with a warmup graph left over from startup,
        while another rank saw an empty shard and has to capture lazily. Letting
        the former replay while the latter captures makes HCCL graph collectives
        diverge. Synchronize the cache-presence bit before deciding whether to
        reuse the entry; if any rank is missing, all ranks recapture it. Local
        signatures can differ for empty ranks (embedding/top-k are absent),
        so a cached local key must still participate on every call.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return graph_key not in self._graphs
        presence = torch.tensor(
            [int(graph_key in self._graphs)],
            dtype=torch.int32,
            device=input_ids.device,
        )
        from xllm.python import distributed

        distributed.all_reduce_(presence, group_name="dp")
        return int(presence.item()) != self.dp_size

    def _synchronize_dp_graph_execute(self, input_ids: torch.Tensor) -> None:
        """Keep graph replay starts ordered after the warmup collective.

        ``warmup`` has already resolved the paired graph key. Repeating the
        variant all-gather in ``execute`` would copy the same NPU tensor back
        to the host on every decode step. A one-value reduction keeps the
        cross-rank launch barrier without rebuilding the Python cache key.
        """
        if self.dp_size <= 1 or not dist.is_initialized():
            return
        ready = torch.ones(1, dtype=torch.int32, device=input_ids.device)
        from xllm.python import distributed

        distributed.all_reduce_(ready, group_name="dp")

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _GraphKey | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        decode_metadata: DecodeAclGraphMetadata | None = None,
    ) -> ModelExecutionOutput:
        batch_size = input_ids.shape[0]
        if graph_key is not None:
            self._synchronize_dp_graph_execute(input_ids)
        prepare_kwargs = {} if decode_metadata is None else {"decode_metadata": decode_metadata}
        expanded = self._expanded_verify_view(metadata)
        if expanded is not None:
            prepare_kwargs["expanded"] = expanded
        entry = self._prepare_graph_entry(
            input_ids,
            positions,
            metadata,
            input_embedding,
            mtp_topk_indices,
            graph_key=graph_key,
            eplb=eplb,
            **prepare_kwargs,
        )

        assert self._stream is not None
        current_stream = torch.npu.current_stream()
        has_task_updates = bool(entry.graph_tasks)
        replay_stream = self._stream if has_task_updates else current_stream
        if has_task_updates:
            self._stream.wait_stream(current_stream)
        with torch.npu.stream(replay_stream):
            entry.graph.replay()
            output = self._slice_output(
                entry.static_output,
                batch_size,
                detach=(
                    self.num_decoding_tokens > 1
                    or self._is_spec_draft
                    or input_embedding is not None
                    or mtp_topk_indices is not None
                ),
            )

        if not getattr(entry, "replay_logged", False):
            logger.info(
                "Python ACL graph first replay: model=%s bucket=%d rows=%d mtp_topk=%s capacity_pages=%d capacity_tokens=%d",
                type(self.model).__name__,
                entry.batch_size,
                batch_size,
                mtp_topk_indices is not None,
                entry.kv_capacity_pages,
                entry.kv_capacity_pages * self._logical_page_size,
            )
            entry.replay_logged = True

        if has_task_updates:
            self._initialize_task_updates()
            if self._update_done_event is None:
                self._update_done_event = torch.npu.Event()
            self._update_after_replay(entry, replay_stream, self._update_done_event)
            self._taskless_replay_stream = None
            current_stream.wait_stream(replay_stream)
        else:
            self._taskless_replay_stream = replay_stream
        self._update_done_recorded = has_task_updates
        # The graph replay waits on each task's external event, but the task
        # update itself runs on a separate stream.  Schedule-overlap returns
        # to C++ without a device-wide synchronize, so the next MTP draft or
        # target graph can otherwise start while this runner is still updating
        # FIA/MLA task parameters.  Order the caller's compute stream after
        # those updates before exposing the output to the next model stage.
        if has_task_updates:
            current_stream.wait_event(self._update_done_event)
        return output

    def _prepare_graph_entry(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        graph_key: _GraphKey | None = None,
        eplb: EplbRuntimeState | None = None,
        expanded: ExpandedDecodeMetadata | None = None,
        decode_metadata: DecodeAclGraphMetadata | None = None,
        kv_capacity_pages: int = 0,
        allow_capacity_capture: bool = False,
    ) -> AclGraphEntry:
        batch_size = input_ids.shape[0]
        padded_batch_size = self._padded_batch_size(batch_size, metadata)

        if expanded is None:
            expanded = self._expanded_verify_view(metadata)
        if self._device_kv_lengths:
            if decode_metadata is None:
                decode_metadata = self._decode_metadata(metadata, expanded)
            selected_pages = self._capacity_pages(decode_metadata)
            if not kv_capacity_pages:
                kv_capacity_pages = selected_pages
            if kv_capacity_pages not in kv_capacity_buckets(self._capacity_page_limit()):
                raise ValueError("KV capacity graph has an unsupported KV capacity")
            if not allow_capacity_capture and selected_pages != kv_capacity_pages:
                raise ValueError("KV capacity graph capacity does not match the current Host upper bound")
        resolved_spans: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        if graph_key is None:
            graph_key, kpool_spans, expanded_spans = self._resolve_graph_key(
                input_ids,
                metadata,
                padded_batch_size,
                input_embedding,
                mtp_topk_indices,
                expanded,
                kv_capacity_pages=kv_capacity_pages,
                decode_metadata=decode_metadata,
            )
            resolved_spans = (kpool_spans, expanded_spans)
        if self._device_kv_lengths and graph_key[-1] != kv_capacity_pages:
            raise ValueError("KV capacity graph key differs from its metadata")
        entry = self._graphs.get(graph_key)
        first_capture = entry is None
        decode_kwargs = {} if decode_metadata is None else {"decode_metadata": decode_metadata}
        if first_capture:
            if self._device_kv_lengths and not allow_capacity_capture:
                raise RuntimeError("KV capacity graph was not captured during startup warmup")
            entry = self._allocate_entry(
                padded_batch_size,
                input_ids,
                metadata,
                mtp_topk_indices,
                expanded=expanded,
                kpool_query_lens=None if resolved_spans is None else resolved_spans[0],
                expanded_query_lens=None if resolved_spans is None else resolved_spans[1],
                **({"kv_capacity_pages": kv_capacity_pages} if self._device_kv_lengths else {}),
                **decode_kwargs,
            )
            self._graphs[graph_key] = entry

        if self._stream is None:
            self._stream = torch.npu.Stream(device=input_ids.device)

        # The previous replay may still be reading the capture buffers on the
        # graph stream while the scheduler prepares the next step on the
        # current stream.  Wait before mutating static inputs/metadata; the
        # later graph-stream wait only orders the new replay after these
        # writes and cannot protect this earlier update.
        if self._update_done_recorded:
            # Task updates run on a separate stream and read the same static
            # metadata tensors that _fill_entry mutates.  Waiting only for the
            # graph replay leaves that read/write pair unordered when the
            # scheduler advances quickly under overlap.
            assert self._update_done_event is not None
            torch.npu.current_stream().wait_event(self._update_done_event)

        current_stream = torch.npu.current_stream()
        if self._taskless_replay_stream is not None and not _same_npu_stream(
            current_stream, self._taskless_replay_stream
        ):
            current_stream.wait_stream(self._taskless_replay_stream)

        self._prepare_graph_eplb_state(entry, eplb, metadata, first_capture=first_capture)

        self._fill_entry(
            entry,
            input_ids,
            positions,
            metadata,
            batch_size,
            input_embedding,
            mtp_topk_indices,
            expanded=expanded,
            **decode_kwargs,
        )

        capture_linear_state = first_capture and self._has_linear_cache
        if capture_linear_state:
            current_stream.wait_stream(self._stream)
            # Slot zero is reserved for padding. Only scratch checkpoints are
            # initialized; the request metadata and live state stay untouched.
            for cache in self.layer_caches:
                if cache.conv is not None and cache.ssm is not None:
                    if cache.conv.shape[0] <= 0 or cache.ssm.shape[0] % cache.conv.shape[0] != 0:
                        raise RuntimeError("linear SSM cache does not align with state slots")
                    stride = cache.ssm.shape[0] // cache.conv.shape[0]
                    cache.conv[0].zero_()
                    cache.ssm[:stride].zero_()
            entry.static_metadata.linear_state_indices.zero_()
            accepted = entry.static_metadata.num_accepted_tokens
            if accepted is not None:
                accepted.fill_(1)
                entry.static_metadata.num_accepted_tokens_host_values = (1,) * accepted.numel()

        self._prepare_attention(entry, entry.static_metadata, replay=not first_capture)

        if first_capture:
            self._capture(entry, self._stream)
            if capture_linear_state:
                # Capture reads the persistent state indices. Restore real
                # metadata only after capture, then refresh derived tables in place.
                current_stream.wait_stream(self._stream)
                self._fill_entry(
                    entry,
                    input_ids,
                    positions,
                    metadata,
                    batch_size,
                    input_embedding,
                    mtp_topk_indices,
                    expanded=expanded,
                    **decode_kwargs,
                )
                self._prepare_attention(entry, entry.static_metadata, replay=True)
        return entry

    @staticmethod
    def _graph_key(
        padded_batch_size: int,
        is_expanded: bool,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        kpool_query_lens: tuple[int, ...] = (),
        expanded_query_lens: tuple[int, ...] = (),
    ) -> _GraphKey:
        """Return the key for a shape- and metadata-specific graph."""
        topk_signature = (None, None, None)
        if mtp_topk_indices is not None:
            topk_signature = (
                mtp_topk_indices.dtype,
                mtp_topk_indices.device,
                tuple(mtp_topk_indices.shape[1:]),
            )
        input_signature = (
            (input_embedding.dtype, input_embedding.device, tuple(input_embedding.shape[1:]))
            if input_embedding is not None
            else (None, None, None)
        )
        return (
            padded_batch_size,
            is_expanded,
            *input_signature,
            *topk_signature,
            kpool_query_lens,
            expanded_query_lens,
        )

    def _padded_kpool_query_lens(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
        padded_num_tokens: int,
    ) -> tuple[int, ...]:
        """Return uniform KPool request spans covering the graph bucket."""
        query_lens = tuple(int(length) for length in getattr(metadata, "kpool_query_lens", ()))
        if not query_lens:
            if not self._uses_compressed_kpool_tail:
                return ()
            query_lens = (1,) * num_tokens
        if any(length <= 0 for length in query_lens):
            raise RuntimeError(f"KPool graph query spans must be positive: {query_lens}")
        covered_tokens = sum(query_lens)
        if covered_tokens != num_tokens:
            raise RuntimeError(
                f"KPool graph query spans must cover every input token: covered={covered_tokens}, tokens={num_tokens}"
            )
        query_width = _uniform_span_width(query_lens)
        if query_width is None:
            raise RuntimeError(f"KPool ACL graph requires uniform request spans: {query_lens}")
        if padded_num_tokens % query_width != 0:
            raise RuntimeError(
                "KPool ACL graph token bucket must be divisible by its request width: "
                f"bucket={padded_num_tokens}, width={query_width}"
            )
        return query_lens + (query_width,) * ((padded_num_tokens - num_tokens) // query_width)

    def _allocate_entry(
        self,
        padded_batch_size: int,
        input_ids: torch.Tensor,
        metadata: AttentionMetadata,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        expanded: ExpandedDecodeMetadata | None = None,
        kpool_query_lens: tuple[int, ...] | None = None,
        expanded_query_lens: tuple[int, ...] | None = None,
        decode_metadata: DecodeAclGraphMetadata | None = None,
        kv_capacity_pages: int = 0,
    ) -> AclGraphEntry:
        device = input_ids.device
        if expanded is None:
            expanded = self._expanded_verify_view(metadata)
        is_expanded = expanded is not None
        has_linear_verify_state = is_expanded and (
            self._has_linear_cache or self._model_owns_linear_spec_verify or self._spec_verify_width > 1
        )
        if decode_metadata is None:
            decode_metadata = self._decode_metadata(metadata, expanded)
        paged_kv_indptr = decode_metadata.paged_kv_indptr
        paged_kv_indices = decode_metadata.paged_kv_indices
        paged_kv_last_page_len = decode_metadata.paged_kv_last_page_len
        if self._paged_kv_indices_buffer is None:
            page_size = self._logical_page_size
            # Reserve the scheduler's lookahead page, including for the MTP
            # draft executor, which reports one decoding token per request.
            max_blocks_per_sequence = (self.max_model_len + page_size - 1) // page_size + 1
            self._paged_kv_indices_buffer = torch.zeros(
                self.max_batch * max_blocks_per_sequence,
                dtype=paged_kv_indices.dtype,
                device=device,
            )
            self._max_blocks_per_sequence = max_blocks_per_sequence

        static_block_table = torch.zeros(
            padded_batch_size,
            kv_capacity_pages or self._max_blocks_per_sequence,
            dtype=torch.int32,
            device=device,
        )

        entry = AclGraphEntry()
        entry.kv_capacity_pages = kv_capacity_pages
        entry.batch_size = padded_batch_size
        entry.graph = None
        entry.static_output = None
        entry.graph_tasks = []
        entry.execution_state = AclGraphExecutionState({}, serial_workspace_pool=self._serial_workspace_pool)
        entry.replay_logged = False
        entry.static_input_ids = torch.zeros(padded_batch_size, dtype=input_ids.dtype, device=device)
        entry.static_positions = torch.zeros(padded_batch_size, dtype=torch.int32, device=device)
        entry.static_input_embedding = None
        entry.static_mtp_topk_indices = None
        if mtp_topk_indices is not None:
            entry.static_mtp_topk_indices = torch.zeros(
                (padded_batch_size, *mtp_topk_indices.shape[1:]),
                dtype=mtp_topk_indices.dtype,
                device=mtp_topk_indices.device,
            )
        if kpool_query_lens is None:
            kpool_query_lens = self._padded_kpool_query_lens(metadata, input_ids.numel(), padded_batch_size)
        if expanded_query_lens is None:
            expanded_query_lens = self._expanded_query_lens(metadata) if is_expanded else ()
        expanded_graph_query_lens = self._padded_expanded_query_lens(expanded_query_lens, padded_batch_size)
        # Device-length attention consumes one query per physical row, even
        # in padded draft buckets. GDN state slots remain logical sequences.
        graph_query_lens = (1,) * padded_batch_size if self._device_kv_lengths else expanded_graph_query_lens
        entry.static_metadata = StaticGraphAttentionMetadata(
            slot_mapping=torch.zeros(
                padded_batch_size,
                dtype=torch.int32,
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
            q_cu_seq_lens=(
                torch.arange(padded_batch_size + 1, dtype=torch.int32, device=device)
                if self._device_kv_lengths
                else torch.zeros(padded_batch_size + 1, dtype=torch.int32, device=device)
                if is_expanded
                else None
            ),
            q_cu_seq_lens_host_values=list(range(padded_batch_size + 1)) if self._device_kv_lengths else None,
            q_seq_lens=(torch.tensor(graph_query_lens, dtype=torch.int32, device=device) if graph_query_lens else None),
            q_seq_lens_host=(
                torch.tensor(expanded_graph_query_lens or graph_query_lens, dtype=torch.int32)
                if graph_query_lens
                else None
            ),
            linear_state_indices=torch.zeros(
                len(expanded_graph_query_lens) if has_linear_verify_state else padded_batch_size,
                dtype=torch.int32,
                device=device,
            ),
            num_accepted_tokens=(
                torch.ones(len(expanded_graph_query_lens), dtype=torch.int32, device=device)
                if has_linear_verify_state
                else None
            ),
            kv_seq_lens_host_values=[1] * padded_batch_size,
            block_table=static_block_table,
            dp_execution_token_counts=(padded_batch_size,) * self.dp_size if self.dp_size > 1 else (),
            dp_is_decode=tuple([1] * self.dp_size) if self.dp_size > 1 else (),
            is_dummy=bool(getattr(metadata, "is_dummy", False)),
            is_spec_verify=bool(getattr(metadata, "is_spec_verify", False) or is_expanded),
            kpool_query_lens=kpool_query_lens,
            kpool_query_lens_device=(
                torch.tensor(kpool_query_lens, dtype=torch.int32, device=device) if kpool_query_lens else None
            ),
            # One mask slot per padded row across all DP ranks. aclnnMegaMoe
            # dispatches the fixed graph shape over EP, so padded lanes must be
            # marked inactive or they are routed as real tokens.
            mega_moe_token_mask=(
                torch.zeros(
                    padded_batch_size * self.dp_size,
                    dtype=torch.int8,
                    device=device,
                )
                if self._enable_mega_moe_token_mask
                else None
            ),
        )
        entry.kv_seq_lens_delta = torch.empty(padded_batch_size, dtype=torch.int32, device=device)
        entry.expanded_query_cu_seq_lens = (
            torch.arange(padded_batch_size + 1, dtype=torch.int32, device=device) if is_expanded else None
        )
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
                    entry.static_metadata.kv_seq_lens_host_values
                    if getattr(
                        self.attention_backend,
                        "requires_host_kv_lengths",
                        False,
                    )
                    or not getattr(self.attention_backend, "is_mla", False)
                    else None
                ),
            )
        entry.static_metadata.multi_block_tables = self.attention_backend.create_graph_block_tables(
            padded_batch_size,
            self.max_model_len,
            self._max_blocks_per_sequence,
        )
        if entry.static_metadata.multi_block_tables:
            entry.static_metadata.new_cache_slots_host_values = []
            entry.static_metadata.dsa_graph_block_table_cols = self._max_blocks_per_sequence
            entry.static_metadata.dsa_graph_mode = True
        return entry

    def _fill_entry(
        self,
        entry: AclGraphEntry,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        batch_size: int,
        input_embedding: torch.Tensor | None,
        mtp_topk_indices: torch.Tensor | None = None,
        *,
        expanded: ExpandedDecodeMetadata | None = None,
        decode_metadata: DecodeAclGraphMetadata | None = None,
    ) -> None:
        padded_batch_size = entry.batch_size
        static_metadata = entry.static_metadata
        static_metadata.is_dummy = bool(getattr(metadata, "is_dummy", False))
        slot_mapping = self._effective_slot_mapping(metadata, batch_size, input_ids.device)
        kpool_query_lens = self._padded_kpool_query_lens(
            metadata,
            batch_size,
            padded_batch_size,
        )
        if kpool_query_lens != static_metadata.kpool_query_lens:
            raise RuntimeError(
                "KPool query-span layout changed for an existing ACL graph: "
                f"captured={static_metadata.kpool_query_lens}, current={kpool_query_lens}"
            )
        expanded_query_lens = self._expanded_query_lens(metadata) if expanded is not None else ()
        if expanded is not None and not expanded_query_lens:
            raise RuntimeError("expanded decode graph requires scheduler-provided query lengths")
        if self._device_kv_lengths and expanded is not None and sum(expanded_query_lens) != batch_size:
            raise RuntimeError("expanded decode query lengths must cover every input token")
        if expanded is not None and not self._device_kv_lengths:
            if static_metadata.q_seq_lens is None or static_metadata.q_seq_lens_host is None:
                raise RuntimeError("expanded decode graph is missing static query-length buffers")
            if len(expanded_query_lens) > static_metadata.q_seq_lens.numel():
                raise RuntimeError("expanded decode query groups exceed the captured graph layout")
            actual_query_lens_host = torch.tensor(expanded_query_lens, dtype=torch.int32)
            static_metadata.q_seq_lens_host[: len(expanded_query_lens)].copy_(actual_query_lens_host)
            static_metadata.q_seq_lens[: len(expanded_query_lens)].copy_(actual_query_lens_host)
            if len(expanded_query_lens) < static_metadata.q_seq_lens.numel():
                query_width = _uniform_span_width(expanded_query_lens)
                if query_width is None or query_width <= 0:
                    # Replay groups may be non-uniform; the captured padding
                    # rows then fall back to the per-row layout, and the
                    # scheduler contract keeps replay groups <= captured rows.
                    query_width = 1
                static_metadata.q_seq_lens[len(expanded_query_lens) :].fill_(query_width)
                static_metadata.q_seq_lens_host[len(expanded_query_lens) :].fill_(query_width)
        if decode_metadata is None:
            decode_metadata = self._decode_metadata(metadata, expanded)
        block_table = decode_metadata.block_table
        kv_seq_lens = decode_metadata.kv_seq_lens
        kv_seq_lens_host_values = decode_metadata.kv_seq_lens_host_values
        paged_kv_indptr = decode_metadata.paged_kv_indptr
        paged_kv_indices = decode_metadata.paged_kv_indices
        paged_kv_last_page_len = decode_metadata.paged_kv_last_page_len
        if entry.static_metadata.multi_block_tables:
            self._fill_graph_dsa_positions(entry, positions)
        if batch_size != block_table.shape[0]:
            raise RuntimeError("ACL graph decode batch size must match metadata sequences")
        self._validate_decode_token_layout(
            input_ids,
            positions,
            slot_mapping,
            block_table.shape[0],
        )
        is_expanded = expanded is not None
        cumulative_kv_seq_lens = self._cumulative_lengths(
            kv_seq_lens,
            None if is_expanded else metadata.kv_cu_seq_lens,
        )
        graph_positions = positions.to(torch.int32).contiguous()
        kernels.update_decode_graph_metadata(
            input_ids,
            graph_positions,
            slot_mapping,
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
        if expanded is not None and static_metadata.q_cu_seq_lens is not None and not self._device_kv_lengths:
            # Expanded attention has one query token per physical row. The
            # caller can still provide cumulative ends per logical request;
            # preserve that grouping in q_seq_lens, not in this row buffer.
            if entry.expanded_query_cu_seq_lens is None:
                raise RuntimeError("expanded decode graph is missing row query offsets")
            query_cu_seq_lens = entry.expanded_query_cu_seq_lens[: batch_size + 1]
            query_cu_seq_lens_host_values = list(range(batch_size + 1))
            static_metadata.q_cu_seq_lens[: batch_size + 1].copy_(query_cu_seq_lens)
            static_metadata.q_cu_seq_lens_host_values = query_cu_seq_lens_host_values
            if padded_batch_size > batch_size:
                terminal = query_cu_seq_lens_host_values[-1]
                static_metadata.q_cu_seq_lens[batch_size + 1 :].fill_(terminal)
                static_metadata.q_cu_seq_lens_host_values.extend([terminal] * (padded_batch_size - batch_size))
        # Pure MLA does not consume recurrent state. The scheduler can still carry
        # one linear-state index per request while GLM MTP adds a nonuniform
        # number of repair rows; those unrelated indices need no expansion.
        linear_state_indices = (
            None
            if self.attention_backend.is_mla and not self._has_linear_cache
            else getattr(metadata, "linear_state_indices", None)
        )
        if static_metadata.num_accepted_tokens is not None:
            self._fill_linear_verify_state(entry, metadata, expanded_query_lens, linear_state_indices)
            static_metadata.linear_state_write_indices = static_metadata.linear_state_indices
        elif linear_state_indices is not None:
            # MTP metadata carries one state index per sequence while the
            # decode graph input is expanded to one row per speculative token.
            # Repeat each sequence index across its token rows before copying
            # into the token-shaped persistent buffer.
            if linear_state_indices.numel() < batch_size:
                if batch_size % linear_state_indices.numel() != 0:
                    raise RuntimeError(
                        "ACL graph MTP linear_state_indices must divide the "
                        f"expanded token count (tokens={batch_size}, "
                        f"sequences={linear_state_indices.numel()})"
                    )
                repeat_count = batch_size // linear_state_indices.numel()
                linear_state_indices = linear_state_indices.repeat_interleave(repeat_count)
            if linear_state_indices.numel() < batch_size:
                raise RuntimeError(
                    "ACL graph linear_state_indices must contain one value "
                    f"per decode row (rows={batch_size}, "
                    f"values={linear_state_indices.numel()})"
                )
            static_metadata.linear_state_indices[:batch_size].copy_(linear_state_indices[:batch_size])
        if static_metadata.num_accepted_tokens is None and padded_batch_size > batch_size:
            static_metadata.linear_state_indices[batch_size:].zero_()
        self._fill_host_metadata(entry, kv_seq_lens_host_values, batch_size)
        self._fill_new_cache_slots_host_metadata(entry, metadata, batch_size)
        if self._device_kv_lengths:
            static_metadata.prepared_attention_state = self.attention_backend.prepare_metadata(
                static_metadata, device_kv_lengths=True
            )

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

        if mtp_topk_indices is not None:
            if entry.static_mtp_topk_indices is None:
                raise RuntimeError("MTP graph entry was captured without top-k input")
            if mtp_topk_indices.shape[0] != batch_size:
                raise ValueError("MTP top-k input row count must match decode batch")
            if mtp_topk_indices.shape[1:] != entry.static_mtp_topk_indices.shape[1:]:
                raise ValueError(
                    "MTP top-k input shape changed for an ACL graph bucket: "
                    f"got {tuple(mtp_topk_indices.shape[1:])}, "
                    f"expected {tuple(entry.static_mtp_topk_indices.shape[1:])}"
                )
            entry.static_mtp_topk_indices[:batch_size].copy_(mtp_topk_indices)
            if padded_batch_size > batch_size:
                entry.static_mtp_topk_indices[batch_size:].zero_()

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
            static_metadata.slot_mapping[batch_size:].fill_(-1)
            entry.kv_seq_lens_delta[batch_size:].fill_(1)

        self._fill_dsa_block_tables(
            static_metadata,
            metadata,
            batch_size,
        )
        if static_metadata.mega_moe_token_mask is not None:
            self._fill_mega_moe_token_mask(
                entry, (batch_size,) if self.dp_size == 1 else metadata.dp_execution_token_counts
            )

    @staticmethod
    def _fill_linear_verify_state(
        entry: AclGraphEntry,
        metadata: AttentionMetadata,
        query_lens: tuple[int, ...],
        state_indices: torch.Tensor | None,
    ) -> None:
        width = _uniform_span_width(query_lens)
        if width is None or width <= 1 or state_indices is None:
            raise ValueError("linear verify graph requires fixed-width requests and state indices")
        sequence_count = len(query_lens)
        if state_indices.numel() == sequence_count * width:
            state_indices = state_indices[::width]
        if state_indices.numel() != sequence_count:
            raise ValueError("linear verify state indices do not match the request count")
        accepted = metadata.num_accepted_tokens
        accepted_host = tuple(getattr(metadata, "num_accepted_tokens_host_values", ()))
        if accepted is None or accepted.ndim != 1 or accepted.numel() != sequence_count:
            raise ValueError("linear verify accepted counts must be sequence-scoped")
        if accepted_host and (
            len(accepted_host) != sequence_count or any(count < 1 or count > width for count in accepted_host)
        ):
            raise ValueError("linear verify accepted counts must be in [1, verify width]")
        if not accepted_host and (
            accepted.dtype != torch.int32 or accepted.device != state_indices.device or not accepted.is_contiguous()
        ):
            raise ValueError("linear verify requires final contiguous int32 accepted counts")
        static = entry.static_metadata
        static.linear_state_indices[:sequence_count].copy_(state_indices)
        static.linear_state_indices[sequence_count:].zero_()
        static.num_accepted_tokens[:sequence_count].copy_(accepted)
        static.num_accepted_tokens[sequence_count:].fill_(1)
        static.num_accepted_tokens_host_values = (
            accepted_host + (1,) * (static.num_accepted_tokens.numel() - sequence_count) if accepted_host else ()
        )

    def _fill_dsa_block_tables(
        self,
        static_metadata: StaticGraphAttentionMetadata,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Refresh every stable DSA manager table for the current decode."""
        static_tables = static_metadata.multi_block_tables
        if not static_tables:
            return
        source_tables = getattr(metadata, "multi_block_tables", ()) or ()
        if not source_tables:
            if getattr(metadata, "is_dummy", False):
                for table in static_tables:
                    table.zero_()
                return
            raise RuntimeError("ACL graph requires all captured manager block tables")
        if len(source_tables) < len(static_tables):
            raise RuntimeError(f"ACL graph DSA manager count changed: {len(source_tables)} < {len(static_tables)}")
        # MTP draft layers consume a prefix of the target's cache managers.
        source_tables = source_tables[: len(static_tables)]

        for manager_id, (target, source) in enumerate(zip(static_tables, source_tables, strict=True)):
            target.zero_()
            if source is None:
                continue
            if source.dim() != 2:
                raise RuntimeError(f"ACL graph DSA manager {manager_id} block table must be two-dimensional")
            source_rows = int(source.shape[0])
            if source_rows != batch_size:
                if source_rows <= 0 or batch_size % source_rows != 0:
                    raise RuntimeError(
                        f"ACL graph DSA manager {manager_id} row count "
                        f"{source_rows} does not match batch size {batch_size}"
                    )
                source = source.repeat_interleave(batch_size // source_rows, dim=0)
            if source.shape[1] > target.shape[1]:
                raise RuntimeError(
                    f"ACL graph DSA manager {manager_id} requires {source.shape[1]} "
                    f"columns, capacity is {target.shape[1]}"
                )
            target[:batch_size, : source.shape[1]].copy_(source[:batch_size])

    def _fill_graph_dsa_positions(
        self,
        entry: AclGraphEntry,
        positions: torch.Tensor,
    ) -> None:
        """Keep a stable DSA position tensor synced with the static bucket."""
        static_metadata = entry.static_metadata
        padded_batch_size = entry.batch_size
        if static_metadata.dsa_positions is None:
            static_metadata.dsa_positions = torch.zeros(
                padded_batch_size,
                dtype=torch.int64,
                device=entry.static_positions.device,
            )
        graph_positions = positions.to(device=entry.static_positions.device, dtype=torch.int64)
        copy_count = min(int(graph_positions.numel()), padded_batch_size)
        static_metadata.dsa_positions[:copy_count].copy_(graph_positions[:copy_count])
        if padded_batch_size > copy_count:
            static_metadata.dsa_positions[copy_count:].zero_()

    def _fill_host_metadata(
        self,
        entry: AclGraphEntry,
        kv_seq_lens: list[int] | None,
        batch_size: int,
    ) -> None:
        if getattr(self.attention_backend, "is_mla", False) and not getattr(
            self.attention_backend,
            "requires_host_kv_lengths",
            False,
        ):
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

    @staticmethod
    def _effective_slot_mapping(
        metadata: AttentionMetadata,
        token_count: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Resolve scheduler cache slots when DSV4 omits the top-level tensor."""
        slot_mapping = getattr(metadata, "slot_mapping", None)
        if slot_mapping is not None and slot_mapping.dim() == 1 and slot_mapping.numel() == token_count:
            return slot_mapping.to(device=device, dtype=torch.int32).contiguous()
        host_slots = getattr(metadata, "new_cache_slots_host_values", None)
        if host_slots is None or len(host_slots) != token_count:
            raise RuntimeError("ACL graph decode requires one scheduler cache slot per token")
        return torch.tensor(host_slots, dtype=torch.int32, device=device)

    @staticmethod
    def _fill_new_cache_slots_host_metadata(
        entry: AclGraphEntry,
        metadata: AttentionMetadata,
        batch_size: int,
    ) -> None:
        """Copy scheduler-resolved physical DSA slots into graph metadata."""
        source_slots = getattr(metadata, "new_cache_slots_host_values", None)
        target_slots = entry.static_metadata.new_cache_slots_host_values
        if target_slots is None:
            return
        if source_slots is None or (not source_slots and getattr(metadata, "is_dummy", False)):
            entry.static_metadata.new_cache_slots_host_values = []
            return
        if len(source_slots) < batch_size:
            raise RuntimeError("decode ACL graph requires one host cache slot per token")
        real_slots = [int(slot) for slot in source_slots[:batch_size]]
        padding_slots = [0] * (entry.batch_size - batch_size)
        entry.static_metadata.new_cache_slots_host_values = real_slots + padding_slots
