# Copyright 2025-2026 The xLLM Authors.
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

"""NPU attention backend using Fused-Infer-Attention (FIA).

Registers as the PrivateUse1 (NPU) backend for the Python model executor.
Prefill uses FIA TND with causal mask; decode uses FIA TND with block_table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch_npu

from xllm.python import distributed, kernels
from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
    MlaPreprocessContext,
)
from xllm.python.attention.expanded_decode_metadata import (
    resolve_expanded_decode_metadata,
)
from xllm.python.model_executor.cp_utils import cp_gather_kv
from xllm.python.model_executor.forward_context import (
    AclGraphTask,
    get_execution_buffer,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.model_executor.cp_utils import CpContext

# Ascend FIA sparse_mode values (see CANN aclnnFusedInferAttentionScore docs).
# 0: no compressed mask; used for single-query decode where no causal mask is
#    needed.
# 3: rightDownCausal; the causal mask is right-aligned to the KV tail, for the
#    prefix-cache / chunked-prefill case where q_len < kv_len so the new queries
#    attend the full cached prefix plus their own tokens (mode 2, leftUpCausal,
#    only aligns when q_len == kv_len and would misalign on a cache hit).
_SPARSE_MODE_NONE = 0
_SPARSE_MODE_RIGHT_DOWN_CAUSAL = 3

_HAS_FIA_V2 = hasattr(torch.ops.npu, "npu_fused_infer_attention_score_v2") and hasattr(
    torch_npu, "_npu_fused_infer_attention_score_v2_get_max_workspace"
)


@dataclass(frozen=True, slots=True)
class _PreparedMlaAttention:
    # Keep final input views and private Host values for one eager Slot.
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    actual_seq_q: torch.Tensor
    actual_seq_kv: torch.Tensor
    query_ends: list[int]
    kv_lengths: list[int]
    max_query_len: int
    max_seq_len: int
    is_prefill: bool
    is_chunked_prefill: bool
    has_kv_shard: bool = False


@dataclass(frozen=True, slots=True)
class _SfaPageLayout:
    source_page_ids: torch.Tensor
    target_page_ids: torch.Tensor
    block_table: torch.Tensor
    page_count: int


def _mla_graph_max_seqlen_k(
    block_table: torch.Tensor,
    page_size: int,
) -> int:
    """Return a replay-stable KV length bound for MLA graph metadata."""
    max_seqlen_k = int(block_table.shape[1]) * int(page_size)
    if max_seqlen_k <= 0:
        raise RuntimeError("MLA graph block-table capacity must be positive")
    return max_seqlen_k


def _build_stable_sfa_page_layout(
    materialized_block_table: torch.Tensor,
) -> _SfaPageLayout:
    """Build a deterministic, legacy-compatible SFA-only page layout."""
    if materialized_block_table.ndim != 2:
        raise RuntimeError("materialized SFA block table must be two-dimensional")

    valid_pages = materialized_block_table >= 0
    stable_block_table = torch.arange(
        materialized_block_table.numel(),
        dtype=torch.int32,
        device=materialized_block_table.device,
    ).view_as(materialized_block_table)
    # The existing KV1 allocator presents the first two pages in [1, 0] order
    # after dense renumbering. Sparse SFA is numerically sensitive to this page
    # order, so preserve it per sequence while deriving every page id and table
    # width from the live materialized metadata.
    if materialized_block_table.shape[1] > 1:
        swap_rows = valid_pages[:, 1]
        first_pages = stable_block_table[:, 0].clone()
        second_pages = stable_block_table[:, 1].clone()
        stable_block_table[:, 0] = torch.where(
            swap_rows,
            second_pages,
            first_pages,
        )
        stable_block_table[:, 1] = torch.where(
            swap_rows,
            first_pages,
            second_pages,
        )
    stable_block_table = torch.where(
        valid_pages,
        stable_block_table,
        torch.full_like(stable_block_table, -1),
    ).contiguous()
    source_page_ids = materialized_block_table.masked_select(valid_pages).to(torch.int64)
    target_page_ids = stable_block_table.masked_select(valid_pages).to(torch.int64)
    return _SfaPageLayout(
        source_page_ids=source_page_ids,
        target_page_ids=target_page_ids,
        block_table=stable_block_table,
        page_count=materialized_block_table.numel(),
    )


@dataclass(frozen=True, slots=True)
class _PreparedPagedAttention:
    block_table: torch.Tensor | None
    query_ends: list[int] | None
    actual_seq_q: list[int]
    actual_seq_kv: list[int]


class NpuPagedAttentionBackend(AttentionBackend):
    """NPU attention backend dispatching to npu_fused_infer_attention_score."""

    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        is_mla: bool,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.scale = scale
        self.sliding_window = sliding_window
        self.dtype = dtype
        self.device = device
        self._use_fia_v2 = _HAS_FIA_V2
        self._is_mla = is_mla
        self._uses_sparse_mla = False

        self._kv_caches: list[LayerCache] = []
        self._num_kv_blocks: int | None = None
        self._page_size: int | None = None
        self._metadata: AttentionMetadata | _PreparedMlaAttention | None = None
        self._graph_workspace: torch.Tensor | None = None
        self._graph_outputs: dict[int, torch.Tensor] = {}
        self._graph_lses: dict[int, torch.Tensor] = {}
        self._current_graph_output: torch.Tensor | None = None
        self._current_graph_lse: torch.Tensor | None = None
        self._use_expanded_decode = False
        self._block_table_i32: torch.Tensor | None = None
        self._actual_seq_lens: list[int] | None = None
        self._actual_seq_q: list[int] | torch.Tensor = []
        self._actual_seq_kv: list[int] | torch.Tensor = []
        self._mla_actual_seq_q: torch.Tensor | None = None
        self._mla_actual_seq_kv: torch.Tensor | None = None
        self._mla_actual_seq_q_host: list[int] | None = None
        self._mla_actual_seq_kv_host: list[int] | None = None
        self._mla_graph_workspaces: dict[tuple[int, ...], torch.Tensor] = {}
        self._mla_graph_outputs: dict[tuple[int, ...], torch.Tensor] = {}
        self._mla_graph_lses: dict[tuple[int, ...], torch.Tensor] = {}
        # CP MLA index contexts are requested once per decoder layer. Keep the
        # segment-row block-table view for the current forward so every layer
        # reuses the same contiguous tensor instead of launching another
        # index_select and allocation.
        self._mla_cp_block_tables: dict[tuple[int, int], torch.Tensor] = {}
        self._mla_quant_indexer_metadata: dict[tuple[object, ...], torch.Tensor] = {}
        self._mla_max_seqlen_q = 0
        self._mla_max_seqlen_k = 0
        self._kv_owner_representatives: torch.Tensor | None = None
        self._materialized_block_table: torch.Tensor | None = None
        self._sfa_page_layout: _SfaPageLayout | None = None
        self._causal_mask = (
            torch.triu(torch.ones(2048, 2048, dtype=torch.float32), 1).to(torch.int8).contiguous().to(device)
        )

    @property
    def num_kv_blocks(self) -> int:
        if self._num_kv_blocks is None:
            raise RuntimeError("full-attention KV caches are not bound")
        return self._num_kv_blocks

    @property
    def page_size(self) -> int:
        if self._page_size is None:
            raise RuntimeError("full-attention KV caches are not bound")
        return self._page_size

    @property
    def is_mla(self) -> bool:
        return self._is_mla

    @property
    def requires_host_kv_lengths(self) -> bool:
        """Whether ACL Graph replay must update FIA's host KV-length list."""
        return self._is_mla and not self._uses_sparse_mla

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        full_attention_caches = [
            (cache.key, cache.value) for cache in kv_caches if cache.key is not None and cache.value is not None
        ]
        if not full_attention_caches:
            raise RuntimeError("no full-attention KV cache is bound")

        page_sizes = {key.shape[1] for key, _ in full_attention_caches}
        if len(page_sizes) != 1:
            raise RuntimeError("full-attention layers use inconsistent page sizes")

        num_kv_blocks = {key.shape[0] for key, _ in full_attention_caches}
        if len(num_kv_blocks) != 1:
            raise RuntimeError("full-attention layers use inconsistent KV block counts")

        self._kv_caches = kv_caches
        self._page_size = page_sizes.pop()
        self._num_kv_blocks = num_kv_blocks.pop()
        self._uses_sparse_mla = self._is_mla and any(cache.index is not None for cache in kv_caches)

    @staticmethod
    def _query_sequence_ends(
        q_cu_seq_lens: torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor | None:
        """Accept both NPU q-cumulative layouts used by the runtime."""
        if q_cu_seq_lens is None:
            return None
        if q_cu_seq_lens.numel() == batch_size:
            return q_cu_seq_lens.to(torch.int32)
        if q_cu_seq_lens.numel() == batch_size + 1:
            return q_cu_seq_lens[1:].to(torch.int32)
        raise RuntimeError(
            "q cumulative sequence lengths must contain either one value per "
            "sequence or a leading zero plus one value per sequence"
        )

    @property
    def supports_prepared_metadata(self) -> bool:
        return True

    def prepare_metadata(self, metadata: AttentionMetadata) -> _PreparedPagedAttention | _PreparedMlaAttention:
        """Prepare one ordinary Slot without Device work or active-state writes."""
        expanded = getattr(metadata, "expanded_decode_metadata", None)
        if (
            getattr(metadata, "is_spec_verify", False)
            or getattr(metadata, "has_kv_shard", False)
            or (expanded is not None and expanded.enabled)
        ):
            raise ValueError("prepared paged metadata requires ordinary unsharded attention")
        q_lens = metadata.q_seq_lens
        if q_lens is None:
            raise ValueError("prepared paged metadata requires query lengths")
        batch_size = q_lens.numel()
        query_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
        if query_ends is None:
            raise ValueError("prepared paged metadata requires Host query ends")
        if len(query_ends) == batch_size + 1 and query_ends[0] == 0:
            query_ends = query_ends[1:]
        if len(query_ends) != batch_size:
            raise ValueError("prepared Host query ends must match the batch")
        block_table = metadata.block_table
        actual_seq_q: list[int] = []
        actual_seq_kv: list[int] = []
        if block_table is not None:
            if block_table.dtype != torch.int32 or not block_table.is_contiguous():
                raise ValueError("prepared block table must already be contiguous int32")
            if block_table.shape[0] != batch_size:
                raise ValueError("prepared block table must match the batch")
            kv_lengths = metadata.kv_seq_lens_host_values
            if kv_lengths is None or len(kv_lengths) != batch_size:
                raise ValueError("prepared Host KV lengths must match the batch")
            actual_seq_q = list(range(1, batch_size + 1))
            actual_seq_kv = list(kv_lengths)
        if self._is_mla:
            if block_table is None:
                raise ValueError("prepared MLA requires a block table for every forward type")
            query_tensor = metadata.q_cu_seq_lens
            kv_tensor = metadata.kv_seq_lens
            slot_mapping = metadata.slot_mapping
            for tensor in (query_tensor, kv_tensor, slot_mapping):
                if (
                    tensor is None
                    or tensor.dtype != torch.int32
                    or tensor.ndim != 1
                    or not tensor.is_contiguous()
                    or tensor.device != block_table.device
                ):
                    raise ValueError("prepared MLA requires contiguous int32 final input views")
            if query_tensor.numel() != batch_size or kv_tensor.numel() != batch_size:
                raise ValueError("prepared MLA lengths must have one entry per sequence")
            if not query_ends or query_ends[-1] != slot_mapping.numel():
                raise ValueError("prepared MLA query ends must describe all input tokens")
            query_lengths = [end - start for start, end in zip([0, *query_ends[:-1]], query_ends)]
            return _PreparedMlaAttention(
                block_table=block_table,
                slot_mapping=slot_mapping,
                actual_seq_q=query_tensor,
                actual_seq_kv=kv_tensor,
                query_ends=list(query_ends),
                kv_lengths=actual_seq_kv,
                max_query_len=max(query_lengths, default=0),
                max_seq_len=max(actual_seq_kv, default=0),
                is_prefill=metadata.is_prefill,
                is_chunked_prefill=metadata.is_chunked_prefill,
            )
        return _PreparedPagedAttention(block_table, list(query_ends), actual_seq_q, actual_seq_kv)

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        prepared = getattr(metadata, "prepared_attention_state", None)
        if prepared is not None and graph_mode:
            raise ValueError("prepared attention metadata requires eager execution")
        if isinstance(prepared, _PreparedMlaAttention):
            if not self._is_mla:
                raise ValueError("prepared MLA state requires MLA attention")
            self._metadata = prepared
            self._use_expanded_decode = False
            self._block_table_i32 = prepared.block_table
            self._actual_seq_lens = None
            self._actual_seq_q = []
            self._actual_seq_kv = []
            self._mla_actual_seq_q = prepared.actual_seq_q
            self._mla_actual_seq_kv = prepared.actual_seq_kv
            self._mla_actual_seq_q_host = prepared.query_ends if self.requires_host_kv_lengths else None
            self._mla_actual_seq_kv_host = prepared.kv_lengths if self.requires_host_kv_lengths else None
            self._mla_max_seqlen_q = prepared.max_query_len
            self._mla_max_seqlen_k = prepared.max_seq_len
            self._mla_quant_indexer_metadata.clear()
            self._kv_owner_representatives = None
            self._materialized_block_table = None
            self._sfa_page_layout = None
            return
        if prepared is not None:
            if self._is_mla or not isinstance(prepared, _PreparedPagedAttention):
                raise ValueError("prepared paged state requires ordinary attention")
            self._metadata = metadata
            self._use_expanded_decode = False
            self._block_table_i32 = prepared.block_table
            self._actual_seq_lens = prepared.query_ends
            self._actual_seq_q = prepared.actual_seq_q
            self._actual_seq_kv = prepared.actual_seq_kv
            return
        self._metadata = metadata
        expanded = resolve_expanded_decode_metadata(metadata, block_size=self.logical_page_size)
        self._use_expanded_decode = expanded is not None
        block_table = expanded.block_table if expanded is not None else metadata.block_table
        kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        kv_seq_lens_host_values = (
            expanded.kv_seq_lens_host_values
            if expanded is not None
            else getattr(metadata, "kv_seq_lens_host_values", None)
        )

        if block_table is not None:
            self._block_table_i32 = block_table.to(torch.int32)
            real_batch = block_table.shape[0]
        else:
            self._block_table_i32 = None
            real_batch = 0

        if self._use_expanded_decode or graph_mode or self._is_mla:
            self._actual_seq_lens = None
        elif metadata.q_cu_seq_lens is not None:
            q_seq_lens = getattr(metadata, "q_seq_lens", None)
            if q_seq_lens is not None:
                batch_size = q_seq_lens.numel()
            elif metadata.block_table is not None:
                batch_size = metadata.block_table.shape[0]
            else:
                batch_size = max(metadata.q_cu_seq_lens.numel() - 1, 0)
            host_query_ends = getattr(metadata, "q_cu_seq_lens_host_values", None)
            if host_query_ends:
                if len(host_query_ends) == batch_size + 1 and host_query_ends[0] == 0:
                    host_query_ends = host_query_ends[1:]
                if len(host_query_ends) != batch_size:
                    raise RuntimeError("host query ends must have one entry per sequence")
                self._actual_seq_lens = host_query_ends
            else:
                q_seq_ends = self._query_sequence_ends(
                    metadata.q_cu_seq_lens,
                    batch_size,
                )
                self._actual_seq_lens = q_seq_ends.cpu().tolist()
        else:
            self._actual_seq_lens = None

        if self._block_table_i32 is not None and not self._is_mla:
            if kv_seq_lens_host_values is None:
                raise RuntimeError("decode attention requires scheduler-provided host KV lengths")
            if len(kv_seq_lens_host_values) != real_batch:
                if len(kv_seq_lens_host_values) > real_batch:
                    kv_seq_lens_host_values = kv_seq_lens_host_values[:real_batch]
                else:
                    raise RuntimeError("host KV lengths must have one entry per block-table row")
            self._actual_seq_q: list[int] = list(range(1, real_batch + 1))
            self._actual_seq_kv: list[int] = list(kv_seq_lens_host_values)
        else:
            self._actual_seq_q = []
            self._actual_seq_kv = []

        if graph_mode and self._block_table_i32 is not None and not self._is_mla:
            graph_batch_size = self._block_table_i32.shape[0]
            graph_state = get_forward_context().execution_state
            if graph_state is None:
                if self._graph_workspace is None:
                    self._graph_workspace = self._allocate_graph_workspace(
                        graph_batch_size,
                        self._block_table_i32,
                    )
                self._current_graph_output = self._graph_outputs.setdefault(
                    graph_batch_size,
                    torch.empty(
                        graph_batch_size,
                        self.num_heads,
                        self.head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                self._current_graph_lse = self._graph_lses.setdefault(
                    graph_batch_size,
                    torch.empty(0, dtype=self.dtype, device=self.device),
                )
            else:
                self._graph_workspace = get_execution_buffer(
                    ("FIA_WORKSPACE", graph_batch_size),
                    lambda: self._allocate_graph_workspace(
                        graph_batch_size,
                        self._block_table_i32,
                    ),
                )
                self._current_graph_output = get_execution_buffer(
                    ("FIA_OUTPUT", graph_batch_size),
                    lambda: torch.empty(
                        graph_batch_size,
                        self.num_heads,
                        self.head_dim,
                        dtype=self.dtype,
                        device=self.device,
                    ),
                )
                self._current_graph_lse = get_execution_buffer(
                    ("FIA_LSE", graph_batch_size),
                    lambda: torch.empty(0, dtype=self.dtype, device=self.device),
                )

        # Pre-cache MLA (sparse SFA) seq-lens once per step; shared by
        # execute_mla / mla_index_context instead of re-derived per layer.
        self._mla_cp_block_tables.clear()
        self._mla_quant_indexer_metadata.clear()
        if self._is_mla and kv_seq_lens is not None:
            mla_device = kv_seq_lens.device
            actual_seq_kv = kv_seq_lens.to(torch.int32).to(mla_device)
            if self._use_expanded_decode:
                actual_seq_q = torch.arange(
                    1,
                    actual_seq_kv.numel() + 1,
                    dtype=torch.int32,
                    device=mla_device,
                )
            elif metadata.q_cu_seq_lens is not None:
                actual_seq_q = self._query_sequence_ends(
                    metadata.q_cu_seq_lens,
                    int(actual_seq_kv.numel()),
                ).to(mla_device)
            else:
                batch = kv_seq_lens.size(0)
                actual_seq_q = torch.arange(1, batch + 1, dtype=torch.int32, device=mla_device)
            if graph_mode:
                graph_batch = int(actual_seq_kv.numel())
                self._mla_actual_seq_q = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_Q", graph_batch),
                    lambda: torch.empty_like(actual_seq_q),
                )
                self._mla_actual_seq_kv = get_execution_buffer(
                    ("MLA_ACTUAL_SEQ_KV", graph_batch),
                    lambda: torch.empty_like(actual_seq_kv),
                )
                self._mla_actual_seq_q.copy_(actual_seq_q)
                self._mla_actual_seq_kv.copy_(actual_seq_kv)
            else:
                self._mla_actual_seq_q = actual_seq_q
                self._mla_actual_seq_kv = actual_seq_kv
            if self.requires_host_kv_lengths:
                if metadata.is_prefill or metadata.is_chunked_prefill:
                    self._mla_actual_seq_q_host = actual_seq_q.cpu().tolist()
                else:
                    self._mla_actual_seq_q_host = list(range(1, int(actual_seq_kv.numel()) + 1))
                if kv_seq_lens_host_values is not None:
                    self._mla_actual_seq_kv_host = list(kv_seq_lens_host_values)
                else:
                    self._mla_actual_seq_kv_host = actual_seq_kv.cpu().tolist()
            else:
                self._mla_actual_seq_q_host = None
                self._mla_actual_seq_kv_host = None
            if metadata.is_prefill or metadata.is_chunked_prefill:
                q_seq_lens = getattr(metadata, "q_seq_lens", None)
                if q_seq_lens is not None and q_seq_lens.numel() > 0:
                    self._mla_max_seqlen_q = int(q_seq_lens.max().item())
                else:
                    seq_starts = torch.cat([actual_seq_q.new_zeros(1), actual_seq_q[:-1]])
                    self._mla_max_seqlen_q = int((actual_seq_q - seq_starts).max().item())
            else:
                self._mla_max_seqlen_q = 1
            if graph_mode and self._block_table_i32 is not None:
                # Scalar tiling metadata must remain valid across graph replay.
                self._mla_max_seqlen_k = _mla_graph_max_seqlen_k(
                    self._block_table_i32,
                    self.logical_page_size,
                )
            elif kv_seq_lens_host_values:
                self._mla_max_seqlen_k = max(kv_seq_lens_host_values)
            else:
                self._mla_max_seqlen_k = int(actual_seq_kv.max().item())
        else:
            self._mla_actual_seq_q = None
            self._mla_actual_seq_kv = None
            self._mla_actual_seq_q_host = None
            self._mla_actual_seq_kv_host = None
            self._mla_max_seqlen_q = 0
            self._mla_max_seqlen_k = 0

        self._prepare_kv_shard_materialization(metadata)

    def _allocate_graph_workspace(
        self,
        graph_batch_size: int,
        block_table: torch.Tensor,
    ) -> torch.Tensor:
        block_size = self.page_size
        dummy_q = torch.empty(
            graph_batch_size,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        dummy_kv = torch.empty(
            self.num_kv_blocks,
            block_size,
            self.num_kv_heads * self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        if self._use_fia_v2:
            return torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                query=dummy_q,
                key=dummy_kv,
                value=dummy_kv,
                block_table=block_table,
                input_layout="TND",
                block_size=block_size,
                actual_seq_qlen=self._actual_seq_q,
                actual_seq_kvlen=self._actual_seq_kv,
                num_key_value_heads=self.num_kv_heads,
                num_query_heads=self.num_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                softmax_scale=self.scale,
                return_softmax_lse=False,
            )
        return torch_npu._npu_fused_infer_attention_score_get_max_workspace(
            query=dummy_q,
            key=dummy_kv,
            value=dummy_kv,
            block_table=block_table,
            input_layout="TND",
            block_size=block_size,
            actual_seq_lengths=self._actual_seq_q,
            actual_seq_lengths_kv=self._actual_seq_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            scale=self.scale,
            softmax_lse_flag=False,
        )

    def _prepare_kv_shard_materialization(self, metadata: AttentionMetadata) -> None:
        self._kv_owner_representatives = None
        self._materialized_block_table = None
        self._sfa_page_layout = None
        if not self._is_mla or not (metadata.is_prefill or metadata.is_chunked_prefill):
            return
        if not metadata.has_kv_shard or metadata.kv_split_size <= 1:
            return
        if self._block_table_i32 is None:
            raise RuntimeError("sharded MLA prefill requires a block table")
        cp_size = distributed.cp_world_size(self.device)
        if cp_size <= 1 or cp_size % metadata.kv_split_size:
            raise RuntimeError("KV split must be a positive divisor of the active CP group")

        local_owner = torch.tensor([metadata.kv_split_rank], dtype=torch.int64, device=self.device)
        owner_by_cp_rank = distributed.all_gather(local_owner, 0, cp_size, "cp")
        if torch.any((owner_by_cp_rank < 0) | (owner_by_cp_rank >= metadata.kv_split_size)).item():
            raise RuntimeError("KV split rank must be within the active KV split")
        expected_replicas = cp_size // metadata.kv_split_size
        owner_counts = torch.bincount(owner_by_cp_rank, minlength=metadata.kv_split_size)
        expected_counts = torch.full_like(owner_counts, expected_replicas)
        if owner_counts.numel() != metadata.kv_split_size or not torch.equal(owner_counts, expected_counts):
            raise RuntimeError("KV owner distribution does not match the active CP/KV topology")
        representatives = [
            torch.argmax((owner_by_cp_rank == owner).to(torch.int64)) for owner in range(metadata.kv_split_size)
        ]
        self._kv_owner_representatives = torch.stack(representatives)

        block_table = self._block_table_i32
        entry_ids = torch.arange(
            block_table.numel(),
            dtype=block_table.dtype,
            device=block_table.device,
        ).view_as(block_table)
        owner_offsets = torch.arange(metadata.kv_split_size, dtype=block_table.dtype, device=block_table.device)
        expanded = entry_ids.unsqueeze(-1) * metadata.kv_split_size + owner_offsets
        expanded = torch.where(block_table.unsqueeze(-1) >= 0, expanded, torch.full_like(expanded, -1))
        self._materialized_block_table = expanded.flatten(1).contiguous()
        self._sfa_page_layout = _build_stable_sfa_page_layout(self._materialized_block_table)

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        metadata = self._metadata
        assert metadata is not None

        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        k_cache, v_cache = layer_cache.key, layer_cache.value
        if k_cache is None or v_cache is None:
            raise RuntimeError(f"KV cache is missing for layer {layer_id}")
        num_tokens = q.shape[0]

        k_3d = k.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        v_3d = v.view(num_tokens, self.num_kv_heads, self.head_dim).contiguous()
        q_3d = q.view(num_tokens, self.num_heads, self.head_dim).contiguous()

        # Context-Parallel prefill: q/k/v are this rank's sequence shard while the
        # slot_mapping/metadata still describe the full global sequence (C++ does
        # not pre-shard the Python qwen3 path). All-gather K/V to the full
        # sequence, persist this rank's KV shard, and attend over its causal
        # prefix.
        cp_context = get_forward_context().cp_context
        if cp_context is not None:
            if not layer.causal:
                raise NotImplementedError("non-causal draft attention does not support context parallelism")
            if cp_context.has_prefix:
                raise NotImplementedError(
                    "non-MLA Python CP does not support chunked prefill with an existing KV prefix"
                )
            return self._prefill_cp(q_3d, k_3d, v_3d, metadata, cp_context, k_cache, v_cache)

        # Write KV to paged cache (kernel expects [T, kv_heads, head_dim]).
        kernels.reshape_paged_cache(metadata.slot_mapping, k_3d, v_3d, k_cache, v_cache)

        if metadata.is_prefill or metadata.is_chunked_prefill:
            if self._use_expanded_decode:
                return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)
            return self._prefill(
                q_3d,
                k_3d,
                v_3d,
                k_cache,
                v_cache,
                metadata,
                num_tokens,
                layer,
            )
        return self._decode(q_3d, k_cache, v_cache, metadata, num_tokens)

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor | None,
        k_pe_3d: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Absorbed-MLA attention. Returns [T, H, kv_lora]; caller bmm's W_UV."""
        metadata = self._metadata
        assert metadata is not None, "execute_mla called before prepare()"
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        # MLA reuses the K/V slots for the latent (nope) and rope caches.
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer_id}")
        if self._block_table_i32 is None:
            raise RuntimeError("MLA requires a block table")
        if self._mla_actual_seq_q is None or self._mla_actual_seq_kv is None:
            raise RuntimeError("MLA requires query and KV sequence lengths")

        cp_context = get_forward_context().cp_context
        if cp_context is None:
            if not cache_is_preprocessed:
                if k_latent_3d is None or k_pe_3d is None:
                    raise RuntimeError("MLA cache inputs are required")
                torch.ops.xllm_ops.reshape_paged_cache(
                    metadata.slot_mapping,
                    k_latent_3d,
                    k_pe_3d,
                    nope_cache,
                    rope_cache,
                )
            if topk is None:
                return self._mla_dense_fia_v2(
                    q_latent,
                    q_pe,
                    nope_cache,
                    rope_cache,
                    self._block_table_i32,
                    layer_id,
                )
            return self._mla_sparse(
                q_latent,
                q_pe,
                nope_cache,
                rope_cache,
                topk,
                self._block_table_i32,
                self._mla_actual_seq_q,
                self._mla_actual_seq_kv,
                layer_id,
            )

        if cache_is_preprocessed:
            raise RuntimeError("CP prefill does not support preprocessed MLA cache inputs")
        if topk is None:
            raise RuntimeError("CP prefill requires sparse MLA index output")
        if k_latent_3d is None or k_pe_3d is None:
            raise RuntimeError("CP prefill requires MLA cache inputs")
        global_latent = cp_gather_kv(k_latent_3d, cp_context).contiguous()
        global_rope = cp_gather_kv(k_pe_3d, cp_context).contiguous()
        cache_slots = metadata.local_slot_mapping if metadata.has_kv_shard else metadata.slot_mapping
        assert cache_slots is not None
        torch.ops.xllm_ops.reshape_paged_cache(
            cache_slots,
            global_latent,
            global_rope,
            nope_cache,
            rope_cache,
        )

        attention_nope, block_table = self._materialize_cp_cache(nope_cache, metadata, cp_context)
        attention_rope, _ = self._materialize_cp_cache(rope_cache, metadata, cp_context)
        if cp_context.query_index.numel() == 0:
            return q_latent.new_zeros(q_latent.shape)
        if metadata.has_kv_shard and metadata.kv_split_size > 1:
            attention_nope, attention_rope, block_table = self._materialize_sfa_layout(
                attention_nope,
                attention_rope,
            )

        query_index = cp_context.query_index
        segment_sequences = cp_context.segment_seq_indices
        q_real = q_latent.index_select(0, query_index).contiguous()
        q_pe_real = q_pe.index_select(0, query_index).contiguous()
        topk_real = topk.index_select(0, query_index).contiguous()
        local_block_table = block_table.index_select(0, segment_sequences).contiguous()
        output = self._mla_sparse(
            q_real,
            q_pe_real,
            attention_nope,
            attention_rope,
            topk_real,
            local_block_table,
            cp_context.q_cu_seqlens_tensor,
            cp_context.segment_kv_seq_lens_tensor,
            layer_id,
        )
        local_output = q_latent.new_zeros(q_latent.shape)
        local_output.index_copy_(0, query_index, output)
        return local_output

    def mla_preprocess_context(
        self,
        layer: Attention,
    ) -> MlaPreprocessContext | None:
        metadata = self._metadata
        if metadata is None or metadata.is_prefill or metadata.is_chunked_prefill:
            return None
        layer_cache = self._kv_caches[layer.layer_id]
        kv_cache, rope_cache = layer_cache.key, layer_cache.value
        if kv_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")
        return MlaPreprocessContext(
            kv_cache=kv_cache,
            rope_cache=rope_cache,
            slot_mapping=metadata.slot_mapping,
        )

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        metadata = self._metadata
        assert metadata is not None, "mla_index_context called before prepare()"
        assert self._block_table_i32 is not None
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        layer_cache = self._kv_caches[layer.layer_id]
        index_cache = layer_cache.index
        if index_cache is None:
            raise RuntimeError(f"MLA index cache is missing for layer {layer.layer_id}")
        index_cache_scale = layer_cache.index_scale
        slot_mapping = metadata.local_slot_mapping if metadata.has_kv_shard else metadata.slot_mapping
        if slot_mapping is None:
            raise RuntimeError("MLA index cache requires a slot mapping")
        cp_context = get_forward_context().cp_context
        block_table = self._block_table_i32
        if cp_context is not None:
            block_table = self._segment_block_table(block_table, cp_context)
        return MlaIndexContext(
            index_cache=index_cache,
            slot_mapping=slot_mapping,
            block_table=block_table,
            actual_seq_q=self._mla_actual_seq_q if cp_context is None else cp_context.q_cu_seqlens_tensor,
            actual_seq_kv=self._mla_actual_seq_kv if cp_context is None else cp_context.segment_kv_seq_lens_tensor,
            index_cache_scale=index_cache_scale,
            get_quant_indexer_metadata=lambda num_heads_q, head_dim, sparse_count, cmp_ratio: (
                self._get_quant_indexer_metadata(
                    num_heads_q,
                    index_cache.size(2),
                    head_dim,
                    sparse_count,
                    cmp_ratio,
                    cp_context,
                )
            ),
            update_index_cache=lambda values, scales: self._update_mla_index_cache(
                index_cache,
                index_cache_scale,
                slot_mapping,
                values,
                scales,
            ),
            materialize_index_cache=lambda: self._materialize_mla_index_cache(
                index_cache,
                index_cache_scale,
                metadata,
                cp_context,
            ),
            cp_context=cp_context,
        )

    def _segment_block_table(
        self,
        block_table: torch.Tensor,
        cp_context: CpContext,
    ) -> torch.Tensor:
        """Return the cached block-table rows for this CP segment layout."""
        # Keep this helper usable by lightweight test doubles that bypass
        # ``__init__`` while retaining the normal per-forward cache on the
        # production backend.
        cache = getattr(self, "_mla_cp_block_tables", None)
        if cache is None:
            cache = {}
            self._mla_cp_block_tables = cache
        cache_key = (id(cp_context), id(block_table))
        segmented = cache.get(cache_key)
        if segmented is None:
            segmented = block_table.index_select(0, cp_context.segment_seq_indices).contiguous()
            cache[cache_key] = segmented
        return segmented

    def _get_quant_indexer_metadata(
        self,
        num_heads_q: int,
        num_heads_k: int,
        head_dim: int,
        sparse_count: int,
        cmp_ratio: int,
        cp_context: CpContext | None = None,
    ) -> torch.Tensor:
        assert self._mla_actual_seq_q is not None
        assert self._mla_actual_seq_kv is not None
        # Warmup runs before graph capture without another prepare(). Its
        # metadata lives outside the graph pool and is released by the next
        # prepare(), so capturing a consumer of that cached tensor leaves a
        # stale address in replay. Capture the metadata producer as well, once
        # per graph, to retain its storage and refresh sequence-dependent data.
        context = get_forward_context()
        is_graph_capture = context.acl_graph is not None
        # Graph entries use distinct static sequence-length buffers. Include
        # their addresses so a graph capture cannot reuse QLI tiling metadata
        # produced for another entry with the same tensor shape. Eager
        # execution keeps the original compact cache key.
        entry_scope = ()
        if context.execution_state is not None:
            entry_scope = (
                self._mla_actual_seq_q.data_ptr(),
                self._mla_actual_seq_kv.data_ptr(),
            )
        cache_key = (
            num_heads_q,
            head_dim,
            sparse_count,
            cmp_ratio,
            is_graph_capture,
            *entry_scope,
        )
        metadata = self._mla_quant_indexer_metadata.get(cache_key)
        if metadata is None:
            actual_seq_q = self._mla_actual_seq_q
            actual_seq_kv = self._mla_actual_seq_kv
            max_seqlen_q = self._mla_max_seqlen_q
            max_seqlen_k = self._mla_max_seqlen_k
            if cp_context is not None:
                actual_seq_q = cp_context.q_cu_seqlens_tensor
                actual_seq_kv = cp_context.segment_kv_seq_lens_tensor
                ends = cp_context.q_cu_seqlens
                max_seqlen_q = max(
                    (end - (ends[index - 1] if index else 0) for index, end in enumerate(ends)),
                    default=0,
                )
                max_seqlen_k = max(cp_context.segment_kv_seq_lens, default=0)
            metadata = kernels.quant_lightning_indexer_metadata(
                num_heads_q,
                num_heads_k,
                head_dim,
                actual_seq_q,
                actual_seq_kv,
                max_seqlen_q,
                max_seqlen_k,
                sparse_count,
                cmp_ratio,
            )
            self._mla_quant_indexer_metadata[cache_key] = metadata
        return metadata

    @staticmethod
    def _update_mla_index_cache(
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor | None,
        slot_mapping: torch.Tensor,
        values: torch.Tensor,
        scales: torch.Tensor | None,
    ) -> None:
        if slot_mapping.numel() == 0:
            return
        cache_view = index_cache.view(-1, index_cache.size(-1))
        # ScatterNdUpdateV2 skips negative slots on device. Keep the row count
        # static: nonzero would synchronize the captured ACLGraph stream.
        scatter_indices = slot_mapping.reshape(-1, 1)
        kernels.scatter_nd_update(cache_view, scatter_indices, values)
        if index_cache_scale is not None and scales is not None:
            scale_view = index_cache_scale.view(-1, index_cache_scale.size(-1))
            kernels.scatter_nd_update(
                scale_view,
                scatter_indices,
                scales,
            )

    def _materialize_cp_cache(
        self,
        cache: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: CpContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if cp_context is None or not metadata.has_kv_shard or metadata.kv_split_size <= 1:
            assert self._block_table_i32 is not None
            return cache, self._block_table_i32
        if self._kv_owner_representatives is None or self._materialized_block_table is None:
            raise RuntimeError("KV shard materialization was not prepared")
        assert self._block_table_i32 is not None
        flat_blocks = self._block_table_i32.reshape(-1)
        safe_blocks = flat_blocks.clamp_min(0).to(torch.int64)
        local_blocks = cache.index_select(0, safe_blocks)
        gathered = distributed.all_gather(local_blocks, 0, cp_context.cp_size, "cp")
        gathered = gathered.view(cp_context.cp_size, flat_blocks.numel(), *cache.shape[1:])
        owner_blocks = gathered.index_select(0, self._kv_owner_representatives)
        order = [1, 0, *range(2, owner_blocks.dim())]
        materialized = owner_blocks.permute(order).reshape(
            flat_blocks.numel() * metadata.kv_split_size,
            *cache.shape[1:],
        )
        return materialized.contiguous(), self._materialized_block_table

    def _materialize_mla_index_cache(
        self,
        index_cache: torch.Tensor,
        index_cache_scale: torch.Tensor | None,
        metadata: AttentionMetadata,
        cp_context: CpContext | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        materialized_cache, block_table = self._materialize_cp_cache(
            index_cache,
            metadata,
            cp_context,
        )
        materialized_scale = None
        if index_cache_scale is not None:
            materialized_scale, _ = self._materialize_cp_cache(
                index_cache_scale,
                metadata,
                cp_context,
            )
        if cp_context is not None:
            block_table = self._segment_block_table(block_table, cp_context)
        return materialized_cache, materialized_scale, block_table

    def _materialize_sfa_layout(
        self,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        layout = self._sfa_page_layout
        if layout is None:
            raise RuntimeError("SFA cache layout was not prepared")
        target_nope = nope_cache.new_zeros((layout.page_count, *nope_cache.shape[1:]))
        target_rope = rope_cache.new_zeros((layout.page_count, *rope_cache.shape[1:]))
        target_nope.index_copy_(
            0,
            layout.target_page_ids,
            nope_cache.index_select(0, layout.source_page_ids),
        )
        target_rope.index_copy_(
            0,
            layout.target_page_ids,
            rope_cache.index_select(0, layout.source_page_ids),
        )
        return target_nope, target_rope, layout.block_table

    def _mla_sparse(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        topk: torch.Tensor,
        block_table: torch.Tensor,
        actual_seq_q: torch.Tensor,
        actual_seq_kv: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        if actual_seq_q is None:
            actual_seq_q = self._mla_actual_seq_q
        if actual_seq_kv is None:
            actual_seq_kv = self._mla_actual_seq_kv
        out = get_execution_buffer(
            ("SFA_OUTPUT", layer_id) + tuple(q_latent.shape),
            lambda: torch.empty_like(q_latent),
        )
        return kernels.sparse_flash_attention_out(
            q_latent,
            nope_cache,
            nope_cache,
            topk,
            block_table,
            actual_seq_q,
            actual_seq_kv,
            q_pe,
            rope_cache,
            self.scale,
            1,
            "TND",
            "PA_BSND",
            3,
            out,
        )  # [T, H, kv_lora]

    def _mla_dense_fia_v2_out(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        block_table: torch.Tensor,
        workspace: torch.Tensor,
        output: torch.Tensor,
        softmax_lse: torch.Tensor,
    ) -> None:
        if self._mla_actual_seq_q_host is None:
            raise RuntimeError("dense MLA requires query sequence lengths")
        if self._mla_actual_seq_kv_host is None:
            raise RuntimeError("dense MLA requires KV sequence lengths")
        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        is_prefill = bool(
            self._metadata is not None and (self._metadata.is_prefill or self._metadata.is_chunked_prefill)
        )
        torch.ops.npu.npu_fused_infer_attention_score_v2.out(
            q_latent,
            nope_flat,
            nope_flat,
            query_rope=q_pe,
            key_rope=rope_flat,
            pse_shift=None,
            atten_mask=self._causal_mask if is_prefill else None,
            actual_seq_qlen=self._mla_actual_seq_q_host,
            actual_seq_kvlen=self._mla_actual_seq_kv_host,
            block_table=block_table,
            num_query_heads=self.num_heads,
            num_key_value_heads=1,
            softmax_scale=self.scale,
            input_layout="TND",
            sparse_mode=(_SPARSE_MODE_RIGHT_DOWN_CAUSAL if is_prefill else _SPARSE_MODE_NONE),
            block_size=block_size,
            return_softmax_lse=False,
            workspace=workspace,
            out=[output, softmax_lse],
        )

    def _mla_dense_fia_v2(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        nope_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        block_table: torch.Tensor,
        layer_id: int,
    ) -> torch.Tensor:
        """Run dense absorbed MLA with FIA v2 and separate RoPE caches."""
        if not self._use_fia_v2:
            raise RuntimeError("dense MLA requires FIA v2 support")
        if self._mla_actual_seq_q_host is None:
            raise RuntimeError("dense MLA requires query sequence lengths")
        if self._mla_actual_seq_kv_host is None:
            raise RuntimeError("dense MLA requires KV sequence lengths")

        block_size = nope_cache.size(1)
        nope_flat = nope_cache.view(nope_cache.size(0), block_size, -1)
        rope_flat = rope_cache.view(rope_cache.size(0), block_size, -1)
        is_prefill = bool(
            self._metadata is not None and (self._metadata.is_prefill or self._metadata.is_chunked_prefill)
        )
        common_kwargs = {
            "query_rope": q_pe,
            "key_rope": rope_flat,
            "pse_shift": None,
            "atten_mask": self._causal_mask if is_prefill else None,
            "actual_seq_qlen": self._mla_actual_seq_q_host,
            "actual_seq_kvlen": self._mla_actual_seq_kv_host,
            "block_table": block_table,
            "num_query_heads": self.num_heads,
            "num_key_value_heads": 1,
            "softmax_scale": self.scale,
            "input_layout": "TND",
            "sparse_mode": (_SPARSE_MODE_RIGHT_DOWN_CAUSAL if is_prefill else _SPARSE_MODE_NONE),
            "block_size": block_size,
            "return_softmax_lse": False,
        }

        graph_context = get_forward_context().acl_graph
        if graph_context is None:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                q_latent,
                nope_flat,
                nope_flat,
                **common_kwargs,
            )
            return output

        output_key = ("MLA_DENSE_OUTPUT", layer_id) + tuple(q_latent.shape)
        graph_state = get_forward_context().execution_state
        if graph_state is None:
            output = self._mla_graph_outputs.get(output_key)
            if output is None:
                output = torch.empty_like(q_latent)
                self._mla_graph_outputs[output_key] = output
                self._mla_graph_lses[output_key] = torch.empty(
                    0,
                    dtype=q_latent.dtype,
                    device=q_latent.device,
                )
            softmax_lse = self._mla_graph_lses[output_key]
            workspace = self._mla_graph_workspaces.get(output_key)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    q_latent,
                    nope_flat,
                    nope_flat,
                    **common_kwargs,
                )
                self._mla_graph_workspaces[output_key] = workspace
        else:
            output = get_execution_buffer(output_key, lambda: torch.empty_like(q_latent))
            softmax_lse = get_execution_buffer(
                ("MLA_DENSE_LSE", layer_id) + tuple(q_latent.shape),
                lambda: torch.empty(0, dtype=q_latent.dtype, device=q_latent.device),
            )
            workspace = get_execution_buffer(
                ("MLA_DENSE_WORKSPACE", layer_id) + tuple(q_latent.shape),
                lambda: torch_npu._npu_fused_infer_attention_score_v2_get_max_workspace(
                    q_latent,
                    nope_flat,
                    nope_flat,
                    **common_kwargs,
                ),
            )

        stream = graph_context.stream
        event = torch.npu.ExternalEvent()
        event.wait(stream)
        event.reset(stream)
        torch.npu.graph_task_group_begin(stream)
        try:
            self._mla_dense_fia_v2_out(
                q_latent,
                q_pe,
                nope_cache,
                rope_cache,
                block_table,
                workspace,
                output,
                softmax_lse,
            )
        except Exception:
            torch.npu.graph_task_group_end(stream)
            raise
        handle = torch.npu.graph_task_group_end(stream)

        def _update_mla_fia_v2_args() -> None:
            self._mla_dense_fia_v2_out(
                q_latent,
                q_pe,
                nope_cache,
                rope_cache,
                block_table,
                workspace,
                output,
                softmax_lse,
            )

        graph_context.tasks.append(AclGraphTask(event, handle, _update_mla_fia_v2_args))
        return output

    # ------------------------------------------------------------------
    # Prefill: packed TND with causal mask
    # ------------------------------------------------------------------

    def _prefill(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_tokens: int,
        layer: Attention,
    ) -> torch.Tensor:
        actual_seq = self._cumulative_seq_lens(metadata, num_tokens)
        use_attention_mask = layer.causal or layer.fia_use_attention_mask
        atten_mask = self._causal_mask if use_attention_mask else None
        sparse_mode = layer.fia_sparse_mode
        if sparse_mode is None:
            sparse_mode = _SPARSE_MODE_RIGHT_DOWN_CAUSAL if layer.causal else _SPARSE_MODE_NONE

        # Prefix-cache hit (or chunked prefill with prior context): part of the
        # KV already lives in the paged cache, so this forward only carries the
        # new tokens (q_len < kv_len). Attend over the full paged KV via
        # block_table, mirroring _decode. Without this, the new query tokens
        # would only see their own KV (actual_seq_lengths_kv == q_len) and never
        # the cached prefix, diverging from a full recompute.
        if metadata.block_table is not None:
            block_size = k_cache.size(1)
            k_flat = k_cache.view(k_cache.size(0), block_size, -1)
            v_flat = v_cache.view(v_cache.size(0), block_size, -1)
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d,
                k_flat,
                v_flat,
                pse_shift=None,
                atten_mask=atten_mask,
                block_table=self._block_table_i32,
                actual_seq_lengths=actual_seq,
                actual_seq_lengths_kv=self._actual_seq_kv,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                block_size=block_size,
                sparse_mode=sparse_mode,
                pre_tokens=layer.fia_pre_tokens,
                next_tokens=layer.fia_next_tokens,
                softmax_lse_flag=False,
            )
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_3d,
            k_3d,
            v_3d,
            pse_shift=None,
            atten_mask=atten_mask,
            actual_seq_lengths=actual_seq,
            actual_seq_lengths_kv=actual_seq,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=sparse_mode,
            pre_tokens=layer.fia_pre_tokens,
            next_tokens=layer.fia_next_tokens,
            softmax_lse_flag=False,
        )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Context-Parallel prefill: all-gather KV, attend over causal prefix
    # ------------------------------------------------------------------

    def _prefill_cp(
        self,
        q_3d: torch.Tensor,
        k_3d: torch.Tensor,
        v_3d: torch.Tensor,
        metadata: AttentionMetadata,
        cp_context: CpContext,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Prefill attention for this rank's zigzag sequence shard.

        q/k/v hold this rank's ``total_local`` rows (two owned chunks per
        sequence, padding rows zeroed). We all-gather K/V back to the full
        global-order sequence, write the complete KV into the paged cache (so a
        later non-CP decode sees every position), then run one FIA over this
        rank's real queries. Each owned (sequence, half) segment is a packed
        sub-sequence: its ``real_count`` queries attend the causal prefix
        ``[0, segment_start + real_count)`` selected by ``kv_gather_index``.
        With ``sparse_mode=3`` (right-aligned causal) query row ``i`` of a
        segment attends KV ``[0, segment_start + i]`` — its exact global causal
        range. Segments are independent sub-sequences delimited by
        ``q_cu_seqlens`` / ``kv_cu_seqlens``, so both owned chunks resolve in a
        single call.
        """
        local_tokens = q_3d.shape[0]

        kv_global_k = cp_gather_kv(k_3d, cp_context)
        kv_global_v = cp_gather_kv(v_3d, cp_context)

        # Persist the full global-order KV into this rank's paged cache.
        kernels.reshape_paged_cache(
            metadata.slot_mapping,
            kv_global_k.contiguous(),
            kv_global_v.contiguous(),
            k_cache,
            v_cache,
        )

        # A CP rank can own only padding chunks when every sequence in the batch
        # is shorter than the zigzag chunk grid (e.g. a 1-token prompt with
        # cp_size > 1). It then has no real queries. The KV all-gather above
        # already ran (collectives must stay in lockstep across ranks) and the
        # full global KV is now in this rank's paged cache, so skip the FIA:
        # calling it with a 0-row query and empty actual_seq_lengths is rejected
        # by npu_fused_infer_attention. Return the all-zero shard directly.
        if cp_context.query_index.numel() == 0:
            return q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)

        # Real queries this rank owns, packed per (sequence, half) segment.
        q_real = q_3d.index_select(0, cp_context.query_index).contiguous()
        # Each segment's causal KV prefix, packed in the same segment order.
        kv_prefix_k = kv_global_k.index_select(0, cp_context.kv_gather_index).contiguous()
        kv_prefix_v = kv_global_v.index_select(0, cp_context.kv_gather_index).contiguous()

        output, _ = torch.ops.npu.npu_fused_infer_attention_score(
            q_real,
            kv_prefix_k,
            kv_prefix_v,
            pse_shift=None,
            atten_mask=self._causal_mask,
            actual_seq_lengths=cp_context.q_cu_seqlens,
            actual_seq_lengths_kv=cp_context.kv_cu_seqlens,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=3,
            softmax_lse_flag=False,
        )
        output = output.reshape(-1, self.num_heads * self.head_dim)

        # Scatter real-query outputs back into the padded [total_local] layout;
        # padding rows stay zero (they are never selected by restore_index in
        # the subsequent all-gather merge).
        out_local = q_3d.new_zeros(local_tokens, self.num_heads * self.head_dim)
        out_local.index_copy_(0, cp_context.query_index, output)
        return out_local

    # ------------------------------------------------------------------
    # Decode: FIA with block_table (paged KV, no gather)
    # ------------------------------------------------------------------

    def _fia_out(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_size: int,
        *,
        actual_seq_q: list[int] | torch.Tensor | None = None,
        actual_seq_kv: list[int] | torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
        output: torch.Tensor | None = None,
        lse: torch.Tensor | None = None,
    ) -> None:
        # Graph-task updates can run after another graph entry has prepared the
        # same backend.  Resolve all mutable per-entry state before launching
        # the operator so the deferred update cannot write into that entry's
        # buffers with the next entry's metadata.
        actual_seq_q = self._actual_seq_q if actual_seq_q is None else actual_seq_q
        actual_seq_kv = self._actual_seq_kv if actual_seq_kv is None else actual_seq_kv
        block_table = self._block_table_i32 if block_table is None else block_table
        workspace = self._graph_workspace if workspace is None else workspace
        output = self._current_graph_output if output is None else output
        lse = self._current_graph_lse if lse is None else lse
        if self._use_fia_v2:
            torch.ops.npu.npu_fused_infer_attention_score_v2.out(
                q,
                k,
                v,
                query_rope=None,
                key_rope=None,
                pse_shift=None,
                atten_mask=None,
                actual_seq_qlen=actual_seq_q,
                actual_seq_kvlen=actual_seq_kv,
                block_table=block_table,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                return_softmax_lse=False,
                workspace=workspace,
                out=[output, lse],
            )
            return

        torch.ops.npu.npu_fused_infer_attention_score.out(
            q,
            k,
            v,
            pse_shift=None,
            atten_mask=None,
            actual_seq_lengths=actual_seq_q,
            actual_seq_lengths_kv=actual_seq_kv,
            block_table=block_table,
            num_heads=self.num_heads,
            scale=self.scale,
            input_layout="TND",
            num_key_value_heads=self.num_kv_heads,
            sparse_mode=_SPARSE_MODE_NONE,
            block_size=block_size,
            softmax_lse_flag=False,
            workspace=workspace,
            out=[output, lse],
        )

    def _decode(
        self,
        q_3d: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        metadata: AttentionMetadata,
        num_tokens: int,
    ) -> torch.Tensor:
        block_size = k_cache.size(1)
        k_flat = k_cache.view(k_cache.size(0), block_size, -1)
        v_flat = v_cache.view(v_cache.size(0), block_size, -1)

        graph_context = get_forward_context().acl_graph
        if graph_context is not None:
            if self._current_graph_output is None:
                raise RuntimeError("ACL graph output buffer is not prepared")
            actual_seq_q = self._actual_seq_q
            actual_seq_kv = self._actual_seq_kv
            block_table = self._block_table_i32
            workspace = self._graph_workspace
            output = self._current_graph_output
            lse = self._current_graph_lse
            stream = graph_context.stream
            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            torch.npu.graph_task_group_begin(stream)
            try:
                self._fia_out(
                    q_3d,
                    k_flat,
                    v_flat,
                    block_size,
                    actual_seq_q=actual_seq_q,
                    actual_seq_kv=actual_seq_kv,
                    block_table=block_table,
                    workspace=workspace,
                    output=output,
                    lse=lse,
                )
            except Exception:
                torch.npu.graph_task_group_end(stream)
                raise
            handle = torch.npu.graph_task_group_end(stream)

            def _update_fia_args() -> None:
                self._fia_out(
                    q_3d,
                    k_flat,
                    v_flat,
                    block_size,
                    actual_seq_q=actual_seq_q,
                    actual_seq_kv=actual_seq_kv,
                    block_table=block_table,
                    workspace=workspace,
                    output=output,
                    lse=lse,
                )

            graph_context.tasks.append(AclGraphTask(event, handle, _update_fia_args))
            return output.reshape(num_tokens, self.num_heads * self.head_dim)

        if self._use_fia_v2:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score_v2(
                q_3d,
                k_flat,
                v_flat,
                query_rope=None,
                key_rope=None,
                pse_shift=None,
                atten_mask=None,
                actual_seq_qlen=self._actual_seq_q[:num_tokens],
                actual_seq_kvlen=self._actual_seq_kv[:num_tokens],
                block_table=self._block_table_i32,
                num_query_heads=self.num_heads,
                softmax_scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                return_softmax_lse=False,
            )
        else:
            output, _ = torch.ops.npu.npu_fused_infer_attention_score(
                q_3d,
                k_flat,
                v_flat,
                pse_shift=None,
                atten_mask=None,
                actual_seq_lengths=self._actual_seq_q[:num_tokens],
                actual_seq_lengths_kv=self._actual_seq_kv[:num_tokens],
                block_table=self._block_table_i32,
                num_heads=self.num_heads,
                scale=self.scale,
                input_layout="TND",
                num_key_value_heads=self.num_kv_heads,
                sparse_mode=_SPARSE_MODE_NONE,
                block_size=block_size,
                softmax_lse_flag=False,
            )
        return output.reshape(num_tokens, self.num_heads * self.head_dim)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cumulative_seq_lens(
        self,
        metadata: AttentionMetadata,
        num_tokens: int,
    ) -> list[int]:
        if self._actual_seq_lens is not None:
            return self._actual_seq_lens
        return [num_tokens]
