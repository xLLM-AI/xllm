# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""NPU MLA backend that shards decode KV across the DCP group."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from xllm.python.attention.backend import AttentionMetadata, LayerCache, MlaIndexContext
from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.attention.npu_paged_attention import NpuPagedAttentionBackend
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
)
from xllm.python.model_executor.forward_context import copy_into_execution_buffer, get_forward_context

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


@dataclass
class _ProcessGroupCoordinator:
    world_size: int
    rank_in_group: int
    device_group: ProcessGroup


def _coordinator(group: ProcessGroup) -> _ProcessGroupCoordinator:
    return _ProcessGroupCoordinator(
        world_size=group.size(),
        rank_in_group=group.rank(),
        device_group=group,
    )


def dcp_layer_options(layer: Attention) -> int:
    cfg = getattr(layer, "cfg", None)
    return int(getattr(cfg, "index_topk", 2048)) if cfg is not None else 2048


class SfaDcpAttentionBackend(NpuPagedAttentionBackend):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        scale: float,
        sliding_window: int,
        device: torch.device,
        dtype: torch.dtype,
        dcp_group: ProcessGroup,
        *,
        index_topk: int,
        max_num_reqs: int,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            scale=scale,
            sliding_window=sliding_window,
            device=device,
            dtype=dtype,
        )
        self._dcp_group = dcp_group
        self._index_topk = index_topk
        self._max_num_reqs = max_num_reqs
        self._kv_layout: KVShardLayout | None = None
        self._impl: AscendSFADCPImpl | None = None
        self._builder: AscendSFADCPMetadataBuilder | None = None
        self._local_slot_mapping: torch.Tensor | None = None
        self._expanded_indexer_block_table: torch.Tensor | None = None
        self._sfa_metadata: AscendSFADCPMetadata | None = None

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._kv_layout = KVShardLayout(
            self.page_size,
            self._dcp_group.size(),
            self._dcp_group.rank(),
        )
        coordinator = _coordinator(self._dcp_group)
        self._impl = AscendSFADCPImpl(
            coordinator,
            scale=self.scale,
            index_topk=self._index_topk,
            layout=self._kv_layout,
        )
        self._builder = AscendSFADCPMetadataBuilder(
            layout=self._kv_layout,
            device=self.device,
            max_num_reqs=max(self._max_num_reqs, 1),
        )

    def _ensure_builder_capacity(self, num_reqs: int) -> None:
        if self._builder is None or self._kv_layout is None:
            raise RuntimeError("SFA DCP backend requires bind_kv_caches before execute")
        if num_reqs <= self._builder.dcp_local_seq_lens_buf.shape[0]:
            return
        raise RuntimeError(
            "SFA DCP builder buffer is too small; "
            f"max_num_reqs={self._builder.dcp_local_seq_lens_buf.shape[0]}, "
            f"num_reqs={num_reqs}"
        )

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        super().prepare(metadata, graph_mode=graph_mode)
        self._sfa_metadata = None
        if self._kv_layout is None or self._builder is None:
            return
        if metadata.block_table is None:
            raise RuntimeError("SFA DCP requires a block table.")
        if metadata.kv_seq_lens is None:
            raise RuntimeError("SFA DCP requires kv_seq_lens.")

        local_slots = self._kv_layout.localize_slots(metadata.slot_mapping)
        if graph_mode:
            local_slots = copy_into_execution_buffer(
                ("DCP_LOCAL_SLOTS", tuple(local_slots.shape)),
                local_slots,
            )
        self._local_slot_mapping = local_slots

        if self._block_table_i32 is not None:
            expanded = self._kv_layout.expand_indexer_block_table(self._block_table_i32)
            if graph_mode:
                expanded = copy_into_execution_buffer(
                    ("DCP_INDEXER_BT", tuple(expanded.shape)),
                    expanded,
                )
                padded_rows = metadata.slot_mapping < 0
                if padded_rows.numel() == expanded.shape[0]:
                    expanded[padded_rows] = -1
            self._expanded_indexer_block_table = expanded
        else:
            self._expanded_indexer_block_table = None

        num_reqs = int(metadata.block_table.shape[0])
        num_input_tokens = int(local_slots.numel())
        self._ensure_builder_capacity(num_reqs)
        seq_lens = metadata.kv_seq_lens.to(dtype=torch.int32)[:num_reqs]
        local_seq_lens = self._kv_layout.local_seq_lens(seq_lens)
        if graph_mode:
            local_seq_lens = copy_into_execution_buffer(
                ("DCP_LOCAL_SEQ", tuple(local_seq_lens.shape)),
                local_seq_lens,
            )

        num_prefills = 0
        if not graph_mode and (metadata.is_prefill or metadata.is_chunked_prefill):
            num_prefills = num_reqs

        attn_metadata = self._builder.build(
            slot_mapping=local_slots,
            block_table=self._block_table_i32
            if self._block_table_i32 is not None
            else metadata.block_table.to(torch.int32),
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            dcp_local_seq_lens=local_seq_lens,
            num_prefills=num_prefills,
        )
        attn_metadata.dcp_context.slot_mapping = local_slots[:num_input_tokens]
        attn_metadata.dcp_context.seq_lens = local_seq_lens[:num_reqs]
        if self._block_table_i32 is not None:
            attn_metadata.dcp_context.block_table = self._block_table_i32[:num_reqs]
        self._sfa_metadata = attn_metadata

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        context = super().mla_index_context(layer)
        if self._expanded_indexer_block_table is None:
            return context
        return MlaIndexContext(
            index_cache=context.index_cache,
            slot_mapping=context.slot_mapping,
            block_table=self._expanded_indexer_block_table,
            actual_seq_q=context.actual_seq_q,
            actual_seq_kv=context.actual_seq_kv,
            index_cache_scale=context.index_cache_scale,
            get_quant_indexer_metadata=context.get_quant_indexer_metadata,
            update_index_cache=context.update_index_cache,
        )

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor,
        k_latent_3d: torch.Tensor,
        k_pe_3d: torch.Tensor,
        layer: Attention,
        topk: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if topk is None:
            raise NotImplementedError("dense MLA (topk=None) is not supported on SfaDcpAttentionBackend")
        if self._impl is None or self._kv_layout is None:
            raise RuntimeError("SFA DCP backend requires bind_kv_caches before execute_mla")
        attn_metadata = self._sfa_metadata
        if attn_metadata is None:
            raise RuntimeError("SFA DCP execute_mla requires prepare()")
        if self._mla_actual_seq_q is None or self._mla_actual_seq_kv is None:
            raise RuntimeError("SFA DCP execute_mla requires MLA sequence lengths from prepare()")
        ctx = get_forward_context()
        layer_cache = ctx.layer_caches[layer.layer_id]
        nope_cache, rope_cache = layer_cache.key, layer_cache.value
        if nope_cache is None or rope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")

        attn_metadata.dcp_context.gather_context = None
        torch.ops.xllm_ops.reshape_paged_cache(
            attn_metadata.dcp_context.slot_mapping,
            k_latent_3d,
            k_pe_3d,
            nope_cache,
            rope_cache,
        )
        kv_cache = (nope_cache, rope_cache)
        self._impl._store_parallel_kv(k_pe_3d, k_latent_3d, None, kv_cache, attn_metadata)
        self._impl._record_query_gather_context(q_latent, q_pe, attn_metadata)
        return self._impl._execute_sparse_flash_attention_process(
            q_latent,
            q_pe,
            kv_cache,
            topk,
            attn_metadata,
            self._mla_actual_seq_q,
            self._mla_actual_seq_kv,
        )
