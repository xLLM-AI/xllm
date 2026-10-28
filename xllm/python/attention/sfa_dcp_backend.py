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

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

from xllm.python.attention.backend import (
    AttentionMetadata,
    LayerCache,
    MlaIndexContext,
    MlaPreprocessContext,
    has_rope_dim,
)
from xllm.python.attention.expanded_decode_metadata import (
    resolve_expanded_decode_metadata,
)
from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.attention.npu_paged_attention import (
    NpuPagedAttentionBackend,
    _PreparedMlaAttention,
    write_mla_paged_cache,
)
from xllm.python.layers.sfa_dcp import (
    AscendSFADCPImpl,
    AscendSFADCPMetadata,
    AscendSFADCPMetadataBuilder,
    DCPContext,
)
from xllm.python.model_executor.cp_utils import cp_gather_kv
from xllm.python.model_executor.forward_context import (
    copy_into_execution_buffer,
    get_forward_context,
)

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention


@dataclass
class _ProcessGroupCoordinator:
    world_size: int
    rank_in_group: int
    device_group: ProcessGroup


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
            is_mla=True,
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

    @property
    def logical_page_size(self) -> int:
        return self.page_size * self._dcp_group.size()

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        super().bind_kv_caches(kv_caches)
        self._kv_layout = KVShardLayout(
            self.page_size,
            self._dcp_group.size(),
            self._dcp_group.rank(),
        )
        self._impl = AscendSFADCPImpl(
            _ProcessGroupCoordinator(
                world_size=self._dcp_group.size(),
                rank_in_group=self._dcp_group.rank(),
                device_group=self._dcp_group,
            ),
            scale=self.scale,
            index_topk=self._index_topk,
            layout=self._kv_layout,
        )
        self._builder = AscendSFADCPMetadataBuilder(
            layout=self._kv_layout,
            device=self.device,
            max_num_reqs=max(self._max_num_reqs, 1),
        )

    def prepare_owned_graph_metadata(self, metadata: AttentionMetadata) -> None:
        """Reject Unified warmup until DCP derivations are part of its graph."""
        raise NotImplementedError("Unified MTP graphs do not support DCP-derived metadata")

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
        owned_metadata: bool = False,
    ) -> None:
        super().prepare(metadata, graph_mode=graph_mode, owned_metadata=owned_metadata)
        # DCP derives separate buffers; updating the source arena is insufficient.
        self._graph_metadata_updated_in_place = False
        self._sfa_metadata = None
        if self._kv_layout is None or self._builder is None:
            return
        prepared = getattr(metadata, "prepared_attention_state", None)
        if isinstance(prepared, _PreparedMlaAttention):
            # Prepare retains only final Slot views. Derive DCP metadata here,
            # on the serialized Launch stream and inside ACL graph capture, so
            # MTP's Device-updated lengths and slots remain live on every replay.
            slot_mapping = prepared.slot_mapping
            kv_seq_lens = prepared.actual_seq_kv
            if graph_mode:
                self._mla_max_seqlen_k = prepared.block_table.shape[1] * self.logical_page_size
        else:
            expanded = resolve_expanded_decode_metadata(metadata, block_size=self.logical_page_size)
            slot_mapping = metadata.slot_mapping
            kv_seq_lens = expanded.kv_seq_lens if expanded is not None else metadata.kv_seq_lens
        block_table = self._block_table_i32
        if block_table is None:
            raise RuntimeError("SFA DCP requires a block table.")
        if kv_seq_lens is None:
            raise RuntimeError("SFA DCP requires kv_seq_lens.")

        local_slots = self._kv_layout.localize_slots(slot_mapping)
        if graph_mode:
            local_slots = copy_into_execution_buffer(
                ("DCP_LOCAL_SLOTS", tuple(local_slots.shape)),
                local_slots,
            )
        self._local_slot_mapping = local_slots

        # ACL graph padded lanes keep logical block 0 and kv_seq_len=1 so
        # sparse MLA tiling stays valid. Expanding that row yields pages
        # 0..dcp_size-1. Do not rewrite those pages to -1: LightningIndexer
        # treats -1 as a huge page id and MTE-OOB.
        expanded = self._kv_layout.expand_indexer_block_table(block_table)
        if graph_mode:
            expanded = copy_into_execution_buffer(
                ("DCP_INDEXER_BT", tuple(expanded.shape)),
                expanded,
            )
        self._expanded_indexer_block_table = expanded

        # Speculative verification expands each request into one row per
        # proposed token.  Use the same expanded row layout as the parent
        # paged-attention backend; reading the outer metadata here would build
        # a DCP context with fewer requests than local slots and query rows.
        num_reqs = int(block_table.shape[0])
        num_input_tokens = int(local_slots.numel())
        seq_lens = kv_seq_lens.to(dtype=torch.int32)[:num_reqs]
        local_seq_lens = self._kv_layout.local_seq_lens(seq_lens)
        if graph_mode:
            local_seq_lens = copy_into_execution_buffer(
                ("DCP_LOCAL_SEQ", tuple(local_seq_lens.shape)),
                local_seq_lens,
            )

        num_prefills = 0
        if not graph_mode and not self._use_expanded_decode and (metadata.is_prefill or metadata.is_chunked_prefill):
            num_prefills = num_reqs

        if isinstance(prepared, _PreparedMlaAttention) and num_prefills == 0:
            # The ordinary builder owns a shared length buffer. Each prepared
            # graph entry must instead capture its own persistent tensors.
            self._sfa_metadata = AscendSFADCPMetadata(
                num_prefills=0,
                dcp_context=DCPContext(
                    slot_mapping=local_slots,
                    block_table=block_table,
                    seq_lens=local_seq_lens,
                ),
            )
            return

        self._sfa_metadata = self._builder.build(
            slot_mapping=local_slots,
            block_table=block_table,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_input_tokens=num_input_tokens,
            dcp_local_seq_lens=local_seq_lens,
            num_prefills=num_prefills,
        )

    def prepare_graph_replay(self, metadata: AttentionMetadata) -> None:
        prepared = getattr(metadata, "prepared_attention_state", None)
        if not isinstance(prepared, _PreparedMlaAttention):
            # Ordinary decode graphs update their static inputs before replay.
            # Only prepared graphs capture the DCP metadata transforms.
            super().prepare_graph_replay(metadata)
            return
        execution_state = get_forward_context().execution_state
        if execution_state is None:
            raise RuntimeError("prepared DCP graph replay requires an execution state")
        buffers = execution_state.persistent_buffers
        num_reqs, block_count = prepared.block_table.shape
        local_slots = buffers.get(("DCP_LOCAL_SLOTS", tuple(prepared.slot_mapping.shape)))
        local_seq_lens = buffers.get(("DCP_LOCAL_SEQ", (num_reqs,)))
        expanded_block_table = buffers.get(("DCP_INDEXER_BT", (num_reqs, block_count * self._dcp_group.size())))
        if local_slots is None or local_seq_lens is None or expanded_block_table is None:
            raise RuntimeError("prepared DCP graph replay requires captured metadata buffers")

        # The prepared parent branch only installs Host state and final views.
        # Captured Device operators refresh these entry-owned DCP buffers from
        # the live Slot inputs during replay; do not enqueue the same work here.
        super().prepare(metadata, graph_mode=True)
        self._mla_max_seqlen_k = block_count * self.logical_page_size
        self._local_slot_mapping = local_slots
        self._expanded_indexer_block_table = expanded_block_table
        self._sfa_metadata = AscendSFADCPMetadata(
            num_prefills=0,
            dcp_context=DCPContext(
                slot_mapping=local_slots,
                block_table=prepared.block_table,
                seq_lens=local_seq_lens,
            ),
        )

    def mla_preprocess_context(self, layer: Attention) -> MlaPreprocessContext | None:
        context = super().mla_preprocess_context(layer)
        if context is None:
            return None
        if self._local_slot_mapping is None:
            raise RuntimeError("SFA DCP MLA preprocessing requires prepare()")
        # Latent KV is sharded; the index cache still uses global logical slots.
        return replace(context, slot_mapping=self._local_slot_mapping)

    def indexer_block_table(self) -> torch.Tensor:
        if self._expanded_indexer_block_table is not None:
            return self._expanded_indexer_block_table
        return super().indexer_block_table()

    def mla_index_context(self, layer: Attention) -> MlaIndexContext:
        context = super().mla_index_context(layer)
        indexer_table = self._expanded_indexer_block_table
        if indexer_table is None:
            return context
        if context.cp_context is not None:
            indexer_table = self._segment_block_table(indexer_table, context.cp_context)

        def materialize_index_cache() -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
            index_cache, index_cache_scale, _ = context.materialize_index_cache()
            return index_cache, index_cache_scale, indexer_table

        return replace(
            context,
            block_table=indexer_table,
            materialize_index_cache=materialize_index_cache,
        )

    def execute_mla(
        self,
        q_latent: torch.Tensor,
        q_pe: torch.Tensor | None,
        k_latent_3d: torch.Tensor | None,
        k_pe_3d: torch.Tensor | None,
        layer: Attention,
        topk: torch.Tensor | None = None,
        cache_is_preprocessed: bool = False,
    ) -> torch.Tensor:
        """Run sparse MLA, optionally reusing cache data prepared by the fused path."""
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
        if nope_cache is None:
            raise RuntimeError(f"MLA latent cache is missing for layer {layer.layer_id}")
        # Validate the rope story before touching any cache: query and KV decide
        # "NoPE?" independently, so a mixed state must not reach the SFA kernel.
        if has_rope_dim(q_pe) != has_rope_dim(rope_cache):
            raise RuntimeError(
                f"MLA rope cache and query disagree on layer {layer.layer_id}: "
                f"query rope dim={0 if q_pe is None else q_pe.shape[-1]}, "
                f"cache rope dim={0 if rope_cache is None else rope_cache.shape[-1]}"
            )

        attn_metadata.dcp_context.gather_context = None
        cp_context = ctx.cp_context
        if cp_context is not None and cache_is_preprocessed:
            raise RuntimeError("CP prefill requires unfused MLA cache inputs")
        if not cache_is_preprocessed:
            if k_latent_3d is None:
                raise RuntimeError("SFA DCP requires K tensors unless MLA preprocessing wrote the cache")
            if cp_context is not None:
                # Restore global token order before each owner writes its slots.
                k_latent_3d = cp_gather_kv(k_latent_3d, cp_context).contiguous()
                k_pe_3d = cp_gather_kv(k_pe_3d, cp_context).contiguous() if k_pe_3d is not None else None
            write_mla_paged_cache(
                attn_metadata.dcp_context.slot_mapping,
                k_latent_3d,
                k_pe_3d,
                nope_cache,
                rope_cache,
            )

        kv_cache = (nope_cache, rope_cache)
        self._impl._record_dcp_kv_gather_context(kv_cache, attn_metadata)
        if cp_context is not None:
            query_index = cp_context.query_index
            block_table = attn_metadata.dcp_context.kv_gather_block_table
            if block_table is None:
                raise RuntimeError("CP prefill requires DCP KV gather metadata")
            output = self._impl._execute_sparse_flash_attention_process(
                q_latent.index_select(0, query_index).contiguous(),
                q_pe.index_select(0, query_index).contiguous() if q_pe is not None else None,
                kv_cache,
                topk.index_select(0, query_index).contiguous(),
                attn_metadata,
                cp_context.q_cu_seqlens_tensor,
                cp_context.segment_kv_seq_lens_tensor,
                block_table=self._segment_block_table(block_table, cp_context),
            )
            local_output = torch.zeros_like(q_latent)
            local_output.index_copy_(0, query_index, output)
            return local_output

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
