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

"""DeepSeek-V4 compressed-attention backend.

Orchestrates the model's sliding attention, Compressed Sparse Attention (CSA),
and Heavily Compressed Attention (HCA) layers. It consumes the legacy xLLM
runtime :class:`DsaMetadata` contract built by :mod:`dsa_metadata` and drives
``sparse_attn_sharedkv``, KV compression, and the CSA lightning indexer. This is
the Python counterpart of C++ ``DSAttentionImpl``
(``core/layers/npu_torch/deepseek_sparse_attention.cpp``).

The backend owns no KV storage: caches are bound from the C++ executor's
``LayerCache`` 11-tuple. Per step it builds current-forward metadata, resolves
the per-layer cache mapping, writes new KV into the sliding-window cache, runs
the CSA/HCA compressor, runs the indexer for CSA, and dispatches the sparse
attention kernel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from xllm.python.attention.backend import (
    AttentionBackend,
    CsaIndexContext,
    LayerCache,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs,
)
from xllm.python.model_executor.forward_context import get_forward_context

if TYPE_CHECKING:
    from xllm.python.attention.backend import AttentionMetadata
    from xllm.python.layers.attention import Attention

# Official DeepSeek-V4 layer type names. Compression ratios are the runtime
# representation used by the C++ cache metadata contract.
SLIDING_ATTENTION = "sliding_attention"
COMPRESSED_SPARSE_ATTENTION = "compressed_sparse_attention"
HEAVILY_COMPRESSED_ATTENTION = "heavily_compressed_attention"

# Sparse mask modes used by C++ DSAttentionImpl (rightDownCausal variants).
_MASK_MODE_RIGHT_DOWN_CAUSAL = 3
_MASK_MODE_COMPRESS = 4


def _attention_type_for_compress_ratio(compress_ratio: int) -> str:
    attention_types = {
        1: SLIDING_ATTENTION,
        4: COMPRESSED_SPARSE_ATTENTION,
        128: HEAVILY_COMPRESSED_ATTENTION,
    }
    try:
        return attention_types[compress_ratio]
    except KeyError as error:
        raise ValueError(f"unsupported DeepSeek-V4 compression ratio: {compress_ratio}") from error


@dataclass
class _CompressedAttentionCacheMapping:
    """Per-layer resolved cache indices (mirrors C++ ``DsaCacheMapping``)."""

    cmp_cache_idx: int = -1
    index_cache_idx: int = -1
    indexer_scale_cache_idx: int = -1
    ori_cache_idx: int = -1
    kv_state_cache_idx: int = -1
    score_state_cache_idx: int = -1
    index_kv_state_cache_idx: int = -1
    index_score_state_cache_idx: int = -1


@dataclass(frozen=True)
class _CompressedAttentionForwardMeta:
    """Subset of C++ ModelInputParams::meta used by DSV4 metadata builders."""

    q_max_seq_len: int
    kv_max_seq_len: int


class CsaAttentionBackend(AttentionBackend):
    """Backend for DeepSeek-V4 sliding-attention, CSA, and HCA layers on NPU.

    The filename and class use CSA terminology because CSA is the indexed sparse
    path. The same backend also dispatches HCA and sliding-attention layers,
    selected from each layer's compression ratio.
    """

    def __init__(
        self,
        compress_ratios: list[int],
        window_size: int,
        n_layers: int,
        num_heads: int,
        attn_head_dim: int,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs(compress_ratios, window_size, n_layers)
        self._builder = DsaMetadataBuilder(self.caches_info, self.group_infos)
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim**-0.5

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None

    # -- AttentionBackend interface -----------------------------------------

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        self._kv_caches = kv_caches

    def _current_forward_metadata(self) -> AttentionMetadata:
        try:
            return get_forward_context().metadata
        except RuntimeError:
            if self._metadata is None:
                raise
            return self._metadata

    def prepare(
        self,
        metadata: AttentionMetadata,
        *,
        graph_mode: bool = False,
    ) -> None:
        if graph_mode:
            raise NotImplementedError("DeepSeek-V4 ACL graph support is not part of the eager CSA/HCA backend")
        self._metadata = metadata

    def prepare_csa_metadata_for_forward(
        self,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build compressed-attention metadata in the current model forward."""
        metadata = metadata or self._metadata
        assert metadata is not None
        multi_block_tables = list(metadata.multi_block_tables)
        kv_seq_lens_host = metadata.kv_seq_lens_host
        kv_seq_lens = (
            kv_seq_lens_host.cpu().tolist() if kv_seq_lens_host is not None and kv_seq_lens_host.numel() > 0 else []
        )
        q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
        q_seq_lens = (
            q_seq_lens_host.cpu().tolist() if q_seq_lens_host is not None and q_seq_lens_host.numel() > 0 else None
        )
        # The dsa_* fields are legacy C++/pybind contract names. Their tensors
        # are model-owned and scoped to the current forward.
        positions = getattr(metadata, "dsa_positions", None)
        if positions is None:
            positions = torch.empty(0, dtype=torch.int64)
        base_cos_sin = getattr(metadata, "dsa_cos_sin", None)
        compressed_metadata = self._builder.build(
            multi_block_tables=multi_block_tables,
            kv_seq_lens=kv_seq_lens,
            q_seq_lens=q_seq_lens,
            positions=positions,
            dsa_cos_sin=base_cos_sin,
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
            enable_graph=False,
        )
        self._populate_compressed_attention_rope(compressed_metadata, metadata)
        self._move_metadata_to_device(compressed_metadata)
        self._build_precomputed_metadata(compressed_metadata, metadata)
        metadata.dsa_metadata = compressed_metadata

    def select_compressed_attention_layer_rope(
        self,
        layer_id: int,
        cos_sin_cache: torch.Tensor,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Select the main q/kv RoPE group for the current DSV4 layer.

        C++ updates ``DSAMetadata::layer_id/cos/sin`` in the model layer loop
        from ``input_rope_by_ratio``. Python keeps the full cache here because
        the model and indexer gather it with the current input positions, but
        the selected group and lifetime are otherwise identical.
        """
        metadata = metadata or self._metadata
        if metadata is None or metadata.dsa_metadata is None:
            raise RuntimeError("compressed-attention metadata must be prepared before selecting layer RoPE")
        compressed_metadata = metadata.dsa_metadata
        chunks = cos_sin_cache.chunk(2, dim=-1)
        compressed_metadata.layer_id = layer_id
        compressed_metadata.cos_table = chunks[0].contiguous()
        compressed_metadata.sin_table = chunks[1].contiguous()

    def _populate_compressed_attention_rope(
        self,
        compressed_metadata: DsaMetadata,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build request-shaped RoPE tensors for the current forward."""
        metadata = metadata or self._metadata
        css = getattr(metadata, "dsa_cos_sin", None) if metadata is not None else None
        if compressed_metadata.cos_table is None and css is not None and css.numel() > 0:
            compressed_metadata.cos_table, compressed_metadata.sin_table = (
                tensor.contiguous() for tensor in css.chunk(2, dim=-1)
            )
        csa_cos_sin = getattr(metadata, "dsa_c4_cos_sin", None) if metadata is not None else None
        if csa_cos_sin is not None and compressed_metadata.c4_pad_positions.numel() > 0:
            csa_indices = compressed_metadata.c4_pad_positions.clamp_min(0).long().to(csa_cos_sin.device)
            compressed_metadata.c4_cos, compressed_metadata.c4_sin = (
                tensor.contiguous() for tensor in csa_cos_sin.index_select(0, csa_indices).chunk(2, dim=-1)
            )
        hca_cos_sin = getattr(metadata, "dsa_c128_cos_sin", None) if metadata is not None else None
        if hca_cos_sin is not None and compressed_metadata.c128_pad_positions.numel() > 0:
            hca_indices = compressed_metadata.c128_pad_positions.clamp_min(0).long().to(hca_cos_sin.device)
            compressed_metadata.c128_cos, compressed_metadata.c128_sin = (
                tensor.contiguous() for tensor in hca_cos_sin.index_select(0, hca_indices).chunk(2, dim=-1)
            )

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """Run one sliding-attention, CSA, or HCA layer.

        ``q``/``k``/``v`` here are the model-projected, RoPE-applied tensors the
        DeepseekV4 attention layer hands in; the backend only owns cache writes
        and the kernel dispatch.
        """
        metadata = self._current_forward_metadata()
        compressed_metadata = getattr(metadata, "dsa_metadata", None)
        assert compressed_metadata is not None
        # Late-populate RoPE if prepare ran before the model
        # attached the RoPE tables (prepare is called by the executor before
        # model.forward, so dsa_cos_sin may have been None at prepare time).
        if compressed_metadata.cos_table is None or compressed_metadata.sin_table is None:
            self._populate_compressed_attention_rope(compressed_metadata, metadata)
        # Late-populate CSA/HCA RoPE tables using the C++ metadata contract's
        # c4_pad_positions/c128_pad_positions fields.
        # Mirrors C++ DeepseekV4RotaryEmbedding::build(positions_map) per group.
        # Also late-populate input_positions (prepare ran before model.forward
        # set self._positions, so it was empty at prepare time).
        if compressed_metadata.input_positions.numel() == 0:
            pos = getattr(metadata, "dsa_positions", None)
            if pos is not None and pos.numel() > 0:
                compressed_metadata.input_positions = pos
        if compressed_metadata.c4_cos is None or compressed_metadata.c128_cos is None:
            self._populate_compressed_attention_rope(compressed_metadata, metadata)
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        attention_type = _attention_type_for_compress_ratio(compress_ratio)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        is_prefill = metadata.is_prefill
        is_chunked_prefill = metadata.is_chunked_prefill
        use_temporary_prefill_kv = is_prefill and not is_chunked_prefill
        # 1) Prepare ori_kv for attention (mirrors C++ :790-816).
        # Full prefill: use kv directly as temporary PA_ND (don't scatter to paged).
        # Decode/chunked: scatter to paged SWA cache.
        ori_kv = layer_cache.swa
        ori_slot = _get_layer_cache_tensor(compressed_metadata.slot_mappings, layer_id, mapping.ori_cache_idx)
        ori_block_table = _get_layer_cache_tensor(compressed_metadata.block_tables, layer_id, mapping.ori_cache_idx)
        if use_temporary_prefill_kv:
            # Prefill: build temporary PA_ND cache from kv (mirrors C++
            # build_prefill_pa_nd_kv, deepseek_sparse_attention.cpp:272-368).
            ori_kv_for_attn, ori_block_table_for_attn = _build_prefill_pa_nd_kv(
                k,
                compressed_metadata.actual_seq_lengths_query,
                ori_block_table,
                self.window_size,
            )
        else:
            if ori_kv is not None and ori_slot is not None:
                _scatter_by_slot(ori_kv, ori_slot, k)
            ori_kv_for_attn = ori_kv
            ori_block_table_for_attn = ori_block_table

        # 2) Compressor: pool KV into the compressed cache when ratio > 1.
        cmp_kv = layer_cache.key
        cmp_slot = _get_layer_cache_tensor(compressed_metadata.slot_mappings, layer_id, mapping.cmp_cache_idx)
        cmp_block_table = _get_layer_cache_tensor(compressed_metadata.block_tables, layer_id, mapping.cmp_cache_idx)
        if compress_ratio > 1 and cmp_kv is not None and cmp_slot is not None:
            compressor_fn = getattr(self, "_compressor_fn", None)
            if compressor_fn is None:
                raise RuntimeError(f"{attention_type} compressor is required")
            compressed = compressor_fn(
                layer_id,
                layer_cache,
                compressed_metadata,
                mapping,
                cmp_block_table,
                compress_ratio,
            )
            _scatter_by_slot(cmp_kv, cmp_slot, compressed)

        # 3) Indexer: select top-k compressed blocks when ratio == 4.
        compress_topk_idxs: torch.Tensor | None = None
        if compress_ratio == 4 and cmp_kv is not None:
            indexer_fn = getattr(self, "_indexer_fn", None)
            if indexer_fn is None:
                raise RuntimeError("CSA indexer is required for compressed_sparse_attention")
            compress_topk_idxs = indexer_fn(
                layer_id,
                layer_cache,
                compressed_metadata,
                mapping,
                q,
            )
            if compress_topk_idxs is None:
                raise RuntimeError("CSA indexer returned no top-k indices")

        # 4) Two-stage sparse attention over original + compressed KV.
        # The metadata tensors live on CPU (DsaMetadataBuilder); move to device
        # for the NPU kernel, matching the C++ H2D transfer of packed metadata.
        if compress_ratio == 1:
            sparse_meta = compressed_metadata.c1_metadata
        elif compress_ratio == 4:
            sparse_meta = compressed_metadata.c4_metadata
        elif compress_ratio == 128:
            sparse_meta = compressed_metadata.c128_metadata
        else:
            sparse_meta = None
        if sparse_meta is None:
            raise RuntimeError(f"sparse metadata is missing for {attention_type}")
        seq_q = compressed_metadata.actual_seq_lengths_query
        seq_kv = compressed_metadata.actual_seq_lengths_kv
        sparse_meta_for_kernel = sparse_meta
        ori_block_table_for_kernel = ori_block_table_for_attn
        cmp_block_table_for_kernel = cmp_block_table
        # Match C++ DSAttention's optional contract exactly: prefill and
        # chunked prefill pass query cu-seqlens, while decode leaves
        # cu_seqlens_ori_kv as std::nullopt. A defined empty tensor selects a
        # different ACL optional-input path and causes small decode drift.
        use_prefill_attn = is_prefill or is_chunked_prefill
        cu_seqlens_ori_kv_for_attn = seq_q if use_prefill_attn else None
        out, _lse = _sparse_attn_sharedkv(
            q=q,
            ori_kv=ori_kv_for_attn,
            cmp_kv=cmp_kv if compress_ratio > 1 else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=compress_topk_idxs,
            ori_block_table=ori_block_table_for_kernel,
            cmp_block_table=cmp_block_table_for_kernel if compress_ratio > 1 else None,
            cu_seqlens_q=seq_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv_for_attn,
            # C++ passes nullopt for compressed KV cu-seqlens; cmp_kv is PA_ND
            # and addressed through cmp_block_table/topk.
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seq_kv,
            # sinks: the attention sink parameter (attn_sink) is required by the
            # sparse_attn_sharedkv kernel (C++ :949 passes attn_sink_ when loaded).
            sinks=layer.attn_sink if hasattr(layer, "attn_sink") else None,
            metadata=sparse_meta_for_kernel,
            softmax_scale=self.scale,
            cmp_ratio=compress_ratio,
            ori_mask_mode=_MASK_MODE_COMPRESS,
            cmp_mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
            ori_win_left=self.window_size - 1,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=False,
        )
        # Full prefill reads a temporary PA_ND cache so attention does not
        # depend on the persistent SWA cache. Match C++ step 8 by writing the
        # projected KV into the persistent cache only after that attention
        # finishes; decode reads this cache on the next forward.
        if use_temporary_prefill_kv and ori_kv is not None and ori_slot is not None:
            _scatter_by_slot(ori_kv, ori_slot, k)
        return out

    def mla_index_context(self, layer: Attention) -> CsaIndexContext:
        """Hand the CSA indexer its paged index cache and current metadata."""
        metadata = self._current_forward_metadata()
        compressed_metadata = getattr(metadata, "dsa_metadata", None)
        assert compressed_metadata is not None
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        index_slot = _get_layer_cache_tensor(compressed_metadata.slot_mappings, layer_id, mapping.index_cache_idx)
        index_block_table = _get_layer_cache_tensor(compressed_metadata.block_tables, layer_id, mapping.index_cache_idx)
        cmp_block_table = _get_layer_cache_tensor(compressed_metadata.block_tables, layer_id, mapping.cmp_cache_idx)
        return CsaIndexContext(
            index_cache=layer_cache.index if layer_cache.index is not None else torch.empty(0),
            indexer_scale=layer_cache.indexer_scale,
            slot_mapping=index_slot if index_slot is not None else torch.empty(0),
            block_table=index_block_table,
            cmp_block_table=cmp_block_table,
            kv_state=layer_cache.compress_kv_state,
            score_state=layer_cache.compress_score_state,
            kv_block_table=_get_layer_cache_tensor(
                compressed_metadata.block_tables, layer_id, mapping.kv_state_cache_idx
            ),
            score_block_table=_get_layer_cache_tensor(
                compressed_metadata.block_tables, layer_id, mapping.score_state_cache_idx
            ),
            actual_seq_q=compressed_metadata.actual_seq_lengths_query,
            actual_seq_kv=compressed_metadata.actual_seq_lengths_kv,
            start_pos=compressed_metadata.start_pos,
            qli_metadata=compressed_metadata.qli_metadata,
        )

    @property
    def num_kv_blocks(self) -> int:
        if self._kv_caches and self._kv_caches[0].swa is not None:
            return self._kv_caches[0].swa.size(0)
        return 0

    @property
    def page_size(self) -> int:
        if self._kv_caches and self._kv_caches[0].swa is not None and self._kv_caches[0].swa.dim() > 1:
            return self._kv_caches[0].swa.size(1)
        return self.window_size

    # -- model-attached state ----------------------------------------------
    # The DeepseekV4 model owns the RoPE tables, Hadamard matrix, and the
    # compressor/indexer callables. It attaches them to the backend before the
    # first forward so the backend can stage them into the runtime DsaMetadata.

    def attach_rope_tables(
        self,
        positions: torch.Tensor,
        base_cos_sin: torch.Tensor | None,
        graph_bt_cols: int = 0,
        csa_cos_sin: torch.Tensor | None = None,
        hca_cos_sin: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        metadata = metadata or self._metadata
        if metadata is not None:
            metadata.dsa_positions = positions
            metadata.dsa_cos_sin = base_cos_sin
            metadata.dsa_c4_cos_sin = csa_cos_sin
            metadata.dsa_c128_cos_sin = hca_cos_sin

    def attach_compressor(self, fn) -> None:
        """Attach the CSA/HCA KV compressor callback."""
        self._compressor_fn = fn

    def attach_indexer(self, fn) -> None:
        """Attach the CSA lightning-indexer callback."""
        self._indexer_fn = fn

    # -- internals ----------------------------------------------------------

    def _layer_compress_ratio(self, layer_id: int) -> int:
        if layer_id < len(self.caches_info):
            caches = self.caches_info[layer_id]
            for ci in caches:
                if ci.cache_type == DSA_CACHE_TOKEN:
                    return ci.ratio
        return 1

    def _resolve_cache_mapping(self, layer_id: int, compress_ratio: int) -> _CompressedAttentionCacheMapping:
        """Python port of ``resolve_cache_mapping`` (deepseek_sparse_attention.cpp:92)."""
        mapping = _CompressedAttentionCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_ratio_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN and ci.ratio == compress_ratio:
                token_ratio_indices.append(cache_idx)
            if ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_ratio_indices and compress_ratio > 1:
            mapping.cmp_cache_idx = token_ratio_indices[0]
        if len(token_ratio_indices) > 1:
            mapping.index_cache_idx = token_ratio_indices[1]
        if len(token_ratio_indices) > 2:
            mapping.indexer_scale_cache_idx = token_ratio_indices[2]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        if len(swa_indices) > 3:
            mapping.index_kv_state_cache_idx = swa_indices[3]
        if len(swa_indices) > 4:
            mapping.index_score_state_cache_idx = swa_indices[4]
        return mapping

    def _build_precomputed_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: AttentionMetadata,
    ) -> None:
        """Build the AICPU tiling metadata for each compress ratio present.

        Mirrors the C++ ``build_precomputed_metadata`` step: one
        ``sparse_attn_sharedkv_metadata`` per ratio, plus one
        ``quant_lightning_indexer_metadata`` for the qli path.
        """
        from xllm.python import kernels

        seq_q = compressed_metadata.actual_seq_lengths_query
        seq_kv = compressed_metadata.actual_seq_lengths_kv
        batch_size = int(max(compressed_metadata.actual_seq_lengths_kv.numel(), 1))
        forward_meta = _build_compressed_attention_forward_meta(compressed_metadata, metadata)
        max_q = forward_meta.q_max_seq_len
        max_kv = forward_meta.kv_max_seq_len
        is_prefill = max_q > 1
        empty_int32 = torch.empty(0, dtype=torch.int32, device=self.device)
        cu_seqlens_ori_kv = seq_q if is_prefill else empty_int32
        cu_seqlens_cmp_kv = empty_int32
        seqused_q = empty_int32
        seqused_kv = seq_kv
        # Metadata kernels enqueue asynchronously. Retain their tensor inputs
        # on the current forward's DsaMetadata, as C++ DSAMetadata does.
        compressed_metadata.precomputed_metadata_inputs = tuple(
            (seq_q, seq_kv, cu_seqlens_ori_kv, cu_seqlens_cmp_kv, seqused_q, seqused_kv)
        )
        for ratio in (1, 4, 128):
            has_cmp = ratio > 1
            cmp_topk = self.index_topk if ratio == 4 else 0
            sparse_metadata = kernels.sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=seq_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
                seqused_q=seqused_q,
                seqused_kv=seqused_kv,
                batch_size=batch_size,
                max_seqlen_q=max_q,
                max_seqlen_kv=max_kv,
                ori_topk=0,
                cmp_topk=cmp_topk,
                cmp_ratio=ratio,
                ori_mask_mode=_MASK_MODE_COMPRESS,
                cmp_mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
                ori_win_left=max(self.window_size - 1, 0),
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=has_cmp,
            )
            if ratio == 1:
                compressed_metadata.c1_metadata = sparse_metadata
            elif ratio == 4:
                compressed_metadata.c4_metadata = sparse_metadata
            elif ratio == 128:
                compressed_metadata.c128_metadata = sparse_metadata
        query_lens = seq_q[1:].clone() if seq_q.numel() > 1 else compressed_metadata.seq_lens_q
        key_lens = compressed_metadata.seq_lens if compressed_metadata.seq_lens.numel() else seq_kv
        compressed_metadata.precomputed_metadata_inputs += (query_lens, key_lens)
        compressed_metadata.qli_metadata = kernels.quant_lightning_indexer_metadata(
            num_heads_q=max(self.index_n_heads, 1),
            num_heads_k=1,
            head_dim=max(self.index_head_dim, 1),
            query_quant_mode=0,
            key_quant_mode=0,
            actual_seq_lengths_query=query_lens,
            actual_seq_lengths_key=key_lens,
            batch_size=int(max(key_lens.size(0), 1)),
            max_seqlen_q=max(max_q, 1),
            max_seqlen_k=max(max_kv, 1),
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=self.index_topk,
            sparse_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
            pre_tokens=2**63 - 1,
            next_tokens=2**63 - 1,
            cmp_ratio=4,
            device=str(self.device),
        )

    def _move_metadata_to_device(self, compressed_metadata: DsaMetadata) -> None:
        """Mirror ``deepseek_v4_move_dsa_metadata_to_device`` for eager mode."""
        tensor_fields = (
            "seq_lens",
            "seq_lens_q",
            "actual_seq_lengths_query",
            "actual_seq_lengths_kv",
            "kv_cu_seq_lens",
            "max_seqlen_q",
            "max_seqlen_kv",
            "input_positions",
            "c4_pad_positions",
            "c128_pad_positions",
            "start_pos",
            "hadamard",
        )
        for name in tensor_fields:
            tensor = getattr(compressed_metadata, name, None)
            if tensor is not None:
                setattr(compressed_metadata, name, tensor.to(self.device))
        for layer_tensors in compressed_metadata.block_tables:
            for index, tensor in enumerate(layer_tensors):
                layer_tensors[index] = tensor.to(self.device)
        for layer_tensors in compressed_metadata.slot_mappings:
            for index, tensor in enumerate(layer_tensors):
                layer_tensors[index] = tensor.to(self.device)


# ---------------------------------------------------------------------------
# Helpers (faithful ports of C++ free functions).
# ---------------------------------------------------------------------------


def _tensor_max_or_zero(tensor: torch.Tensor | None) -> int:
    if tensor is None or tensor.numel() == 0:
        return 0
    return int(tensor.max().item())


def _build_compressed_attention_forward_meta(
    compressed_metadata: DsaMetadata,
    metadata: AttentionMetadata,
) -> _CompressedAttentionForwardMeta:
    """Mirror the C++ max-seqlen inputs used by build_precomputed_metadata.

    C++ computes sparse metadata max sizes from ModelInputParams::meta plus the
    host q/kv length vectors:
      max(params.meta.q_max_seq_len, max(host.q_seq_lens))
      max(params.meta.kv_max_seq_len, max(host.kv_seq_lens))
    """

    q_max = int(getattr(metadata, "max_query_len", compressed_metadata.max_query_len))
    kv_max = int(getattr(metadata, "max_seq_len", compressed_metadata.max_seq_len))
    q_max = max(q_max, _tensor_max_or_zero(getattr(metadata, "q_seq_lens_host", None)))
    kv_max = max(kv_max, _tensor_max_or_zero(getattr(metadata, "kv_seq_lens_host", None)))
    q_max = max(q_max, int(compressed_metadata.max_query_len))
    kv_max = max(kv_max, int(compressed_metadata.max_seq_len))
    return _CompressedAttentionForwardMeta(q_max_seq_len=q_max, kv_max_seq_len=kv_max)


def _build_prefill_pa_nd_kv(
    kv: torch.Tensor,
    cu_seqlens: torch.Tensor,
    block_table_hint: torch.Tensor | None,
    block_size: int,
    cu_seqlens_dst: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Python port of C++ build_prefill_pa_nd_kv (deepseek_sparse_attention.cpp:272-368).

    Builds a temporary PA_ND format KV cache from the current forward's kv
    tensor, for full prefill attention (no paged cache needed).
    """
    if kv is None or cu_seqlens is None or cu_seqlens.numel() <= 1 or block_size <= 0:
        return torch.empty(0), torch.empty(0)

    batch_size = cu_seqlens.numel() - 1
    cu_cpu = cu_seqlens.to(torch.device("cpu")).to(torch.int64)
    cu = cu_cpu.tolist()

    dst_cu = None
    if cu_seqlens_dst is not None and cu_seqlens_dst.numel() == batch_size + 1:
        dst_cu = cu_seqlens_dst.to(torch.device("cpu")).to(torch.int64).tolist()

    # Compute per-request lengths and block counts.
    dst_lens = []
    total_blocks = 0
    max_blocks_per_req = 0
    for i in range(batch_size):
        q_len = dst_cu[i + 1] - dst_cu[i] if dst_cu is not None else cu[i + 1] - cu[i]
        dst_lens.append(q_len)
        blocks = (q_len + block_size - 1) // block_size
        total_blocks += blocks
        max_blocks_per_req = max(max_blocks_per_req, blocks)

    if total_blocks <= 0:
        return torch.empty(0), torch.empty(0)

    table_cols = max(
        block_table_hint.size(1) if block_table_hint is not None and block_table_hint.dim() > 1 else 0,
        max_blocks_per_req,
    )

    # Block 0 is zero-filled padding block; real blocks are 1-based.
    packed_kv = torch.zeros(
        total_blocks + 1,
        block_size,
        kv.size(1),
        kv.size(2),
        dtype=kv.dtype,
        device=kv.device,
    )

    table_data = [0] * (batch_size * table_cols)
    next_block = 1
    for req in range(batch_size):
        q_start = cu[req]
        src_len = cu[req + 1] - q_start
        q_len = dst_lens[req]
        blocks = (q_len + block_size - 1) // block_size
        if q_len <= 0 or blocks <= 0:
            continue
        for j in range(blocks):
            table_data[req * table_cols + j] = next_block + j
        copy_len = min(q_len, src_len)
        if copy_len > 0:
            target = packed_kv[next_block : next_block + blocks].view(blocks * block_size, kv.size(1), kv.size(2))
            target[q_len - copy_len : q_len].copy_(kv[q_start : q_start + copy_len])
        next_block += blocks

    table = torch.tensor(table_data, dtype=torch.int32, device=kv.device).view(batch_size, table_cols)
    return packed_kv, table


def _get_layer_cache_tensor(
    layer_tensors: list[list[torch.Tensor]],
    layer_id: int,
    cache_idx: int,
) -> torch.Tensor | None:
    """Python port of ``get_layer_cache_tensor`` (deepseek_sparse_attention.cpp:80)."""
    if layer_id < 0 or layer_id >= len(layer_tensors) or cache_idx < 0 or cache_idx >= len(layer_tensors[layer_id]):
        return None
    return layer_tensors[layer_id][cache_idx]


def _scatter_by_slot(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Python port of ``scatter_by_slot`` (deepseek_sparse_attention.cpp:200).

    Writes ``value`` rows into the paged ``cache`` at the physical slots given by
    ``slot_mapping`` (= block_id * block_size + offset).
    """
    if (
        cache is None
        or cache.numel() == 0
        or slot_mapping is None
        or slot_mapping.numel() == 0
        or value is None
        or value.numel() == 0
    ):
        return
    value_2d = value.reshape(-1, value.size(-1))
    cache_2d = cache.view(-1, value_2d.size(1))
    slots = slot_mapping.reshape(-1).to(torch.long).to(cache.device)
    update_rows = min(slots.size(0), value_2d.size(0))
    if update_rows <= 0:
        return
    valid = slots[:update_rows] >= 0
    if cache.device.type == "npu":
        # Match C++ scatter_by_slot exactly. The dedicated NPU kernel preserves
        # the cache's storage/layout semantics; Tensor.index_copy_ selects a
        # different implementation and left the persistent SWA cache invalid
        # for the first decode step.
        from xllm.python import kernels

        slots_slice = slots[:update_rows]
        safe_slots = slots_slice.clamp_min(0)
        valid_mask = slots_slice.ge(0).unsqueeze(1)
        old_values = cache_2d.index_select(0, safe_slots)
        safe_values = torch.where(
            valid_mask,
            value_2d[:update_rows].to(cache.dtype),
            old_values,
        )
        kernels.scatter_nd_update(
            cache_2d,
            safe_slots.reshape(-1, 1),
            safe_values,
        )
        return
    if not valid.any():
        return
    cache_2d.index_copy_(
        0,
        slots[:update_rows][valid],
        value_2d[:update_rows][valid].to(cache.dtype),
    )


def _sparse_attn_sharedkv(**kwargs):
    """Thin indirection so the backend can be unit-tested without the kernel."""
    from xllm.python import kernels

    return kernels.sparse_attn_sharedkv(**kwargs)
