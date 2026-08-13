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

"""DeepSeek-V4 DSA attention backend.

Consumes the :class:`DsaMetadata` built by :mod:`dsa_metadata` and drives the
two-stage sparse attention (``sparse_attn_sharedkv``), the KV compressor, and
the quantized lightning indexer. This is the Python-path counterpart of the
C++ ``DSAttentionImpl`` (core/layers/npu_torch/deepseek_sparse_attention.cpp).

The backend owns no KV storage: caches are bound from the C++ executor's
``LayerCache`` 11-tuple. Per step it (1) builds DSA metadata from the framework
``multi_block_tables``, (2) resolves the per-layer 8-cache mapping, (3) writes
new KV into the SWA cache, (4) runs the compressor into the compressed cache
when ``compress_ratio > 1``, (5) runs the indexer to pick top-k compressed
blocks when ``compress_ratio == 4``, and (6) calls ``sparse_attn_sharedkv``.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING

import torch
import torch_npu

from xllm.python.attention.backend import (
    AttentionBackend,
    DsaIndexContext,
    LayerCache,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    DSACacheInfo,
    DsaMetadata,
    DsaMetadataBuilder,
    build_cache_specs,
)
from xllm.python.model_executor.forward_context import get_forward_context

if TYPE_CHECKING:
    from xllm.python.layers.attention import Attention
    from xllm.python.attention.backend import AttentionMetadata

# Sparse mask modes used by the C++ DSA attention (rightDownCausal variants).
_MASK_MODE_RIGHT_DOWN_CAUSAL = 3
_MASK_MODE_COMPRESS = 4


@dataclass
class _DsaCacheMapping:
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
class _DsaForwardMeta:
    """Subset of C++ ModelInputParams::meta used by DSV4 metadata builders."""

    q_max_seq_len: int
    kv_max_seq_len: int


class DsaAttentionBackend(AttentionBackend):
    """DSA attention backend for DeepSeek-V4 on NPU.

    The model supplies its config so the backend can rebuild the static
    ``caches_info`` / ``group_infos`` once (mirroring
    ``deepseek_v4_build_cache_specs``) and precompute the per-step RoPE / Hadamard
    tables the indexer needs.
    """

    def __init__(
        self,
        compress_ratios: list[int],
        window_size: int,
        n_layers: int,
        num_heads: int,
        attn_head_dim: int,
        metadata_head_dim: int,
        index_topk: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs(
            compress_ratios, window_size, n_layers
        )
        self._builder = DsaMetadataBuilder(self.caches_info, self.group_infos)
        self.window_size = window_size
        self.index_topk = index_topk
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_head_dim = rope_head_dim
        self.num_heads = num_heads
        self.head_dim = attn_head_dim
        self.metadata_head_dim = metadata_head_dim
        self.device = device
        self.dtype = dtype
        self.scale = attn_head_dim ** -0.5

        self._kv_caches: list[LayerCache] = []
        self._metadata: AttentionMetadata | None = None
        self._dsa_metadata: DsaMetadata | None = None
        self._graph_mode = False
        # Legacy mirrors of the current DsaMetadata fields. The formal owner is
        # the per-forward DsaMetadata object, matching C++ DSAMetadata.
        self._sparse_metadata: dict[int, torch.Tensor] = {}
        self._qli_metadata: torch.Tensor | None = None

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
        self._metadata = metadata
        self._graph_mode = graph_mode
        metadata.dsa_graph_mode = graph_mode

    def prepare_dsa_metadata_for_forward(
        self,
        metadata: AttentionMetadata | None = None,
        *,
        force: bool = False,
    ) -> None:
        """Build DSA metadata inside model forward, matching C++ ownership/order."""
        metadata = metadata or self._metadata
        assert metadata is not None
        graph_mode = bool(getattr(metadata, "dsa_graph_mode", self._graph_mode))
        if graph_mode and metadata.dsa_metadata is not None and not force:
            return
        multi_block_tables = list(metadata.multi_block_tables)
        kv_seq_lens_host = metadata.kv_seq_lens_host
        kv_seq_lens = (
            kv_seq_lens_host.cpu().tolist()
            if kv_seq_lens_host is not None and kv_seq_lens_host.numel() > 0
            else []
        )
        q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
        q_seq_lens = (
            q_seq_lens_host.cpu().tolist()
            if q_seq_lens_host is not None and q_seq_lens_host.numel() > 0
            else None
        )
        # DSA RoPE tables and positions are model-owned; the backend reads them
        # off the metadata when the model attaches them (see attach_rope_tables).
        positions = getattr(metadata, "dsa_positions", None)
        if positions is None:
            positions = torch.empty(0, dtype=torch.int64)
        dsa_cos_sin = getattr(metadata, "dsa_cos_sin", None)
        graph_bt_cols = int(
            getattr(metadata, "dsa_graph_block_table_cols", 0)
        )
        dsa_metadata = self._builder.build(
            multi_block_tables=multi_block_tables,
            kv_seq_lens=kv_seq_lens,
            q_seq_lens=q_seq_lens,
            positions=positions,
            dsa_cos_sin=dsa_cos_sin,
            is_prefill=metadata.is_prefill,
            is_chunked_prefill=metadata.is_chunked_prefill,
            enable_graph=graph_mode,
            graph_block_table_capacity_cols=graph_bt_cols,
        )
        self._populate_dsa_rope(dsa_metadata, metadata)
        self._pack_metadata_to_device(dsa_metadata)
        self._build_precomputed_metadata(dsa_metadata, metadata)
        if graph_mode and metadata.dsa_metadata is not None:
            self._update_persistent_dsa_metadata(
                metadata.dsa_metadata, dsa_metadata
            )
            dsa_metadata = metadata.dsa_metadata
        else:
            metadata.dsa_metadata = dsa_metadata
        # Keep a temporary alias for existing backend helpers. Ownership belongs
        # to the current AttentionMetadata, as in C++.
        self._dsa_metadata = metadata.dsa_metadata

    def prepare_graph_forward_metadata(
        self, metadata: AttentionMetadata | None = None
    ) -> None:
        """Refresh persistent DSA graph inputs before capture or replay."""
        metadata = metadata or self._metadata
        if metadata is None or not bool(
            getattr(metadata, "dsa_graph_mode", self._graph_mode)
        ):
            return
        self.prepare_dsa_metadata_for_forward(metadata, force=True)
        # Sparse/QLI metadata is produced by asynchronous AICPU kernels and is
        # consumed immediately by the captured model graph.  Establish the
        # same prepare-before-capture boundary as the C++ graph path.  This is
        # deliberately outside model.forward and therefore outside capture.
        torch_npu.npu.synchronize()

    def rebuild_cp_local_query_metadata(
        self,
        metadata: AttentionMetadata,
        cp_context,
    ) -> None:
        """Localize only the DSV4 query/read metadata for prefill CP.

        Block tables, slot mappings, compressed positions and ``start_pos``
        intentionally remain global because every CP rank writes a complete KV
        replica. This mirrors DeepseekV4ModelImpl::rebuild_cp_local_query_metadata.
        """
        dsa = metadata.dsa_metadata
        if dsa is None:
            raise RuntimeError("DSA metadata must exist before CP localization")
        device = dsa.seq_lens_q.device
        q_lens = torch.tensor(
            cp_context.local_q_seq_lens, dtype=torch.int32, device=device
        )
        kv_lens = torch.tensor(
            cp_context.local_kv_seq_lens, dtype=torch.int32, device=device
        )
        zero = torch.zeros(1, dtype=torch.int32, device=device)
        dsa.seq_lens_q = q_lens
        dsa.actual_seq_lengths_query = torch.cat(
            (zero, q_lens.cumsum(0, dtype=torch.int32))
        )
        dsa.seq_lens = kv_lens
        dsa.actual_seq_lengths_kv = kv_lens
        dsa.kv_cu_seq_lens = cp_context.local_kv_cu_seq_lens.to(device)
        dsa.max_seqlen_q = (
            q_lens.max().reshape(1) if q_lens.numel() else zero.clone()
        )
        dsa.max_seqlen_kv = (
            kv_lens.max().reshape(1) if kv_lens.numel() else zero.clone()
        )
        dsa.max_query_len = max(cp_context.local_q_seq_lens, default=0)
        dsa.max_seq_len = max(cp_context.local_kv_seq_lens, default=0)
        dsa.input_positions = cp_context.local_positions
        dsa.v4_cp_context = cp_context

        # Query RoPE follows the local rows. Compressed c4/c128 RoPE remains
        # global because the compressor and index-cache write paths are global.
        metadata.dsa_positions = cp_context.local_positions
        self._build_precomputed_metadata(
            dsa,
            metadata,
            forward_meta_override=_DsaForwardMeta(
                q_max_seq_len=dsa.max_query_len,
                kv_max_seq_len=dsa.max_seq_len,
            ),
            cu_seqlens_ori_kv_override=cp_context.local_kv_cu_seq_lens,
        )

    @staticmethod
    def _copy_tensor(dst: torch.Tensor | None, src: torch.Tensor | None) -> None:
        if dst is None or src is None:
            if dst is not src:
                raise RuntimeError("DSA graph tensor definition changed")
            return
        if dst.shape != src.shape or dst.dtype != src.dtype:
            raise RuntimeError(
                "DSA graph tensor contract changed: "
                f"{tuple(dst.shape)}/{dst.dtype} != {tuple(src.shape)}/{src.dtype}"
            )
        dst.copy_(src.to(dst.device))

    def _update_persistent_dsa_metadata(
        self, persistent: DsaMetadata, current: DsaMetadata
    ) -> None:
        tensor_fields = (
            "seq_lens", "seq_lens_q", "actual_seq_lengths_kv",
            "actual_seq_lengths_query", "kv_cu_seq_lens", "max_seqlen_kv",
            "max_seqlen_q", "input_positions", "c4_pad_positions",
            "c128_pad_positions", "start_pos",
        )
        for name in tensor_fields:
            self._copy_tensor(getattr(persistent, name), getattr(current, name))
        if len(persistent.block_tables) != len(current.block_tables):
            raise RuntimeError("DSA graph layer count changed")
        for dst_layer, src_layer in zip(
            persistent.block_tables, current.block_tables
        ):
            if len(dst_layer) != len(src_layer):
                raise RuntimeError("DSA graph cache count changed")
            for dst, src in zip(dst_layer, src_layer):
                self._copy_tensor(dst, src)
        for dst_layer, src_layer in zip(
            persistent.slot_mappings, current.slot_mappings
        ):
            for dst, src in zip(dst_layer, src_layer):
                self._copy_tensor(dst, src)
        persistent.max_query_len = current.max_query_len
        persistent.max_seq_len = current.max_seq_len
        persistent.precomputed_metadata_inputs = (
            current.precomputed_metadata_inputs
        )
        persistent.is_acl_graph = True
        for name in (
            "c4_cos", "c4_sin", "c128_cos", "c128_sin",
        ):
            dst = getattr(persistent, name)
            src = getattr(current, name)
            self._copy_tensor(dst, src)
        for name in (
            "c1_metadata", "c4_metadata", "c128_metadata", "qli_metadata"
        ):
            dst = getattr(persistent, name)
            src = getattr(current, name)
            if dst is None:
                setattr(persistent, name, src)
            else:
                self._copy_tensor(dst, src)
        self._sparse_metadata = {
            ratio: tensor
            for ratio, tensor in (
                (1, persistent.c1_metadata),
                (4, persistent.c4_metadata),
                (128, persistent.c128_metadata),
            )
            if tensor is not None
        }
        self._qli_metadata = persistent.qli_metadata

    def select_dsa_layer_rope(
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
            raise RuntimeError("DSA metadata must be prepared before selecting layer RoPE")
        dsa = metadata.dsa_metadata
        chunks = cos_sin_cache.chunk(2, dim=-1)
        dsa.layer_id = layer_id
        dsa.cos_table = chunks[0].contiguous()
        dsa.sin_table = chunks[1].contiguous()

    def _populate_dsa_rope(
        self,
        dsa: DsaMetadata,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build request-shaped RoPE tensors before graph capture/replay."""
        metadata = metadata or self._metadata
        css = (
            getattr(metadata, "dsa_cos_sin", None)
            if metadata is not None
            else None
        )
        if dsa.cos_table is None and css is not None and css.numel() > 0:
            dsa.cos_table, dsa.sin_table = (
                tensor.contiguous() for tensor in css.chunk(2, dim=-1)
            )
        c4css = (
            getattr(metadata, "dsa_c4_cos_sin", None)
            if metadata is not None
            else None
        )
        if c4css is not None and dsa.c4_pad_positions.numel() > 0:
            c4_idx = dsa.c4_pad_positions.clamp_min(0).long().to(c4css.device)
            dsa.c4_cos, dsa.c4_sin = (
                tensor.contiguous()
                for tensor in c4css.index_select(0, c4_idx).chunk(2, dim=-1)
            )
        c128css = (
            getattr(metadata, "dsa_c128_cos_sin", None)
            if metadata is not None
            else None
        )
        if c128css is not None and dsa.c128_pad_positions.numel() > 0:
            c128_idx = dsa.c128_pad_positions.clamp_min(0).long().to(
                c128css.device
            )
            dsa.c128_cos, dsa.c128_sin = (
                tensor.contiguous()
                for tensor in c128css.index_select(0, c128_idx).chunk(2, dim=-1)
            )

    def execute(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: Attention,
    ) -> torch.Tensor:
        """Full DSA attention path for one layer.

        ``q``/``k``/``v`` here are the model-projected, RoPE-applied tensors the
        DeepseekV4 attention layer hands in; the backend only owns cache writes
        and the kernel dispatch.
        """
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        assert dsa is not None
        # Late-populate dsa.cos_table/sin_table if prepare ran before the model
        # attached the RoPE tables (prepare is called by the executor before
        # model.forward, so _dsa_cos_sin may have been None at prepare time).
        if dsa.cos_table is None or dsa.sin_table is None:
            self._populate_dsa_rope(dsa, metadata)
        # Late-populate per-ratio compressed RoPE tables: index the c4/c128
        # compress RoPE cache with the per-token compressed positions
        # (c4_pad_positions / c128_pad_positions, built by DsaMetadataBuilder).
        # Mirrors C++ DeepseekV4RotaryEmbedding::build(positions_map) per group.
        # Also late-populate dsa.input_positions (prepare ran before model.forward
        # set self._positions, so it was empty at prepare time).
        if dsa.input_positions.numel() == 0:
            pos = getattr(metadata, "dsa_positions", None)
            if pos is not None and pos.numel() > 0:
                dsa.input_positions = pos
        if dsa.c4_cos is None or dsa.c128_cos is None:
            self._populate_dsa_rope(dsa, metadata)
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        is_prefill = metadata.is_prefill
        is_chunked_prefill = metadata.is_chunked_prefill
        use_temporary_prefill_kv = is_prefill and not is_chunked_prefill
        cp_context = getattr(dsa, "v4_cp_context", None)
        cp_enabled = cp_context is not None and cp_context.enabled

        # 1) Prepare ori_kv for attention (mirrors C++ :790-816).
        # Full prefill: use kv directly as temporary PA_ND (don't scatter to paged).
        # Decode/chunked: scatter to paged SWA cache.
        ori_kv = layer_cache.swa
        ori_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        ori_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.ori_cache_idx
        )
        if use_temporary_prefill_kv:
            # Prefill: build temporary PA_ND cache from kv (mirrors C++
            # build_prefill_pa_nd_kv, deepseek_sparse_attention.cpp:272-368).
            ori_kv_for_attn, ori_block_table_for_attn = _build_prefill_pa_nd_kv(
                k,
                (
                    cp_context.global_q_cu_seq_lens
                    if cp_enabled
                    else dsa.actual_seq_lengths_query
                ),
                ori_block_table,
                self.window_size,
                (
                    cp_context.local_kv_cu_seq_lens
                    if cp_enabled
                    else None
                ),
            )
        else:
            if ori_kv is not None and ori_slot is not None:
                _scatter_by_slot(ori_kv, ori_slot, k)
            ori_kv_for_attn = ori_kv
            ori_block_table_for_attn = ori_block_table

        # 2) Compressor: pool KV into the compressed cache when ratio > 1.
        cmp_kv = layer_cache.key
        cmp_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.cmp_cache_idx)
        cmp_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.cmp_cache_idx
        )
        if (
            compress_ratio > 1
            and cmp_kv is not None
            and cmp_slot is not None
        ):
            # The model-side compressor module runs the kernel and writes cmp_kv;
            # the backend only stages the block table / slot for it. When the
            # model has not attached a compressor callable, fall back to a plain
            # scatter of the uncompressed KV so the path stays end-to-end.
            compressor_fn = getattr(self, "_compressor_fn", None)
            if compressor_fn is not None:
                compressed = compressor_fn(
                    layer_id,
                    layer_cache,
                    dsa,
                    mapping,
                    cmp_block_table,
                    compress_ratio,
                )
                if (
                    os.getenv("XLLM_DSV4_ZERO_ACTIVE_CMP_BLOCKS", "0") == "1"
                    and cmp_block_table is not None
                    and cmp_block_table.numel() > 0
                ):
                    block_ids = torch.unique(
                        cmp_block_table.reshape(-1).to(torch.long)
                    ).to(cmp_kv.device)
                    block_ids = block_ids[
                        (block_ids >= 0) & (block_ids < cmp_kv.size(0))
                    ]
                    if block_ids.numel() > 0:
                        cmp_kv.index_fill_(0, block_ids, 0)
                _scatter_by_slot(cmp_kv, cmp_slot, compressed)
            else:
                _scatter_by_slot(cmp_kv, cmp_slot, k)

        # 3) Indexer: select top-k compressed blocks when ratio == 4.
        compress_topk_idxs: torch.Tensor | None = None
        if compress_ratio == 4 and cmp_kv is not None:
            indexer_fn = getattr(self, "_indexer_fn", None)
            if indexer_fn is not None:
                compress_topk_idxs = indexer_fn(
                    layer_id,
                    layer_cache,
                    dsa,
                    mapping,
                    q,
                )
                if (
                    compress_topk_idxs is not None
                    and q.shape[0] > 1
                    and os.getenv("XLLM_DSV4_ATTENTION_NUMERIC_DEBUG") == "1"
                    and not torch.npu.is_current_stream_capturing()
                ):
                    import sys as _topk_sys
                    _topk_cpu = compress_topk_idxs.detach().to("cpu").reshape(-1)
                    print(
                        f"[DSV4_NUMERIC] layer_{layer_id}_compress_topk_idxs "
                        f"shape={tuple(compress_topk_idxs.shape)} "
                        f"sum={_topk_cpu.to(torch.float64).sum().item():.12e} "
                        f"first={_topk_cpu[:8].tolist()}",
                        file=_topk_sys.stderr,
                        flush=True,
                    )

        # 4) Two-stage sparse attention over original + compressed KV.
        # The metadata tensors live on CPU (DsaMetadataBuilder); move to device
        # for the NPU kernel, matching the C++ H2D transfer of packed metadata.
        if compress_ratio == 1:
            sparse_meta = dsa.c1_metadata
        elif compress_ratio == 4:
            sparse_meta = dsa.c4_metadata
        elif compress_ratio == 128:
            sparse_meta = dsa.c128_metadata
        else:
            sparse_meta = None
        if (
            os.getenv("XLLM_DSV4_ATTENTION_NUMERIC_DEBUG") == "1"
            and layer_id in (0, 2)
            and q.shape[0] > 1
            and not torch.npu.is_current_stream_capturing()
        ):
            import sys as _sparse_diag_sys

            def _sparse_diag(name, tensor):
                if tensor is None:
                    print(f"[DSV4_NUMERIC] {name} undefined", file=_sparse_diag_sys.stderr, flush=True)
                    return
                values = tensor.detach().to("cpu", dtype=torch.float64).reshape(-1)
                print(
                    f"[DSV4_NUMERIC] {name} shape={tuple(tensor.shape)} "
                    f"dtype={tensor.dtype} sum={values.sum().item():.12e} "
                    f"first={values[:8].tolist()}",
                    file=_sparse_diag_sys.stderr,
                    flush=True,
                )

            _sparse_diag("layer_%d_sparse_ori_kv" % layer_id, ori_kv_for_attn)
            _sparse_diag("layer_%d_sparse_cmp_kv" % layer_id, cmp_kv)
            _sparse_diag(
                "layer_%d_sparse_ori_block_table" % layer_id,
                ori_block_table_for_attn,
            )
            _sparse_diag(
                "layer_%d_sparse_cmp_block_table" % layer_id,
                cmp_block_table,
            )
            _sparse_diag("layer_%d_sparse_metadata" % layer_id, sparse_meta)
        if (
            os.getenv("XLLM_DSV4_CACHE_DEBUG") == "1"
            and layer_id == 2
            and compress_ratio == 4
        ):
            import sys as _cache_diag_sys

            def _cache_diag(name, tensor):
                if tensor is None:
                    print(f"[DSV4_CACHE] {name} undefined", file=_cache_diag_sys.stderr, flush=True)
                    return
                values = tensor.detach().to("cpu", dtype=torch.float64).reshape(-1)
                print(
                    f"[DSV4_CACHE] {name} shape={tuple(tensor.shape)} "
                    f"stride={tuple(tensor.stride())} device={tensor.device} "
                    f"sum={values.sum().item():.12e} "
                    f"first={values[:8].tolist()}",
                    file=_cache_diag_sys.stderr,
                    flush=True,
                )

            _cache_diag("cmp_kv", cmp_kv)
            _cache_diag("cmp_block_table", cmp_block_table)
            _cache_diag("cmp_slots", cmp_slot)
            if cmp_block_table is not None and cmp_block_table.numel() > 0:
                block_id = int(cmp_block_table.reshape(-1)[0].item())
                if 0 <= block_id < cmp_kv.size(0):
                    _cache_diag(f"cmp_block_{block_id}", cmp_kv[block_id])
            _cache_diag("topk", compress_topk_idxs)
        spmeta_device_env = os.getenv("XLLM_DSV4_SPMETA_DEVICE", "npu")
        spmeta_device = (
            torch.device("cpu") if spmeta_device_env == "cpu" else self.device
        )
        seq_q = dsa.actual_seq_lengths_query.to(spmeta_device)
        seq_kv = dsa.actual_seq_lengths_kv.to(spmeta_device)
        if torch.jit.is_scripting():
            sparse_meta_for_kernel = sparse_meta
            ori_block_table_for_kernel = ori_block_table_for_attn
            cmp_block_table_for_kernel = cmp_block_table
        else:
            # Python torch.ops path uses device-side metadata consistently with
            # the DSA compressor and quant_lightning_indexer wrappers.
            sparse_meta_for_kernel = (
                sparse_meta.to(self.device) if sparse_meta is not None else None
            )
            ori_block_table_for_kernel = (
                ori_block_table_for_attn.to(self.device)
                if ori_block_table_for_attn is not None
                else None
            )
            cmp_block_table_for_kernel = (
                cmp_block_table.to(self.device) if cmp_block_table is not None else None
            )
        # Match C++ DSAttention's optional contract exactly: prefill and
        # chunked prefill pass query cu-seqlens, while decode leaves
        # cu_seqlens_ori_kv as std::nullopt. A defined empty tensor selects a
        # different ACL optional-input path and causes small decode drift.
        use_prefill_attn = is_prefill or is_chunked_prefill
        cu_seqlens_ori_kv_for_attn = (
            seq_q
            if use_prefill_attn
            else None
        )
        decode_detail = False
        detail_dir = os.getenv("XLLM_DSV4_DECODE_DETAIL_DIR")
        detail_position = os.getenv("XLLM_DSV4_DECODE_DETAIL_POSITION")
        try:
            detail_layer = int(os.getenv("XLLM_DSV4_DECODE_DETAIL_LAYER", "0"))
            expected_position = int(detail_position) if detail_position is not None else -1
            decode_detail = (
                bool(detail_dir)
                and layer_id == detail_layer
                and dsa.input_positions.numel() == 1
                and os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0") == "0"
                and not torch.npu.is_current_stream_capturing()
                and int(dsa.input_positions.detach().to("cpu").reshape(-1)[0].item())
                == expected_position
            )
        except ValueError:
            decode_detail = False

        def _save_decode_detail(name, tensor):
            if not decode_detail or tensor is None:
                return
            from pathlib import Path

            path = Path(detail_dir)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                tensor.detach().to("cpu"),
                path / f"py_layer_{layer_id}_{name}.pt",
            )

        _save_decode_detail("sparse_ori_kv", ori_kv_for_attn)
        _save_decode_detail("sparse_ori_block_table", ori_block_table_for_kernel)
        _save_decode_detail("sparse_seq_q", seq_q)
        _save_decode_detail("sparse_seq_kv", seq_kv)
        _save_decode_detail("sparse_sinks", layer.attn_sink if hasattr(layer, "attn_sink") else None)
        _save_decode_detail("sparse_metadata", sparse_meta_for_kernel)
        if os.getenv("XLLM_DSV4_SPARSE_DEBUG", "0") == "1":
            import sys as _sys
            print(
                f"[SPARSE BEFORE L{layer_id}] q={tuple(q.shape)} "
                f"ori={None if ori_kv_for_attn is None else tuple(ori_kv_for_attn.shape)} "
                f"cmp={None if cmp_kv is None else tuple(cmp_kv.shape)} "
                f"meta={None if sparse_meta_for_kernel is None else tuple(sparse_meta_for_kernel.shape)}",
                file=_sys.stderr,
                flush=True,
            )
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
        dump_dir = os.getenv("XLLM_DSV4_SPARSE_DUMP_DIR")
        try:
            dump_layer = int(os.getenv("XLLM_DSV4_NUMERIC_LAYER", "2"))
        except ValueError:
            dump_layer = 2
        if (
            dump_dir
            and layer_id == dump_layer
            and q.shape[0] > 1
            and os.getenv("ASCEND_RT_VISIBLE_DEVICES", "0") == "0"
        ):
            from pathlib import Path

            dump_path = Path(dump_dir)
            dump_path.mkdir(parents=True, exist_ok=True)

            def _save(name, tensor):
                if tensor is not None:
                    torch.save(
                        tensor.detach().to("cpu"),
                        dump_path / f"py_layer_{layer_id}_{name}.pt",
                    )

            _save("q", q)
            _save("ori_kv", ori_kv_for_attn)
            if cmp_kv is not None and cmp_kv.size(0) > 1:
                _save("cmp_block_1", cmp_kv[1])
            _save("topk", compress_topk_idxs)
            _save("ori_bt", ori_block_table_for_attn)
            _save("cmp_bt", cmp_block_table)
            _save("seq_q", seq_q)
            _save("seq_kv", seq_kv)
            _save("sinks", layer.attn_sink if hasattr(layer, "attn_sink") else None)
            _save("metadata", sparse_meta)
            _save("out", out)
        # Full prefill reads a temporary PA_ND cache so attention does not
        # depend on the persistent SWA cache. Match C++ step 8 by writing the
        # projected KV into the persistent cache only after that attention
        # finishes; decode reads this cache on the next forward.
        if use_temporary_prefill_kv and ori_kv is not None and ori_slot is not None:
            _scatter_by_slot(ori_kv, ori_slot, k)
        if os.getenv("XLLM_DSV4_SPARSE_DEBUG", "0") == "1":
            import sys as _sys
            print(f"[SPARSE AFTER L{layer_id}] out={tuple(out.shape)}", file=_sys.stderr, flush=True)
        return out

    def mla_index_context(self, layer: Attention) -> DsaIndexContext:
        """Hand the DSA indexer its paged index cache + block tables + slots."""
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        assert dsa is not None
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        index_slot = _get_layer_cache_tensor(
            dsa.slot_mappings, layer_id, mapping.index_cache_idx
        )
        index_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.index_cache_idx
        )
        cmp_block_table = _get_layer_cache_tensor(
            dsa.block_tables, layer_id, mapping.cmp_cache_idx
        )
        return DsaIndexContext(
            index_cache=layer_cache.index if layer_cache.index is not None else torch.empty(0),
            indexer_scale=layer_cache.indexer_scale,
            slot_mapping=index_slot if index_slot is not None else torch.empty(0),
            block_table=index_block_table,
            cmp_block_table=cmp_block_table,
            kv_state=layer_cache.compress_kv_state,
            score_state=layer_cache.compress_score_state,
            kv_block_table=_get_layer_cache_tensor(
                dsa.block_tables, layer_id, mapping.kv_state_cache_idx
            ),
            score_block_table=_get_layer_cache_tensor(
                dsa.block_tables, layer_id, mapping.score_state_cache_idx
            ),
            actual_seq_q=dsa.actual_seq_lengths_query,
            actual_seq_kv=dsa.actual_seq_lengths_kv,
            start_pos=dsa.start_pos,
            qli_metadata=dsa.qli_metadata,
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
    # first forward so the backend can stage them into DsaMetadata.

    def attach_rope_tables(
        self,
        positions: torch.Tensor,
        dsa_cos_sin: torch.Tensor | None,
        graph_bt_cols: int = 0,
        c4_cos_sin: torch.Tensor | None = None,
        c128_cos_sin: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        metadata = metadata or self._metadata
        if metadata is not None:
            metadata.dsa_positions = positions
            metadata.dsa_cos_sin = dsa_cos_sin
            metadata.dsa_c4_cos_sin = c4_cos_sin
            metadata.dsa_c128_cos_sin = c128_cos_sin
            metadata.dsa_graph_block_table_cols = graph_bt_cols
        self._positions = positions
        self._dsa_cos_sin = dsa_cos_sin
        self._c4_cos_sin = c4_cos_sin
        self._c128_cos_sin = c128_cos_sin
        self._graph_bt_cols = graph_bt_cols

    def attach_compressor(self, fn) -> None:
        """``fn(layer_id, layer_cache, dsa, mapping, cmp_block_table) -> Tensor``."""
        self._compressor_fn = fn

    def attach_indexer(self, fn) -> None:
        """``fn(layer_id, layer_cache, dsa, mapping, q) -> Tensor`` (topk idxs)."""
        self._indexer_fn = fn

    # -- internals ----------------------------------------------------------

    def _layer_compress_ratio(self, layer_id: int) -> int:
        if layer_id < len(self.caches_info):
            caches = self.caches_info[layer_id]
            for ci in caches:
                if ci.cache_type == DSA_CACHE_TOKEN:
                    return ci.ratio
        return 1

    def _resolve_cache_mapping(
        self, layer_id: int, compress_ratio: int
    ) -> _DsaCacheMapping:
        """Python port of ``resolve_cache_mapping`` (deepseek_sparse_attention.cpp:92)."""
        mapping = _DsaCacheMapping()
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
        dsa: DsaMetadata,
        metadata: AttentionMetadata,
        *,
        forward_meta_override: _DsaForwardMeta | None = None,
        cu_seqlens_ori_kv_override: torch.Tensor | None = None,
    ) -> None:
        """Build the AICPU tiling metadata for each compress ratio present.

        Mirrors the C++ ``build_precomputed_metadata`` step: one
        ``sparse_attn_sharedkv_metadata`` per ratio, plus one
        ``quant_lightning_indexer_metadata`` for the qli path.
        """
        from xllm.python import kernels

        dsa.c1_metadata = None
        dsa.c4_metadata = None
        dsa.c128_metadata = None
        dsa.qli_metadata = None
        self._sparse_metadata = {}
        self._qli_metadata = None

        # Build sparse_attn_sharedkv_metadata for all ratios present, matching
        # C++ build_precomputed_metadata (deepseek_v4.h:1577+).
        ratios = sorted(
            {
                ci.ratio
                for layer_caches in self.caches_info
                for ci in layer_caches
                if ci.cache_type == DSA_CACHE_TOKEN
            }
        )
        # Always include ratio=1 (C1 layers use sparse_attn_sharedkv too).
        if 1 not in ratios:
            ratios = [1] + ratios

        # C++ build_precomputed_metadata first moves these fields to the
        # metadata device selected from dsa.input_positions. The diagnostic
        # override lets us verify whether the Python torch.ops path is sensitive
        # to host/device placement.
        spmeta_device_env = os.getenv("XLLM_DSV4_SPMETA_DEVICE", "npu")
        spmeta_device = (
            torch.device("cpu") if spmeta_device_env == "cpu" else self.device
        )
        seq_q = dsa.actual_seq_lengths_query.to(spmeta_device)
        seq_kv = dsa.actual_seq_lengths_kv.to(spmeta_device)
        batch_size = int(max(dsa.actual_seq_lengths_kv.numel(), 1))
        forward_meta = forward_meta_override or _build_dsa_forward_meta(
            dsa, metadata
        )
        max_q = forward_meta.q_max_seq_len
        max_kv = forward_meta.kv_max_seq_len
        is_decode_metadata = max_q <= 1
        if (
            is_decode_metadata
            and os.getenv("XLLM_DSV4_DECODE_SEQ_Q_LENS", "0") == "1"
        ):
            seq_q = dsa.seq_lens_q.to(spmeta_device)
        if (
            is_decode_metadata
            and os.getenv("XLLM_DSV4_DECODE_MAX_KV_FLOOR", "0") == "1"
        ):
            max_kv = max(max_kv, self.index_topk, self.window_size, 1)
        empty_int32 = torch.empty(0, dtype=torch.int32, device=spmeta_device)
        cu_seqlens_ori_kv = (
            cu_seqlens_ori_kv_override
            if cu_seqlens_ori_kv_override is not None
            else (seq_q if not is_decode_metadata else empty_int32)
        )
        cu_seqlens_cmp_kv = empty_int32
        seqused_q = empty_int32
        if (
            is_decode_metadata
            and os.getenv("XLLM_DSV4_DECODE_CU_ORI_SEQ", "0") == "1"
        ):
            cu_seqlens_ori_kv = seq_q
        if (
            is_decode_metadata
            and os.getenv("XLLM_DSV4_DECODE_EMPTY_AS_NONE", "0") == "1"
        ):
            cu_seqlens_ori_kv = None
            cu_seqlens_cmp_kv = None
            seqused_q = None
        seqused_kv = seq_kv
        # sparse_attn_sharedkv_metadata is asynchronous.  Retain every tensor
        # created for its optional inputs on the per-forward metadata owner,
        # matching the lifetime of C++ DSAMetadata.
        dsa.precomputed_metadata_inputs = tuple(
            tensor
            for tensor in (
                seq_q,
                seq_kv,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                seqused_q,
                seqused_kv,
            )
            if tensor is not None
        )
        meta_sync = os.getenv("XLLM_DSV4_META_SYNC", "0") == "1"
        metadata_call = os.getenv("XLLM_DSV4_SPMETA_CALL", "direct")
        if meta_sync:
            import sys as _sys
            view_max_q = getattr(metadata, "max_query_len", "missing")
            view_max_kv = getattr(metadata, "max_seq_len", "missing")
            def _shape_or_none(tensor):
                return None if tensor is None else tuple(tensor.shape)
            print(
                f"[SPMETA PRE] decode={is_decode_metadata} ratios={ratios} "
                f"view_max_q={view_max_q} view_max_kv={view_max_kv} "
                f"seq_q={tuple(seq_q.shape)}/{seq_q.device}/"
                f"{seq_q.detach().cpu().tolist() if seq_q.numel() <= 8 else 'large'} "
                f"seq_kv={tuple(seq_kv.shape)}/{seq_kv.device}/"
                f"{seq_kv.detach().cpu().tolist() if seq_kv.numel() <= 8 else 'large'} "
                f"batch={batch_size} max_q={max_q} max_kv={max_kv} "
                f"win={self.window_size} topk={self.index_topk}",
                file=_sys.stderr,
                flush=True,
            )
        for ratio in ratios:
            has_cmp = ratio > 1
            cmp_topk = self.index_topk if ratio == 4 else 0
            if meta_sync:
                import sys as _sys
                print(
                    f"[SPMETA CALL] ratio={ratio} has_cmp={has_cmp} "
                    f"cmp_topk={cmp_topk} cu_ori={_shape_or_none(cu_seqlens_ori_kv)} "
                    f"cu_cmp={_shape_or_none(cu_seqlens_cmp_kv)} "
                    f"seqused_q={_shape_or_none(seqused_q)} "
                    f"seq_kv={seqused_kv.detach().cpu().tolist() if seqused_kv.numel() <= 8 else 'large'} "
                    f"max_kv={max_kv}",
                    file=_sys.stderr,
                    flush=True,
                )
            metadata_args = (
                self.num_heads,
                1,
                self.metadata_head_dim,
                seq_q,
                cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv,
                seqused_q,
                seqused_kv,
                batch_size,
                max_q,
                max_kv,
                0,
                cmp_topk,
                ratio,
                _MASK_MODE_COMPRESS,
                _MASK_MODE_RIGHT_DOWN_CAUSAL,
                self.window_size - 1,
                0,
                "TND",
                "PA_ND",
                True,
                has_cmp,
            )
            if metadata_call == "direct":
                import xllm_runtime

                sparse_metadata = (
                    xllm_runtime.dsv4_sparse_attn_sharedkv_metadata(
                        *metadata_args
                    )
                )
            elif metadata_call == "torch_ops":
                sparse_metadata = kernels.sparse_attn_sharedkv_metadata(
                    *metadata_args
                )
            else:
                raise ValueError(
                    "XLLM_DSV4_SPMETA_CALL must be 'direct' or 'torch_ops', "
                    f"got {metadata_call!r}"
                )
            if ratio == 1:
                dsa.c1_metadata = sparse_metadata
            elif ratio == 4:
                dsa.c4_metadata = sparse_metadata
            elif ratio == 128:
                dsa.c128_metadata = sparse_metadata
            self._sparse_metadata[ratio] = sparse_metadata
            if meta_sync:
                import sys as _sys
                try:
                    import torch_npu as _tn
                    _tn.npu.synchronize()
                    print(
                        f"[SPMETA OK] ratio={ratio} meta={tuple(sparse_metadata.shape)} "
                        f"{sparse_metadata.device}",
                        file=_sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    print(f"[SPMETA FAIL] ratio={ratio} exc={exc}", file=_sys.stderr, flush=True)
                    raise
        if 4 in ratios:
            # Faithful port of deepseek_v4.h:1695-1715 QuantLightningIndexerMetadataParams.
            # query_lens = actual_seq_lengths_query[1:] (drop leading 0, [B+1]->[B]);
            # key_lens = seq_lens (actual_seq_lengths_kv, [B]); num_heads_k = 1 (NOT
            # index_n_heads); max_seqlen_q/k are dynamic from host seqs, not stored max.
            query_lens = (
                seq_q[1:].clone()
                if seq_q.numel() > 1
                else dsa.seq_lens_q.to(self.device)
            )
            key_lens = dsa.seq_lens.to(self.device) if dsa.seq_lens.numel() > 0 else seq_kv
            dsa.precomputed_metadata_inputs += (query_lens, key_lens)
            global_index_num_heads = max(self.index_n_heads, 1)
            qli_metadata = kernels.quant_lightning_indexer_metadata(
                num_heads_q=global_index_num_heads,
                num_heads_k=1,
                head_dim=max(self.index_head_dim, 1),
                query_quant_mode=0,
                key_quant_mode=0,
                actual_seq_lengths_query=query_lens,
                actual_seq_lengths_key=key_lens,
                batch_size=int(max(key_lens.size(0), 1)),
                max_seqlen_q=max(forward_meta.q_max_seq_len, 1),
                max_seqlen_k=max(forward_meta.kv_max_seq_len, 1),
                layout_query="TND",
                layout_key="PA_BSND",
                sparse_count=self.index_topk,
                sparse_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
                pre_tokens=2**63 - 1,
                next_tokens=2**63 - 1,
                cmp_ratio=4,
                device=str(self.device),
            )
            dsa.qli_metadata = qli_metadata
            self._qli_metadata = qli_metadata
            if meta_sync:
                import sys as _sys

                try:
                    import torch_npu as _tn

                    _tn.npu.synchronize()
                    print(
                        f"[QLIMETA OK] query_lens="
                        f"{query_lens.detach().cpu().tolist()} key_lens="
                        f"{key_lens.detach().cpu().tolist()} "
                        f"max_q={forward_meta.q_max_seq_len} "
                        f"max_k={forward_meta.kv_max_seq_len} "
                        f"meta={tuple(qli_metadata.shape)} {qli_metadata.device}",
                        file=_sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[QLIMETA FAIL] query_lens="
                        f"{query_lens.detach().cpu().tolist()} key_lens="
                        f"{key_lens.detach().cpu().tolist()} exc={exc}",
                        file=_sys.stderr,
                        flush=True,
                    )
                    raise

    def _pack_metadata_to_device(self, dsa: DsaMetadata) -> None:
        """Mirror C++ deepseek_v4_pack_dsa_metadata_to_device()."""
        import xllm_runtime

        specs: list[torch.Tensor] = []
        assignments: list[tuple[int, object, str | int]] = []
        spec_by_storage: dict[tuple[int, tuple[int, ...], torch.dtype], int] = {}

        def add(tensor: torch.Tensor | None, owner: object, key: str | int) -> None:
            if tensor is None or tensor.numel() == 0 or not tensor.device.type == "cpu":
                return
            contiguous = tensor.contiguous()
            storage_key = (
                contiguous.data_ptr(),
                tuple(contiguous.shape),
                contiguous.dtype,
            )
            spec_idx = spec_by_storage.get(storage_key)
            if spec_idx is None:
                spec_idx = len(specs)
                spec_by_storage[storage_key] = spec_idx
                specs.append(contiguous)
            assignments.append((spec_idx, owner, key))

        for name in (
            "seq_lens",
            "seq_lens_q",
            "actual_seq_lengths_kv",
            "actual_seq_lengths_query",
            "kv_cu_seq_lens",
            "max_seqlen_kv",
            "max_seqlen_q",
            "input_positions",
            "c4_pad_positions",
            "c128_pad_positions",
            "start_pos",
            "hadamard",
        ):
            add(getattr(dsa, name), dsa, name)
        for layer_tensors in dsa.block_tables:
            for index, tensor in enumerate(layer_tensors):
                add(tensor, layer_tensors, index)
        for layer_tensors in dsa.slot_mappings:
            for index, tensor in enumerate(layer_tensors):
                add(tensor, layer_tensors, index)

        if not specs:
            return
        device_reference = dsa.input_positions
        if device_reference is None or device_reference.device.type == "cpu":
            device_reference = torch.empty(0, device=self.device)
        packed_buffer, views = xllm_runtime.dsv4_pack_metadata_tensors(
            specs, device_reference
        )
        for spec_idx, owner, key in assignments:
            if isinstance(key, str):
                setattr(owner, key, views[spec_idx])
            else:
                owner[key] = views[spec_idx]
        dsa.packed_metadata_buffer = packed_buffer


# ---------------------------------------------------------------------------
# Helpers (faithful ports of C++ free functions).
# ---------------------------------------------------------------------------


def _tensor_max_or_zero(tensor: torch.Tensor | None) -> int:
    if tensor is None or tensor.numel() == 0:
        return 0
    return int(tensor.max().item())


def _build_dsa_forward_meta(
    dsa: DsaMetadata, metadata: AttentionMetadata
) -> _DsaForwardMeta:
    """Mirror the C++ max-seqlen inputs used by build_precomputed_metadata.

    C++ computes sparse metadata max sizes from ModelInputParams::meta plus the
    host q/kv length vectors:
      max(params.meta.q_max_seq_len, max(host.q_seq_lens))
      max(params.meta.kv_max_seq_len, max(host.kv_seq_lens))
    """

    q_max = int(getattr(metadata, "max_query_len", dsa.max_query_len))
    kv_max = int(getattr(metadata, "max_seq_len", dsa.max_seq_len))
    q_max = max(q_max, _tensor_max_or_zero(getattr(metadata, "q_seq_lens_host", None)))
    kv_max = max(kv_max, _tensor_max_or_zero(getattr(metadata, "kv_seq_lens_host", None)))
    q_max = max(q_max, int(dsa.max_query_len))
    kv_max = max(kv_max, int(dsa.max_seq_len))
    return _DsaForwardMeta(q_max_seq_len=q_max, kv_max_seq_len=kv_max)


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
        q_len = (
            dst_cu[i + 1] - dst_cu[i]
            if dst_cu is not None
            else cu[i + 1] - cu[i]
        )
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
        total_blocks + 1, block_size, kv.size(1), kv.size(2),
        dtype=kv.dtype, device=kv.device,
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
            target = packed_kv[next_block:next_block + blocks].view(
                blocks * block_size, kv.size(1), kv.size(2)
            )
            target[q_len - copy_len:q_len].copy_(kv[q_start:q_start + copy_len])
        next_block += blocks

    table = torch.tensor(table_data, dtype=torch.int32, device=kv.device).view(
        batch_size, table_cols
    )
    return packed_kv, table


def _get_layer_cache_tensor(
    layer_tensors: list[list[torch.Tensor]],
    layer_id: int,
    cache_idx: int,
) -> torch.Tensor | None:
    """Python port of ``get_layer_cache_tensor`` (deepseek_sparse_attention.cpp:80)."""
    if (
        layer_id < 0
        or layer_id >= len(layer_tensors)
        or cache_idx < 0
        or cache_idx >= len(layer_tensors[layer_id])
    ):
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
