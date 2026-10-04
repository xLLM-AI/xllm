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
from functools import cache
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
    from xllm.python.model_executor.v4_cp_context import DeepseekV4CpContext

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


class DsaAttentionBackend(AttentionBackend):
    """Unified backend for DeepSeek-V4 sliding, CSA, and HCA layers on NPU.

    The same backend dispatches C1, C4, and C128 layers, selected from each
    layer's compression ratio.
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

    @property
    def uses_executor_cp_context(self) -> bool:
        return False

    def create_graph_block_tables(
        self,
        batch_size: int,
        max_model_len: int,
        max_block_columns: int,
    ) -> tuple[torch.Tensor, ...]:
        tables: list[torch.Tensor] = []
        for group in self.group_infos:
            columns = max_block_columns
            if group.cache_type == DSA_CACHE_TOKEN:
                tokens_per_block = group.ratio * group.block_size
                columns = max((max_model_len + tokens_per_block - 1) // tokens_per_block, 1)
            tables.append(torch.zeros((batch_size, columns), dtype=torch.int32, device=self.device))
        return tuple(tables)

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
        if graph_mode:
            metadata.dsa_graph_mode = True

    def prepare_graph_replay(self, metadata: AttentionMetadata) -> None:
        self.prepare(metadata, graph_mode=True)
        self.refresh_dsa_metadata_for_graph_replay(metadata)

    def reset_forward(self, metadata: AttentionMetadata | None = None) -> None:
        """Drop request-owned DSA state before attaching the next request.

        The model calls this immediately before attaching its current RoPE
        tables.  Keeping reset separate from :meth:`prepare` preserves the
        required attach-then-build ordering for compressed positions.
        """
        metadata = self._metadata if metadata is None else metadata
        if metadata is None:
            return
        metadata.dsa_metadata = None
        metadata.dsa_positions = None
        metadata.dsa_cos_sin = None
        metadata.dsa_c4_cos_sin = None
        metadata.dsa_c128_cos_sin = None
        metadata.dsa_graph_mode = False
        for name in (
            "_compressor_fn",
            "_indexer_fn",
            "_current_hidden",
            "_current_kv_hidden",
            "_current_qr",
            "_current_qr_pertoken_scale",
        ):
            if hasattr(self, name):
                delattr(self, name)

    def prepare_dsa_metadata_for_forward(
        self,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build compressed-attention metadata in the current model forward."""
        metadata = self._metadata if metadata is None else metadata
        assert metadata is not None
        compressed_metadata = self._build_dsa_metadata_for_forward(metadata)
        metadata.dsa_metadata = compressed_metadata

    def _build_dsa_metadata_for_forward(
        self,
        metadata: AttentionMetadata,
    ) -> DsaMetadata:
        """Build one request's complete DSA metadata without publishing it."""
        # MTP draft layers consume only a prefix of the target cache managers.
        multi_block_tables = metadata.multi_block_tables[: len(self.group_infos)]
        max_query_len = int(getattr(metadata, "max_query_len", 0))
        max_seq_len = int(getattr(metadata, "max_seq_len", 0))
        kv_seq_lens_host = getattr(metadata, "kv_seq_lens_host", None)
        if kv_seq_lens_host is not None and kv_seq_lens_host.numel() > 0:
            kv_seq_lens = kv_seq_lens_host.cpu().tolist()
            max_seq_len = max(max_seq_len, max(kv_seq_lens))
        else:
            kv_seq_lens = list(getattr(metadata, "kv_seq_lens_host_values", None) or [])
        q_seq_lens_host = getattr(metadata, "q_seq_lens_host", None)
        if q_seq_lens_host is not None and q_seq_lens_host.numel() > 0:
            q_seq_lens = q_seq_lens_host.cpu().tolist()
            max_query_len = max(max_query_len, max(q_seq_lens))
        else:
            q_seq_lens_tensor = getattr(metadata, "q_seq_lens", None)
            q_seq_lens = (
                q_seq_lens_tensor.cpu().tolist()
                if q_seq_lens_tensor is not None and q_seq_lens_tensor.numel() > 0
                else None
            )
        new_cache_slots = getattr(metadata, "new_cache_slots_host_values", None)
        # The dsa_* fields are legacy C++/pybind contract names. Their tensors
        # are model-owned and scoped to the current forward.
        positions = getattr(metadata, "dsa_positions", None)
        if positions is None:
            positions = torch.empty(0, dtype=torch.int64)
        enable_graph = bool(getattr(metadata, "dsa_graph_mode", False))
        graph_capacity_cols = int(getattr(metadata, "dsa_graph_block_table_cols", 0))
        is_dummy = bool(getattr(metadata, "is_dummy", False))
        if is_dummy:
            row_count = max(
                len(kv_seq_lens),
                len(q_seq_lens or ()),
                int(positions.numel()),
                1,
            )
            dummy_kv_len = max(self.index_topk, self.window_size, 1)
            kv_seq_lens = [dummy_kv_len] * row_count
            q_seq_lens = [1] * row_count
            host_device = kv_seq_lens_host.device if kv_seq_lens_host is not None else torch.device("cpu")
            multi_block_tables = self._build_empty_dp_block_tables(
                multi_block_tables,
                row_count,
                dummy_kv_len,
                host_device,
                graph_mode=enable_graph,
                graph_block_table_capacity_cols=graph_capacity_cols,
            )
        compressed_metadata = self._builder.build(
            multi_block_tables=multi_block_tables,
            kv_seq_lens=kv_seq_lens,
            q_seq_lens=q_seq_lens,
            positions=positions,
            is_prefill=metadata.is_prefill and not is_dummy,
            is_chunked_prefill=metadata.is_chunked_prefill,
            new_cache_slots=new_cache_slots,
            enable_graph=enable_graph,
            graph_block_table_capacity_cols=graph_capacity_cols,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
        )
        # Synthetic warmup metadata may omit rotary caches.
        if getattr(metadata, "dsa_cos_sin", None) is not None and metadata.dsa_cos_sin.numel() > 0:
            compressed_metadata.input_rope_by_ratio = self._build_dsa_rope_metadata(compressed_metadata, metadata)
        else:
            self._populate_compressed_attention_rope(compressed_metadata, metadata)
        self._move_metadata_to_device(compressed_metadata)
        self._build_precomputed_metadata(compressed_metadata)
        return compressed_metadata

    def _build_empty_dp_block_tables(
        self,
        block_tables: list[torch.Tensor],
        batch_size: int,
        dummy_kv_len: int,
        fallback_device: torch.device,
        *,
        graph_mode: bool,
        graph_block_table_capacity_cols: int,
    ) -> list[torch.Tensor]:
        """Use reserved block 0 while preserving each cache manager's graph capacity."""
        normalized: list[torch.Tensor] = []
        for group_id, group_info in enumerate(self.group_infos):
            if group_info.cache_type == DSA_CACHE_TOKEN:
                cache_slot_count = dummy_kv_len // group_info.ratio
            elif group_info.cache_type == DSA_CACHE_SLIDING_WINDOW:
                cache_slot_count = group_info.block_size
            else:
                cache_slot_count = dummy_kv_len
            required_columns = max(
                (cache_slot_count + group_info.block_size - 1) // group_info.block_size,
                1,
            )

            device = fallback_device
            captured_columns = 0
            if graph_mode and group_id < len(block_tables):
                table = block_tables[group_id]
                device = table.device
                captured_columns = int(table.size(1))
            if group_info.cache_type == DSA_CACHE_SLIDING_WINDOW:
                required_storage_columns = max(
                    (dummy_kv_len + group_info.block_size - 1) // group_info.block_size,
                    1,
                )
                if graph_mode and graph_block_table_capacity_cols < required_storage_columns:
                    raise ValueError(
                        "ACL graph SWA block table capacity is too small for "
                        f"empty-DP history: requires {required_storage_columns} "
                        f"columns, capacity is {graph_block_table_capacity_cols}"
                    )
                block_count = required_columns
            else:
                block_count = max(required_columns, captured_columns)
            normalized.append(
                torch.zeros(
                    (batch_size, block_count),
                    dtype=torch.int32,
                    device=device,
                )
            )
        return normalized

    def refresh_dsa_metadata_for_graph_replay(
        self,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Refresh dynamic DSA values while preserving graph-captured storage."""
        metadata = self._metadata if metadata is None else metadata
        if metadata is None or metadata.dsa_metadata is None:
            raise RuntimeError("ACL graph DSA metadata must be captured before replay")
        persistent = metadata.dsa_metadata
        refreshed = self._build_dsa_metadata_for_forward(metadata)
        self._copy_graph_dsa_metadata(persistent, refreshed)

    @classmethod
    def _copy_graph_dsa_metadata(
        cls,
        persistent: DsaMetadata,
        refreshed: DsaMetadata,
    ) -> None:
        """Copy replay values into the tensors retained by ACL graph capture."""
        tensor_fields = (
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
            "c4_cos",
            "c4_sin",
            "c128_cos",
            "c128_sin",
            "c1_metadata",
            "c4_metadata",
            "c128_metadata",
            "qli_metadata",
        )
        copied_fields: dict[int, int] = {}
        for field_name in tensor_fields:
            target, source = getattr(persistent, field_name, None), getattr(refreshed, field_name, None)
            target_id, source_id = id(target), id(source)
            if copied_fields.get(target_id) != source_id:
                cls._copy_graph_tensor(target, source, field_name)
                copied_fields[target_id] = source_id

        if len(persistent.block_tables) != len(refreshed.block_tables):
            raise RuntimeError("ACL graph DSA block-table layer count changed")
        if len(persistent.slot_mappings) != len(refreshed.slot_mappings):
            raise RuntimeError("ACL graph DSA slot-mapping layer count changed")
        copied: dict[int, int] = {}
        for field_name in ("block_tables", "slot_mappings"):
            for layer_id, (persistent_layer, refreshed_layer) in enumerate(
                zip(getattr(persistent, field_name), getattr(refreshed, field_name), strict=True)
            ):
                name = f"{field_name}[{layer_id}]"
                if len(persistent_layer) != len(refreshed_layer):
                    raise RuntimeError(f"ACL graph DSA {name} count changed")
                for index, (target, source) in enumerate(zip(persistent_layer, refreshed_layer, strict=True)):
                    target_id, source_id = id(target), id(source)
                    if copied.get(target_id) != source_id:
                        cls._copy_graph_tensor(target, source, f"{name}[{index}]", pad_value=-1)
                        copied[target_id] = source_id

        if persistent.input_rope_by_ratio.keys() != refreshed.input_rope_by_ratio.keys():
            raise RuntimeError("ACL graph DSA RoPE ratio set changed")
        for ratio, persistent_pair in persistent.input_rope_by_ratio.items():
            refreshed_pair = refreshed.input_rope_by_ratio[ratio]
            cls._copy_graph_tensor(persistent_pair[0], refreshed_pair[0], f"input_rope[{ratio}].cos")
            cls._copy_graph_tensor(persistent_pair[1], refreshed_pair[1], f"input_rope[{ratio}].sin")

        persistent.max_query_len = refreshed.max_query_len
        persistent.max_seq_len = refreshed.max_seq_len
        persistent.is_acl_graph = refreshed.is_acl_graph
        persistent.precomputed_metadata_inputs = refreshed.precomputed_metadata_inputs

    @staticmethod
    def _copy_graph_tensor(
        persistent: torch.Tensor | None,
        refreshed: torch.Tensor | None,
        name: str,
        *,
        pad_value: int = 0,
    ) -> None:
        if persistent is None and refreshed is None:
            return
        if persistent is None or refreshed is None:
            raise RuntimeError(f"ACL graph DSA tensor availability changed for {name}")
        if persistent.dtype != refreshed.dtype or persistent.device != refreshed.device:
            raise RuntimeError(f"ACL graph DSA tensor type changed for {name}")
        if persistent.shape == refreshed.shape:
            if persistent.data_ptr() != refreshed.data_ptr():
                persistent.copy_(refreshed, non_blocking=True)
            return
        prefix_compatible = (
            persistent.dim() == refreshed.dim()
            and refreshed.dim() > 0
            and refreshed.shape[0] <= persistent.shape[0]
            and persistent.shape[1:] == refreshed.shape[1:]
        )
        if not prefix_compatible:
            raise RuntimeError(
                f"ACL graph DSA tensor shape changed for {name}: {tuple(persistent.shape)} != {tuple(refreshed.shape)}"
            )
        persistent.fill_(pad_value)
        persistent[: refreshed.shape[0]].copy_(refreshed, non_blocking=True)

    def localize_dsa_metadata_for_cp(
        self,
        cp_context: DeepseekV4CpContext,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Localize query metadata while retaining global KV write positions and paging."""
        metadata = self._metadata if metadata is None else metadata
        if metadata is None or metadata.dsa_metadata is None:
            raise RuntimeError("compressed-attention metadata must be prepared before CP localization")
        dsa = metadata.dsa_metadata

        dsa.v4_cp_context = cp_context

        local_q = cp_context.local_q_seq_lens
        local_kv = cp_context.local_kv_seq_lens
        device = dsa.seq_lens_q.device if dsa.seq_lens_q.numel() > 0 else self.device

        q_lens = torch.tensor(local_q, dtype=torch.int32, device=device)
        dsa.seq_lens_q = q_lens
        actual_q = torch.cat(
            (
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.cumsum(q_lens, 0, dtype=torch.int32),
            )
        )
        dsa.actual_seq_lengths_query = actual_q
        local_q_max = int(max(local_q, default=0))
        dsa.max_query_len = local_q_max
        dsa.max_seqlen_q = torch.tensor([local_q_max], dtype=torch.int32, device=device)

        kv_lens = torch.tensor(local_kv, dtype=torch.int32, device=device)
        dsa.actual_seq_lengths_kv = kv_lens
        dsa.seq_lens = kv_lens
        dsa.kv_cu_seq_lens = cp_context.local_kv_cu_seq_lens.to(device)
        local_kv_max = int(max(local_kv, default=0))
        dsa.max_seq_len = local_kv_max
        dsa.max_seqlen_kv = torch.tensor([local_kv_max], dtype=torch.int32, device=device)
        # Localize query positions while preserving global KV paging and write positions.
        if cp_context.local_positions is not None and cp_context.local_positions.numel() > 0:
            dsa.input_positions = cp_context.local_positions.to(device)

        # The CP context retains global RoPE pairs before this map is rebuilt for local queries.
        dsa.input_rope_by_ratio = self._build_dsa_rope_metadata(dsa, metadata)

        self._build_precomputed_metadata(dsa)

    def prepare_csa_metadata_for_forward(
        self,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Backward-compatible alias for the canonical DSA API."""
        self.prepare_dsa_metadata_for_forward(metadata)

    def select_dsa_layer_rope(
        self,
        layer_id: int,
        cos_sin_cache: torch.Tensor,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Select request-shaped layer RoPE; clear missing ratios to avoid stale tables."""
        metadata = self._metadata if metadata is None else metadata
        if metadata is None or metadata.dsa_metadata is None:
            raise RuntimeError("compressed-attention metadata must be prepared before selecting layer RoPE")
        compressed_metadata = metadata.dsa_metadata
        compress_ratio = self._layer_compress_ratio(layer_id)
        rope_by_ratio = compressed_metadata.input_rope_by_ratio
        pair = rope_by_ratio.get(compress_ratio, rope_by_ratio.get(1, (None, None)))
        if pair[0] is None or pair[1] is None:
            cos, sin = self._slice_cos_sin_cache(cos_sin_cache)
            pair = (cos, sin)
        compressed_metadata.layer_id = layer_id
        compressed_metadata.cos_table = pair[0].contiguous()
        compressed_metadata.sin_table = pair[1].contiguous()

    def select_compressed_attention_layer_rope(
        self,
        layer_id: int,
        cos_sin_cache: torch.Tensor,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Backward-compatible alias for the canonical DSA API."""
        self.select_dsa_layer_rope(layer_id, cos_sin_cache, metadata)

    def _populate_compressed_attention_rope(
        self,
        compressed_metadata: DsaMetadata,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        """Build request-shaped RoPE tensors for the current forward."""
        metadata = self._metadata if metadata is None else metadata
        css = getattr(metadata, "dsa_cos_sin", None)
        if compressed_metadata.cos_table is None and css is not None and css.numel() > 0:
            compressed_metadata.cos_table, compressed_metadata.sin_table = self._slice_cos_sin_cache(css)
        for ratio in (4, 128):
            cache = getattr(metadata, f"dsa_c{ratio}_cos_sin", None)
            positions = getattr(compressed_metadata, f"c{ratio}_pad_positions")
            if cache is not None and positions.numel() > 0:
                indices = positions.clamp_min(0).to(device=cache.device, dtype=torch.int64)
                cos, sin = self._slice_cos_sin_cache(cache.index_select(0, indices))
                setattr(compressed_metadata, f"c{ratio}_cos", cos)
                setattr(compressed_metadata, f"c{ratio}_sin", sin)

    @staticmethod
    def _slice_cos_sin_cache(
        cos_sin_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return half-width cos/sin tables for the requested rows."""
        if cos_sin_cache is None or cos_sin_cache.numel() == 0:
            empty = torch.empty(0)
            return empty, empty
        chunks = cos_sin_cache.chunk(2, dim=-1)
        return chunks[0].contiguous(), chunks[1].contiguous()

    def _build_dsa_rope_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: AttentionMetadata | None = None,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Build half-width RoPE tables for request rows and compressed cache rows."""
        metadata = self._metadata if metadata is None else metadata
        positions = compressed_metadata.input_positions

        def validate_cache(name: str, cache: torch.Tensor | None) -> torch.Tensor:
            if cache is None or cache.numel() == 0 or cache.dim() != 2:
                raise RuntimeError(f"DeepSeek-V4 {name} RoPE cache must be a 2-D cache")
            return cache

        default_cache = validate_cache("default", getattr(metadata, "dsa_cos_sin", None))
        c4_cache = validate_cache("c4", getattr(metadata, "dsa_c4_cos_sin", None))
        c128_cache = validate_cache("c128", getattr(metadata, "dsa_c128_cos_sin", None))

        def build_pair(cache: torch.Tensor, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            if pos is None or pos.numel() == 0:
                return torch.empty(0), torch.empty(0)
            indexed = cache.index_select(
                0,
                pos.reshape(-1).to(device=cache.device, dtype=torch.int64),
            )
            half = indexed.size(-1) // 2
            cos = indexed[..., :half].contiguous()
            sin = indexed[..., half:].contiguous()
            return cos, sin

        rope_by_ratio = {1: build_pair(default_cache, positions)}
        compressed_metadata.cos_table = compressed_metadata.sin_table = torch.empty(0)
        compressed_metadata.c4_cos, compressed_metadata.c4_sin = build_pair(
            c4_cache, compressed_metadata.c4_pad_positions
        )
        compressed_metadata.c128_cos, compressed_metadata.c128_sin = build_pair(
            c128_cache, compressed_metadata.c128_pad_positions
        )

        if positions is not None and positions.numel() > 0:
            rope_by_ratio[4] = build_pair(c4_cache, positions)

            rope_by_ratio[128] = build_pair(c128_cache, positions)

        return rope_by_ratio

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
        if compressed_metadata.input_positions.numel() == 0:
            pos = getattr(metadata, "dsa_positions", None)
            if pos is not None and pos.numel() > 0:
                compressed_metadata.input_positions = pos
        if (
            compressed_metadata.cos_table is None
            or compressed_metadata.sin_table is None
            or compressed_metadata.c4_cos is None
            or compressed_metadata.c128_cos is None
        ):
            self._populate_compressed_attention_rope(compressed_metadata, metadata)
        layer_id = layer.layer_id
        compress_ratio = self._layer_compress_ratio(layer_id)
        attention_type = _attention_type_for_compress_ratio(compress_ratio)
        mapping = self._resolve_cache_mapping(layer_id, compress_ratio)
        layer_cache = self._kv_caches[layer_id]
        is_dummy = bool(getattr(metadata, "is_dummy", False))
        is_prefill = metadata.is_prefill and not is_dummy
        is_chunked_prefill = metadata.is_chunked_prefill
        use_temporary_prefill_kv = is_prefill and not is_chunked_prefill
        # 1) Prepare ori_kv for attention (mirrors C++ :790-816).
        # Full prefill: use kv directly as temporary PA_ND (don't scatter to paged).
        # Decode/chunked: scatter to paged SWA cache.
        ori_kv = layer_cache.swa
        ori_slot = _get_layer_cache_tensor(compressed_metadata.slot_mappings, layer_id, mapping.ori_cache_idx)
        ori_block_table = _get_layer_cache_tensor(compressed_metadata.block_tables, layer_id, mapping.ori_cache_idx)
        cp_ctx = getattr(compressed_metadata, "v4_cp_context", None)
        cp_enabled = cp_ctx is not None and cp_ctx.enabled()
        if use_temporary_prefill_kv:
            # Prefill: build temporary PA_ND cache from kv (mirrors C++
            # build_prefill_pa_nd_kv, deepseek_sparse_attention.cpp:272-368).
            ori_kv_for_attn, ori_block_table_for_attn = _build_prefill_pa_nd_kv(
                k,
                cp_ctx.global_q_cu_seq_lens if cp_enabled else compressed_metadata.actual_seq_lengths_query,
                ori_block_table,
                self.window_size,
                cp_ctx.local_kv_cu_seq_lens if cp_enabled else None,
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
                compress_ratio,
            )
            _scatter_by_slot(cmp_kv, cmp_slot, compressed)

        # 3) Indexer: select top-k compressed blocks when ratio == 4.
        compress_topk_idxs: torch.Tensor | None = None
        if compress_ratio == 4 and cmp_kv is not None:
            indexer_fn = getattr(self, "_indexer_fn", None)
            if indexer_fn is None:
                raise RuntimeError("CSA indexer is required for compressed_sparse_attention")
            compress_topk_idxs = indexer_fn(layer_id, layer_cache, compressed_metadata, mapping)
            if compress_topk_idxs is None:
                raise RuntimeError("CSA indexer returned no top-k indices")

        # 4) Two-stage sparse attention over original + compressed KV.
        sparse_meta = getattr(compressed_metadata, f"c{compress_ratio}_metadata", None)
        if sparse_meta is None:
            raise RuntimeError(f"sparse metadata is missing for {attention_type}")
        seq_q = compressed_metadata.actual_seq_lengths_query
        seq_kv = compressed_metadata.actual_seq_lengths_kv
        # Match C++ DSAttention's optional contract exactly: prefill and
        # chunked prefill pass query cu-seqlens, while decode leaves
        # cu_seqlens_ori_kv as std::nullopt. A defined empty tensor selects a
        # different ACL optional-input path and causes small decode drift.
        use_prefill_attn = is_prefill or is_chunked_prefill
        if use_prefill_attn:
            # C++ uses the local KV window cumsum under prefill CP, and the
            # query cumsum otherwise. This must match the value baked into the
            # precomputed sparse metadata by build_precomputed_metadata.
            cu_seqlens_ori_kv_for_attn = cp_ctx.local_kv_cu_seq_lens if cp_enabled else seq_q
        else:
            cu_seqlens_ori_kv_for_attn = None
        sinks = getattr(layer, "attn_sink", None) if getattr(layer, "attn_sink_loaded", False) else None
        if sinks is not None:
            sinks = sinks.to(q.device, dtype=torch.float32).contiguous()
        out, _lse = _sparse_attn_sharedkv(
            q=q,
            ori_kv=ori_kv_for_attn,
            cmp_kv=cmp_kv if compress_ratio > 1 else None,
            ori_sparse_indices=None,
            cmp_sparse_indices=compress_topk_idxs,
            ori_block_table=ori_block_table_for_attn,
            cmp_block_table=cmp_block_table if compress_ratio > 1 else None,
            cu_seqlens_q=seq_q,
            cu_seqlens_ori_kv=cu_seqlens_ori_kv_for_attn,
            # C++ passes nullopt for compressed KV cu-seqlens; cmp_kv is PA_ND
            # and addressed through cmp_block_table/topk.
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seq_kv,
            # sinks: the attention sink parameter (attn_sink) is required by the
            # sparse_attn_sharedkv kernel (C++ :949 passes attn_sink_ when loaded).
            sinks=sinks,
            metadata=sparse_meta,
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
        csa_cos_sin: torch.Tensor | None = None,
        hca_cos_sin: torch.Tensor | None = None,
        metadata: AttentionMetadata | None = None,
    ) -> None:
        metadata = self._metadata if metadata is None else metadata
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

    def _build_precomputed_metadata(self, compressed_metadata: DsaMetadata) -> None:
        """Build the AICPU tiling metadata for each compress ratio present.

        Mirrors the C++ ``build_precomputed_metadata`` step: one
        ``sparse_attn_sharedkv_metadata`` per ratio, plus one
        ``quant_lightning_indexer_metadata`` for the qli path.
        """
        from xllm.python import kernels

        seq_q = compressed_metadata.actual_seq_lengths_query
        seq_kv = compressed_metadata.actual_seq_lengths_kv
        batch_size = max(seq_kv.numel(), 1)
        max_q = compressed_metadata.max_query_len
        max_kv = compressed_metadata.max_seq_len
        is_prefill = max_q > 1
        empty_int32 = torch.empty(0, dtype=torch.int32, device=self.device)
        cu_seqlens_ori_kv = empty_int32
        if is_prefill:
            cu_seqlens_ori_kv = (
                compressed_metadata.kv_cu_seq_lens
                if getattr(compressed_metadata, "v4_cp_context", None) is not None
                else seq_q
            )
        # Metadata kernels enqueue asynchronously. Retain their tensor inputs
        # on the current forward's DsaMetadata, as C++ DSAMetadata does.
        compressed_metadata.precomputed_metadata_inputs = (
            seq_q,
            seq_kv,
            cu_seqlens_ori_kv,
            empty_int32,
        )
        for ratio in (1, 4, 128):
            sparse_metadata = kernels.sparse_attn_sharedkv_metadata(
                num_heads_q=self.num_heads,
                num_heads_kv=1,
                head_dim=self.head_dim,
                cu_seqlens_q=seq_q,
                cu_seqlens_ori_kv=cu_seqlens_ori_kv,
                cu_seqlens_cmp_kv=empty_int32,
                seqused_q=empty_int32,
                seqused_kv=seq_kv,
                batch_size=batch_size,
                max_seqlen_q=max_q,
                max_seqlen_kv=max_kv,
                ori_topk=0,
                cmp_topk=self.index_topk if ratio == 4 else 0,
                cmp_ratio=ratio,
                ori_mask_mode=_MASK_MODE_COMPRESS,
                cmp_mask_mode=_MASK_MODE_RIGHT_DOWN_CAUSAL,
                ori_win_left=max(self.window_size - 1, 0),
                ori_win_right=0,
                layout_q="TND",
                layout_kv="PA_ND",
                has_ori_kv=True,
                has_cmp_kv=ratio > 1,
            )
            setattr(compressed_metadata, f"c{ratio}_metadata", sparse_metadata)
        query_lens = seq_q[1:] if seq_q.numel() > 1 else compressed_metadata.seq_lens_q
        key_lens = compressed_metadata.seq_lens if compressed_metadata.seq_lens.numel() else seq_kv
        compressed_metadata.precomputed_metadata_inputs += (query_lens, key_lens)
        compressed_metadata.qli_metadata = kernels.quant_lightning_indexer_metadata(
            num_heads_q=max(self.index_n_heads, 1),
            num_heads_k=1,
            head_dim=max(self.index_head_dim, 1),
            actual_seq_lengths_query=query_lens,
            actual_seq_lengths_key=key_lens,
            max_seqlen_q=max(max_q, 1),
            max_seqlen_k=max(max_kv, 1),
            sparse_count=self.index_topk,
            cmp_ratio=4,
        )

    def _move_metadata_to_device(self, compressed_metadata: DsaMetadata) -> None:
        """Mirror ``deepseek_v4_move_dsa_metadata_to_device`` for eager mode."""

        @cache
        def move(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.to(self.device)

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
            "c4_cos",
            "c4_sin",
            "c128_cos",
            "c128_sin",
        )
        for name in tensor_fields:
            tensor = getattr(compressed_metadata, name, None)
            if tensor is not None:
                setattr(compressed_metadata, name, move(tensor))
        for field_name in ("block_tables", "slot_mappings"):
            for layer_tensors in getattr(compressed_metadata, field_name):
                for index, tensor in enumerate(layer_tensors):
                    layer_tensors[index] = move(tensor)


# Keep the historical import stable.  There is one implementation and one
# runtime type for C1/C4/C128; CSA is only the name of the ratio-4 path.
CsaAttentionBackend = DsaAttentionBackend


# ---------------------------------------------------------------------------
# Helpers (faithful ports of C++ free functions).
# ---------------------------------------------------------------------------


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
        updates = value_2d[:update_rows].to(cache.dtype)
        if get_forward_context().acl_graph is not None:
            # Graph padding maps to reserved block 0. Avoid IndexSelect in the
            # captured path; it is not reliably capturable on NPU.
            safe_values = updates
        else:
            valid_mask = slots_slice.ge(0).unsqueeze(1)
            old_values = cache_2d.index_select(0, safe_slots)
            safe_values = torch.where(valid_mask, updates, old_values)
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
