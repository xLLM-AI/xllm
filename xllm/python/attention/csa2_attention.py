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


from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass, field, replace
from itertools import accumulate
from typing import Any

import torch

from scripts.logger import logger
from xllm.python import kernels
from xllm.python.attention.backend import LayerCache
from xllm.python.attention.csa_attention import (
    DsaAttentionBackend,
    _CompressedAttentionCacheMapping,
    _get_layer_cache_tensor,
    _scatter_by_slot,
)
from xllm.python.attention.dsa_metadata import (
    DSA_CACHE_SLIDING_WINDOW,
    DSA_CACHE_TOKEN,
    READ_TABLE_LAYOUT_RING,
    DsaMetadata,
    DsaMetadataBuilder,
    build_deepseek_v41_cache_config,
    last_ring_writes,
)
from xllm.python.model_executor.forward_context import get_forward_context_or_none
from xllm.python.models.deepseek_v41 import (
    V41_QUANT_ASCEND,
    V41_QUANT_NONE,
    fp4_act_qdq_e4m3_scale,
    fp4_act_qdq_e8m0,
)

# Softmax-scale exponent base of the sparse kernel's running-max floor.
_NEG_INF_FLOOR = -1e30
# Bound the [query, head, KV] score temporary.
_ATTENTION_QUERY_TILE = 256
# Tile query rows to bound FP32 indexer scores without changing per-row top-k.
_INDEXER_QUERY_TILE = 256
# Maximum token rows per invocation of v41_partial_rope_.
_PARTIAL_ROPE_MAX_TOKENS = 4096


@dataclass
class _SwaGroupMetadata:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    cache: torch.Tensor


@dataclass(frozen=True)
class _PreparedCsa2Attention:
    query_lens: tuple[int, ...]
    device_block_tables: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class _Csa2GraphQuery:
    lengths: torch.Tensor
    offsets: torch.Tensor
    max_length: torch.Tensor
    empty: torch.Tensor


@dataclass
class _DecoderSwaReplayPlan:
    metadata: DsaMetadata
    requests: torch.Tensor | None
    table_views: dict[int, torch.Tensor] = field(default_factory=dict)

    def select_table(self, table: torch.Tensor | None) -> torch.Tensor | None:
        if table is None or not table.numel() or self.requests is None:
            return table
        key = id(table)
        if key not in self.table_views:
            self.table_views[key] = table.index_select(0, self.requests)
        return self.table_views[key]


@dataclass
class Csa2LayerContext:
    x: torch.Tensor
    qr: torch.Tensor | None
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor
    qr_quantized: tuple[torch.Tensor, torch.Tensor] | None = None
    swa_cache_written: bool = False


@dataclass(frozen=True)
class _Csa2IndexerMetadata:
    committed: torch.Tensor
    residual: torch.Tensor | None
    tiling: torch.Tensor


@dataclass(frozen=True)
class _Csa2DecodeCompressorMetadata:
    rows: torch.Tensor
    starts: torch.Tensor
    owners: torch.Tensor
    committed: torch.Tensor
    park_slots: torch.Tensor | None = None
    previous_rows: torch.Tensor | None = None


@dataclass(frozen=True)
class _Csa2PrefillSwaPages:
    descriptors: torch.Tensor
    table: torch.Tensor
    lengths: torch.Tensor
    max_length: int


@dataclass(frozen=True)
class _Csa2DraftSwaMetadata:
    history_slots: torch.Tensor
    positions: torch.Tensor
    lengths: torch.Tensor
    sequence_lengths: torch.Tensor
    block_table: torch.Tensor
    tiling: torch.Tensor
    cache: torch.Tensor


@dataclass
class CSA2SharedRuntime:
    key_cache: torch.Tensor | None = None
    key_block_table: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_scale: torch.Tensor | None = None
    indexer_metadata: dict[int, _Csa2IndexerMetadata] = field(default_factory=dict)
    candidate_blocks: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    compress_ratio: int = 0
    topk_idxs: torch.Tensor | None = None
    compressed_indices: torch.Tensor | None = None
    candidates: list[torch.Tensor | None] | None = None
    swa_metadata: torch.Tensor | None = None
    prewritten_kv_layer: int | None = None
    draft_swa: _Csa2DraftSwaMetadata | None = None
    prefill_swa_pages: dict[int, _Csa2PrefillSwaPages] = field(default_factory=dict)
    compressed_mla_metadata: dict[int, torch.Tensor] = field(default_factory=dict)
    ring_write_rows: dict[tuple[torch.Tensor, int], tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    rope_tables: dict[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict
    )
    decode_compressor_metadata: dict[tuple[int, int], _Csa2DecodeCompressorMetadata] = field(default_factory=dict)

    def last_swa_writes(
        self, slots: torch.Tensor, values: torch.Tensor, capacity: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Layers in the same SWA group share the physical slot mapping for this forward.
        key = (slots, capacity)
        if key not in self.ring_write_rows:
            rows = torch.arange(slots.numel(), device=slots.device)
            self.ring_write_rows[key] = last_ring_writes(slots, rows, slot_capacity=capacity)
        destinations, rows = self.ring_write_rows[key]
        return destinations, values.index_select(0, rows)


def apply_interleaved_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    inverse: bool = False,
) -> torch.Tensor:
    half = cos_sin_cache.size(-1) // 2
    rows = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
    cos = rows[..., :half].to(torch.float32)
    sin = rows[..., half:].to(torch.float32)
    if inverse:
        sin = -sin
    for _ in range(x.dim() - 2):
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    dim = x.size(-1)
    tail = x[..., dim - 2 * half :].to(torch.float32)
    even = tail[..., 0::2]
    odd = tail[..., 1::2]
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    rotated = rotated.flatten(-2).to(x.dtype)
    if 2 * half == dim:
        return rotated
    return torch.cat((x[..., : dim - 2 * half], rotated), dim=-1)


def _a3_quantize_indexer_query(query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # The A3 indexer rounds its per-head scale to FP16 before quantizing Q.
    query_fp32 = query.float()
    scale = (query_fp32.abs().amax(dim=-1) / 127).half().clamp_min(2.0**-24)
    codes = (query_fp32 / scale.float().unsqueeze(-1)).round().clamp(-127, 127)
    return codes.to(torch.int8), scale


def _a3_indexer_query_qdq(query: torch.Tensor) -> torch.Tensor:
    codes, scale = _a3_quantize_indexer_query(query)
    return codes.float() * scale.float().unsqueeze(-1)


def _indexer_score(
    q: torch.Tensor,
    index_k: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    score = torch.einsum("shd,td->sht", q.float(), index_k.float())
    score = score.relu_() * weights.float().unsqueeze(-1)
    return score.sum(dim=1)


def _select_candidate_blocks(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    width = score.size(-1)
    padded = torch.nn.functional.pad(score, (0, -width % block_size), value=float("-inf"))
    blocks = padded.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = blocks.size(-1)
    # Always retain the newest, partially filled block; full older blocks can outscore it.
    last = torch.div(compress_lens - 1, block_size, rounding_mode="floor")
    blocks = blocks.masked_fill(
        torch.arange(num_blocks, device=score.device) == last.unsqueeze(-1),
        float("inf"),
    )
    top = blocks.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(blocks, dtype=torch.bool).scatter_(-1, top.indices, top.values > float("-inf"))
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def _indexer_topk(
    score: torch.Tensor,
    compress_lens: torch.Tensor,
    index_topk: int,
    offset: int,
) -> torch.Tensor:
    topk = min(index_topk, score.size(-1))
    order = torch.argsort(score, dim=-1, descending=True, stable=True)
    idxs = order[..., :topk]
    idxs = idxs.sort(dim=-1).values
    finite = torch.isfinite(score.gather(-1, idxs))
    visible = compress_lens.unsqueeze(-1)
    return torch.where((idxs < visible) & finite, idxs + offset, -1).to(torch.int32)


def _picked_row_mask(idxs: torch.Tensor, num_rows: int) -> torch.Tensor:
    picked = idxs >= 0
    valid = picked & (idxs < num_rows)
    safe_idxs = torch.where(valid, idxs, torch.full_like(idxs, num_rows))
    wide_mask = torch.zeros(idxs.shape[0], num_rows + 1, dtype=torch.bool, device=idxs.device)
    wide_mask.scatter_(1, safe_idxs, valid)
    return wide_mask[:, :num_rows]


def _window_visible_idxs(
    positions_seq: torch.Tensor,
    col_pos: torch.Tensor,
    cols: torch.Tensor,
    win: int,
) -> torch.Tensor:
    window_lo = torch.clamp(positions_seq - win + 1, min=0)
    valid = (col_pos.unsqueeze(0) >= window_lo.unsqueeze(-1)) & (col_pos.unsqueeze(0) <= positions_seq.unsqueeze(-1))
    return torch.where(valid, cols.unsqueeze(0).expand(positions_seq.numel(), col_pos.numel()), -1)


def _cache_slots(
    block_table: torch.Tensor, seqs: int | torch.Tensor, rows: torch.Tensor, block_size: int
) -> torch.Tensor:
    cols = torch.div(rows, block_size, rounding_mode="floor") % block_table.size(1)
    blocks = block_table[seqs, cols].to(torch.int64).clamp_min(0)
    return blocks * block_size + rows % block_size


def _token_cache_slots(
    block_table: torch.Tensor | None,
    cache: torch.Tensor | None,
    rows: torch.Tensor,
    seqs: torch.Tensor,
    mask: torch.Tensor | None,
) -> torch.Tensor:
    if block_table is None or cache is None:
        return torch.full_like(rows, -1)
    if block_table.numel() == 0:
        return torch.full_like(rows, -1)
    block_size = cache.size(1)
    slots = _cache_slots(block_table, seqs, rows, block_size)
    if mask is not None:
        slots = torch.where(mask, slots, -1)
    return slots


_GRAPH_TOKEN_CAPACITY_SERVING_CAP = 32768


class Csa2AttentionBackend(DsaAttentionBackend):
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
        kv_source_layer_ids: list[int],
        index_source_layer_ids: list[int],
        candidate_source_layer_id: int,
        candidate_topk_blocks: int,
        candidate_block_size: int,
        max_model_len: int = 0,
        quant_mode: str = V41_QUANT_NONE,
        graph_backend: str = "off",
        dspark_block_size: int = 0,
        dp_rank: int = 0,
        decoder_swa_start_layer: int | None = None,
    ) -> None:
        self.caches_info, self.group_infos = build_deepseek_v41_cache_config(
            compress_ratios,
            kv_source_layer_ids,
            window_size,
            n_layers,
            decoder_swa_start_layer,
        )
        self._builder = DsaMetadataBuilder(
            self.caches_info,
            self.group_infos,
            read_table_layout=READ_TABLE_LAYOUT_RING,
            preserve_query_padding=True,
        )
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
        self.uses_a3_cache_precision = device.type in ("npu", "privateuseone") and torch.npu.get_device_name(
            device
        ).startswith("Ascend910_93")
        self._rope_swap_offsets = None
        if self.uses_a3_cache_precision and dtype == torch.bfloat16 and rope_head_dim == 64:
            self._rope_swap_offsets = torch.tensor(
                [(lane ^ 1) * 4 for lane in range(8 * 64)], dtype=torch.int32, device=device
            )

        self.index_source_layer_ids = sorted({int(source) for source in index_source_layer_ids})
        self.candidate_source_layer_id = int(candidate_source_layer_id)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
        # SparseFlashMla consumes min(visible_compressed_tokens, index_topk)
        # picks per query. A retained partial block can supply just one token.
        min_candidate_tokens = (self.candidate_topk_blocks - 1) * self.candidate_block_size + 1
        self.native_mla_full_topk = self.candidate_source_layer_id < 0 or min_candidate_tokens >= self.index_topk
        if self.candidate_source_layer_id >= 0 and self.candidate_source_layer_id not in self.index_source_layer_ids:
            raise ValueError(
                "DeepSeek-V4.1 candidate source layer must also be an index source layer: "
                f"{self.candidate_source_layer_id}"
            )

        self.moe_graph_capturable = quant_mode == V41_QUANT_ASCEND

        raw_granularity = os.environ.get("XLLM_V41_GRAPH_CAPACITY_GRANULARITY")
        if raw_granularity is not None and raw_granularity.strip() != "":
            self.graph_token_capacity_granularity = int(raw_granularity)
            granularity_source = "XLLM_V41_GRAPH_CAPACITY_GRANULARITY"
        else:
            if max_model_len <= 0:
                self.graph_token_capacity_granularity = 0
            else:
                serving_capped = min(max_model_len, _GRAPH_TOKEN_CAPACITY_SERVING_CAP)
                scaled = (serving_capped + 15) // 16
                self.graph_token_capacity_granularity = max(1024, ((scaled + 1023) // 1024) * 1024)
            granularity_source = f"derived from max_model_len={max_model_len}"
        logger.info(
            f"DeepSeek-V4.1 CSA2 graph_token_capacity_granularity="
            f"{self.graph_token_capacity_granularity} ({granularity_source})"
        )

        if graph_backend == "aclgraph" and not self.moe_graph_capturable:
            logger.warning(
                f"DeepSeek-V4.1 quant_mode={quant_mode} serves the torch MoE, which no ACL graph "
                "can capture; decode-graph admission is disabled and decode stays eager"
            )
        elif graph_backend == "aclgraph" and self.graph_token_capacity_granularity <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 ACL-graph serving needs a static committed-row capacity, but "
                "XLLM_V41_GRAPH_CAPACITY_GRANULARITY is unset/empty and max_model_len was not "
                "provided to derive one. Set the env var or provide max_position_embeddings, or "
                "disable graph execution."
            )

        self._kv_caches: list[LayerCache] = []
        self.uses_device_length_indexer = False
        self.dspark_block_size = dspark_block_size
        self.dp_rank = dp_rank
        self._metadata: Any | None = None
        self._v41_shared: CSA2SharedRuntime | None = None

    def bind_kv_caches(self, kv_caches: list[LayerCache]) -> None:
        indexed = [cache for cache in kv_caches if cache.index is not None]
        quantized = [cache for cache in indexed if cache.index.dtype == torch.int8]
        if quantized:
            if len(quantized) != len(indexed) or any(
                cache.indexer_scale is None
                or cache.indexer_scale.dtype != torch.float16
                or cache.indexer_scale.shape != cache.index.shape[:-1]
                for cache in quantized
            ):
                raise ValueError("DeepSeek-V4.1 INT8 index caches require one FP16 scale per key row")
            if not self.uses_a3_cache_precision or self.index_head_dim != 128 or self.index_n_heads not in (32, 64):
                raise ValueError("DeepSeek-V4.1 INT8 index cache requires compatible A3 indexer geometry")
        super().bind_kv_caches(kv_caches)
        self.uses_device_length_indexer = bool(quantized)

    def reset_forward(self, metadata: Any | None = None, *, preserve_dsa_metadata: bool = False) -> None:
        if not preserve_dsa_metadata:
            super().reset_forward(metadata)
        self._v41_shared = None

    @property
    def supports_prepared_metadata(self) -> bool:
        return True

    @property
    def uses_grouped_cache(self) -> bool:
        return True

    def prepare_graph_replay(self, metadata: Any) -> None:
        if isinstance(getattr(metadata, "prepared_attention_state", None), _PreparedCsa2Attention):
            # Prepared graphs derive lengths, tables and slots inside capture.
            # The inherited Host refresh would overwrite graph-pool scratch
            # whose lifetime overlaps other intermediates in that graph.
            self.prepare(metadata, graph_mode=True)
        else:
            super().prepare_graph_replay(metadata)

    def prepare(self, metadata: Any, *, graph_mode: bool = False) -> None:
        super().prepare(metadata, graph_mode=graph_mode)
        prepared = getattr(metadata, "prepared_attention_state", None)
        if graph_mode and isinstance(prepared, _PreparedCsa2Attention):
            self._graph_query(prepared)

    def _graph_query(self, prepared: _PreparedCsa2Attention) -> _Csa2GraphQuery:
        width = self.dspark_block_size or 1
        rows = len(prepared.query_lens)
        context = get_forward_context_or_none()
        state = context.execution_state if context is not None else None
        key = ("csa2_graph_query", rows, width)
        if state is not None and key in state.persistent_buffers:
            return state.persistent_buffers[key]
        options = {"device": self.device, "dtype": torch.int32}
        query = _Csa2GraphQuery(
            lengths=torch.full((rows,), width, **options),
            offsets=torch.arange(0, (rows + 1) * width, width, **options),
            max_length=torch.full((), width, **options),
            empty=torch.empty(0, **options),
        )
        if state is not None:
            state.persistent_buffers[key] = query
        return query

    def prepare_metadata(self, metadata: Any, *, device_kv_lengths: bool = False) -> _PreparedCsa2Attention:
        """Bind the decode graph's persistent grouped tables."""
        del device_kv_lengths
        tables = (metadata.block_table,) if self.dspark_block_size else tuple(metadata.multi_block_tables)
        query_lens = (self.dspark_block_size or 1,) * tables[0].size(0)
        if len(tables) != len(self.group_infos) or any(
            table.dtype != torch.int32
            or table.device != self.device
            or table.ndim != 2
            or table.shape[0] != len(query_lens)
            or not table.is_contiguous()
            for table in tables
        ):
            raise ValueError("prepared CSA2 requires final row-aligned int32 Device tables for every cache group")
        return _PreparedCsa2Attention(query_lens, tables)

    def _build_prepared_graph_metadata(self, metadata: Any, prepared: _PreparedCsa2Attention) -> DsaMetadata:
        # The decode runner updates lengths and grouped tables before replay.
        # Derive their dependent metadata inside capture.
        width = 1
        if any(length != width for length in prepared.query_lens):
            raise ValueError("prepared CSA2 requires uniform query rows matching its execution width")
        positions = metadata.dsa_positions
        query = self._graph_query(prepared)
        capacity = int(metadata.dsa_graph_token_capacity)
        if capacity <= 0:
            raise ValueError("prepared CSA2 graphs require a positive KV capacity")
        swa_group, *token_groups = self.group_infos
        swa_source, *token_sources = prepared.device_block_tables
        swa_cols = (capacity + swa_group.block_size - 1) // swa_group.block_size
        swa_table = swa_source[:, :swa_cols].contiguous()
        swa_column = torch.div(positions, swa_group.block_size, rounding_mode="floor")
        swa_block = swa_table.gather(1, swa_column.reshape(-1, 1)).flatten()
        active = swa_block > 0
        swa_slot = swa_block.to(torch.int64) * swa_group.block_size + positions % swa_group.block_size
        tables = [swa_table]
        slots = [torch.where(active, swa_slot, -1).to(torch.int32)]
        for group, source in zip(token_groups, token_sources, strict=True):
            ratio = group.ratio
            columns = (capacity + ratio * group.block_size - 1) // (ratio * group.block_size)
            table = source[:, :columns].contiguous()
            tables.append(table)
            index = torch.div(positions + 1, ratio, rounding_mode="floor") - 1
            column = torch.div(index.clamp_min(0), group.block_size, rounding_mode="floor")
            block = table.gather(1, column.reshape(-1, 1)).flatten()
            committed = active & ((positions + 1) % ratio == 0) & (block > 0)
            slot = block.to(torch.int64) * group.block_size + index % group.block_size
            slots.append(torch.where(committed, slot, -1).to(torch.int32))
        lengths = torch.where(active, metadata.kv_seq_lens, 0).to(torch.int32)
        empty = query.empty
        device_lengths = self.uses_device_length_indexer
        # Device-length decode consumers use the per-request lengths directly.
        # The cumulative KV lengths and maxima belong to the generic DSA path.
        kv_offsets = empty if device_lengths else torch.cat((lengths.new_zeros(1), lengths.cumsum(0).to(torch.int32)))
        dsa = DsaMetadata(
            layer_id=-1,
            seq_lens=lengths,
            seq_lens_q=query.lengths,
            actual_seq_lengths_kv=lengths,
            actual_seq_lengths_query=query.offsets,
            kv_cu_seq_lens=kv_offsets,
            max_seqlen_kv=empty if device_lengths else lengths.max(),
            max_seqlen_q=query.max_length,
            max_query_len=width,
            max_seq_len=capacity,
            input_positions=positions,
            c4_pad_positions=empty,
            c128_pad_positions=empty,
            start_pos=empty if device_lengths else lengths - width,
            is_acl_graph=True,
        )
        dsa.seq_lens_q_host = list(prepared.query_lens)
        dsa.graph_token_capacity = capacity
        dsa.query_aligned_token_slots = width == 1
        dsa.block_tables = [[tables[cache.group_id] for cache in layer] for layer in self.caches_info]
        dsa.slot_mappings = [[slots[cache.group_id] for cache in layer] for layer in self.caches_info]
        return dsa

    def _build_dsa_metadata_for_forward(self, metadata: Any) -> DsaMetadata:
        prepared = getattr(metadata, "prepared_attention_state", None)
        if isinstance(prepared, _PreparedCsa2Attention) and self.dspark_block_size:
            query = self._graph_query(prepared)
            lengths = metadata.kv_seq_lens
            dsa = DsaMetadata(
                layer_id=-1,
                seq_lens=lengths,
                seq_lens_q=query.lengths,
                actual_seq_lengths_kv=lengths,
                actual_seq_lengths_query=query.offsets,
                kv_cu_seq_lens=query.empty,
                max_seqlen_kv=query.empty,
                max_seqlen_q=query.max_length,
                max_query_len=self.dspark_block_size,
                max_seq_len=self.window_size + self.dspark_block_size,
                input_positions=metadata.dsa_positions,
                c4_pad_positions=query.empty,
                c128_pad_positions=query.empty,
                start_pos=lengths - self.dspark_block_size,
                is_acl_graph=True,
            )
            dsa.seq_lens_q_host = list(prepared.query_lens)
            dsa.block_tables = [[prepared.device_block_tables[0]] for _ in self.caches_info]
            return dsa
        if isinstance(prepared, _PreparedCsa2Attention) and bool(getattr(metadata, "dsa_graph_mode", False)):
            return self._build_prepared_graph_metadata(metadata, prepared)
        dsa = super()._build_dsa_metadata_for_forward(metadata)
        raw_counts = getattr(metadata, "raw_dp_execution_token_counts", ())
        if getattr(metadata, "is_dummy", False) or (raw_counts and raw_counts[self.dp_rank] == 0):
            # V4's synthetic tiling context is not initialized history for tensor attention.
            dsa.seq_lens.zero_()
            dsa.seq_lens_kv_host = [0] * len(dsa.seq_lens_kv_host)
            dsa.actual_seq_lengths_kv.zero_()
            dsa.kv_cu_seq_lens.zero_()
            dsa.max_seqlen_kv.zero_()
            dsa.max_seq_len = 0
            dsa.start_pos.copy_(-dsa.seq_lens_q)
            for layer_slots in dsa.slot_mappings:
                for slots in layer_slots:
                    slots.fill_(-1)
        return dsa

    def _build_precomputed_metadata(
        self,
        compressed_metadata: DsaMetadata,
        metadata: Any = None,
        cu_seqlens_ori_kv_override: torch.Tensor | None = None,
        max_query_len_override: int | None = None,
        max_seq_len_override: int | None = None,
    ) -> None:
        del (
            compressed_metadata,
            metadata,
            cu_seqlens_ori_kv_override,
            max_query_len_override,
            max_seq_len_override,
        )

    def _resolve_cache_mapping(self, layer_id: int, compress_ratio: int) -> _CompressedAttentionCacheMapping:
        mapping = _CompressedAttentionCacheMapping()
        if layer_id < 0 or layer_id >= len(self.caches_info):
            return mapping
        token_indices: list[int] = []
        swa_indices: list[int] = []
        for cache_idx, ci in enumerate(self.caches_info[layer_id]):
            if ci.cache_type == DSA_CACHE_TOKEN:
                token_indices.append(cache_idx)
            elif ci.cache_type == DSA_CACHE_SLIDING_WINDOW:
                swa_indices.append(cache_idx)
        if token_indices and compress_ratio > 0:
            mapping.cmp_cache_idx = token_indices[0]
        if len(token_indices) > 1:
            mapping.index_cache_idx = token_indices[1]
        if swa_indices:
            mapping.ori_cache_idx = swa_indices[0]
        if len(swa_indices) > 1:
            mapping.kv_state_cache_idx = swa_indices[1]
        if len(swa_indices) > 2:
            mapping.score_state_cache_idx = swa_indices[2]
        return mapping

    def swa_group_metadata(self, dsa: DsaMetadata | None) -> _SwaGroupMetadata | None:
        if dsa is None or not self._kv_caches:
            return None
        mapping = self._resolve_cache_mapping(0, 0)
        layer_cache = self._kv_caches[0]
        if layer_cache.swa is None or mapping.ori_cache_idx < 0:
            return None
        slot_mapping = _get_layer_cache_tensor(dsa.slot_mappings, 0, mapping.ori_cache_idx)
        block_table = _get_layer_cache_tensor(dsa.block_tables, 0, mapping.ori_cache_idx)
        if slot_mapping is None or block_table is None:
            return None
        return _SwaGroupMetadata(
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=layer_cache.swa.size(1),
            cache=layer_cache.swa,
        )

    def apply_projection_rope(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        inverse: bool = False,
    ) -> torch.Tensor:
        dsa = getattr(self._current_forward_metadata(), "dsa_metadata", None)
        return self._apply_rope(value, positions, cos_sin_cache, dsa, self._shared_runtime(), inverse)

    def prepare_projection_rope(
        self, value: torch.Tensor, positions: torch.Tensor, cos_sin_cache: torch.Tensor
    ) -> bool:
        """Materialize the shared coefficients before the Q and KV streams fork."""
        dsa = getattr(self._current_forward_metadata(), "dsa_metadata", None)
        return self._rope_arguments(value, positions, cos_sin_cache, dsa, self._shared_runtime()) is not None

    def _apply_rope(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        dsa: DsaMetadata | None,
        shared: CSA2SharedRuntime,
        inverse: bool = False,
    ) -> torch.Tensor:
        arguments = self._rope_arguments(value, positions, cos_sin_cache, dsa, shared)
        if arguments is None:
            return apply_interleaved_rope(value, positions, cos_sin_cache, inverse=inverse)
        return self._rotate_in_place(value, *arguments, inverse)

    @staticmethod
    def _rotate_in_place(
        value: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        offsets: torch.Tensor,
        inverse: bool,
    ) -> torch.Tensor:
        if value.size(0) <= _PARTIAL_ROPE_MAX_TOKENS:
            torch.ops.xllm_ops.v41_partial_rope_(value, cos, sin, offsets, inverse)
        else:
            for start in range(0, value.size(0), _PARTIAL_ROPE_MAX_TOKENS):
                rows = slice(start, start + _PARTIAL_ROPE_MAX_TOKENS)
                torch.ops.xllm_ops.v41_partial_rope_(value[rows], cos[rows], sin[rows], offsets, inverse)
        return value

    def _rope_arguments(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        dsa: DsaMetadata | None,
        shared: CSA2SharedRuntime,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        swap_offsets = self._rope_swap_offsets
        if (
            swap_offsets is None
            or dsa is None
            or value.ndim not in (2, 3)
            or value.dtype != torch.bfloat16
            or not value.is_contiguous()
            or value.size(0) == 0
            or (dsa.is_acl_graph and value.size(0) > _PARTIAL_ROPE_MAX_TOKENS)
            or positions.numel() != value.size(0)
            or value.size(-1) not in (128, 512)
            or (value.ndim == 3 and not 0 < value.size(1) <= 128)
            or cos_sin_cache.dtype != torch.float32
            or cos_sin_cache.size(-1) != 64
        ):
            return None

        tables = shared.rope_tables
        # Eager compressor positions can be temporary. Keep them alive for the lifetime of the cached coefficients.
        key = (positions, cos_sin_cache)
        if key not in tables:
            compact = cos_sin_cache.index_select(0, positions.reshape(-1).to(torch.int64))
            cos, sin = compact.chunk(2, dim=-1)
            # Flip the even lane before the fused kernel swaps each pair.
            signed_sin = torch.stack((-sin, sin), dim=-1).flatten(1)
            tables[key] = (cos.repeat_interleave(2, dim=-1), signed_sin)
        return *tables[key], swap_offsets

    def prepare_swa_kv(
        self,
        value: torch.Tensor,
        positions: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        layer: Any,
    ) -> tuple[torch.Tensor, bool]:
        dsa = getattr(self._current_forward_metadata(), "dsa_metadata", None)
        shared = self._shared_runtime()
        arguments = self._rope_arguments(value, positions, cos_sin_cache, dsa, shared)
        if arguments is None:
            return apply_interleaved_rope(value, positions, cos_sin_cache), False

        if (
            dsa is not None
            and dsa.is_acl_graph
            and dsa.query_aligned_token_slots
            and self.window_size == 128
            and value.ndim == 2
            and value.size(1) == 512
        ):
            layer_id = layer.layer_id
            mapping = self._resolve_cache_mapping(layer_id, layer.mode.compress_ratio)
            cache = self._kv_caches[layer_id].swa
            slots = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
            if (
                cache is not None
                and cache.dtype == value.dtype
                and cache.is_contiguous()
                and 0 < cache.numel() // 512 <= (1 << 21)
                and slots is not None
                and slots.is_contiguous()
                and slots.dtype == torch.int32
                and slots.numel() == value.size(0)
            ):
                # Prepared single-token rows have distinct write slots. Publish
                # them before attention reads the current and earlier query rows.
                torch.ops.xllm_ops.v41_swa_rope_cache_(value, *arguments, slots.reshape(-1), cache.view(-1, 512))
                return value, True

        return self._rotate_in_place(value, *arguments, False), False

    def _write_cache_rows(self, cache: torch.Tensor, slots: torch.Tensor, value: torch.Tensor, *, graph: bool) -> None:
        if (
            graph
            and self.uses_a3_cache_precision
            and cache.device.type == "npu"
            and slots.dtype == torch.int32
            and slots.numel() == value.size(0) > 0
            and value.dtype == cache.dtype
        ):
            # The A3 op skips -1 slots. Send the prepared indices directly.
            width = value.size(-1)
            kernels.scatter_nd_update(cache.view(-1, width), slots.reshape(-1, 1), value.reshape(-1, width))
            return
        _scatter_by_slot(cache, slots, value)

    def execute_csa2_layer(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        metadata = self._current_forward_metadata()
        dsa = getattr(metadata, "dsa_metadata", None)
        if dsa is None:
            raise RuntimeError("DeepSeek-V4.1 requires prepared DSA metadata before attention")
        if not dsa.is_acl_graph and dsa.max_seq_len == 0:
            # An idle DP shard still runs the layer's EP collectives. Its
            # synthetic query rows have no paged KV to read or update.
            return torch.zeros_like(q)
        mode = layer.mode
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, mode.compress_ratio)
        ctx = layer._csa2_ctx
        if ctx is None:
            raise RuntimeError(f"DeepSeek-V4.1 layer {layer_id} ran without its CSA2 context")
        q_lens = dsa.seq_lens_q_host
        ctx_lens = dsa.seq_lens

        shared = self._shared_runtime()
        if self.dspark_block_size:
            return self._fused_draft_swa(q, kv, layer, dsa, mapping, shared)
        if (
            self.uses_a3_cache_precision
            and mode.compress_ratio == 0
            and self.head_dim == 512
            and self.window_size == 128
            and q.dtype in (torch.bfloat16, torch.float16)
        ):
            return self._fused_swa(q, kv, layer, dsa, mapping, shared, cache_written=ctx.swa_cache_written)

        if mode.is_full and shared.prewritten_kv_layer != layer_id:
            self._run_compressor(layer, ctx, dsa, layer_cache, mapping, q_lens, ctx_lens, shared)
        if mode.is_full or mode.is_reindex:
            self._run_indexer(layer, ctx, dsa, q_lens, ctx_lens, shared)
        topk = shared.topk_idxs if mode.compress_ratio else None
        cmp_indices = shared.compressed_indices if mode.compress_ratio else None
        if mode.is_reuse and topk is None and cmp_indices is None and dsa.block_tables:
            raise RuntimeError(
                f"DeepSeek-V4.1 reuse layer {layer_id} has no shared top-k; its group source must run first"
            )
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        write_swa = layer_cache.swa is not None and swa_slot is not None and swa_slot.numel() > 0
        flat_graph = (
            write_swa and dsa.is_acl_graph and mode.compress_ratio > 0 and all(length == 1 for length in q_lens)
        )
        if flat_graph and not ctx.swa_cache_written:
            # Verification flattens a request's consecutive queries into
            # separate rows. Later rows gather preceding queries from SWA.
            self._write_cache_rows(layer_cache.swa, swa_slot, kv, graph=True)
        if (
            write_swa
            and mode.compress_ratio in (1, 2)
            and cmp_indices is not None
            and cmp_indices.numel()
            and self.uses_a3_cache_precision
            and q.size(-1) == 512
            and self.window_size == 128
            and q.dtype in (torch.bfloat16, torch.float16)
            and layer_cache.swa.dtype == q.dtype
            and shared.key_cache is not None
            and shared.key_cache.dtype == q.dtype
            and self.native_mla_full_topk
            and (not dsa.is_acl_graph or flat_graph)
        ):
            output = self._fused_compressed_attention(q, kv, layer, dsa, mapping, cmp_indices, shared)
        else:
            if topk is None and cmp_indices is not None:
                topk = self._reference_topk_indices(cmp_indices, q_lens)
                shared.topk_idxs = topk
            output = self._csa2_attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared)
        if write_swa and not flat_graph:
            # Chunk attention reads current KV directly, so preserve its old
            # ring rows until every query in the chunk has consumed them.
            if not dsa.is_acl_graph:
                swa_slot, kv = shared.last_swa_writes(swa_slot, kv, layer_cache.swa.numel() // kv.size(-1))
            _scatter_by_slot(layer_cache.swa, swa_slot, kv)
        return output

    def prepare_decoder_replay_kv(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
    ) -> None:
        dsa = self._current_forward_metadata().dsa_metadata
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, layer.mode.compress_ratio)
        shared = self._shared_runtime()
        self._run_compressor(layer, ctx, dsa, layer_cache, mapping, dsa.seq_lens_q_host, dsa.seq_lens, shared)
        shared.prewritten_kv_layer = layer_id

    def prepare_decoder_swa_replay(
        self,
        dsa: DsaMetadata,
        lengths: list[int],
        rows: torch.Tensor | None,
        history_floors: list[int],
        *,
        dummy: bool = False,
    ) -> _DecoderSwaReplayPlan:
        active = [seq for seq, length in enumerate(lengths) if length]
        all_requests = len(active) == len(lengths)
        requests = None if all_requests else dsa.input_positions.new_tensor(active, dtype=torch.int64)
        tail = copy(dsa)
        plan = _DecoderSwaReplayPlan(tail, requests)
        tail.swa_history_floors = tuple(history_floors[seq] for seq in active)
        if rows is not None:
            if not all_requests:
                kv_lens = [dsa.seq_lens_kv_host[seq] for seq in active]
                tail.seq_lens_kv_host = kv_lens
                tail.seq_lens = dsa.seq_lens.new_tensor(kv_lens)
                tail.actual_seq_lengths_kv = tail.seq_lens
                tail.kv_cu_seq_lens = dsa.kv_cu_seq_lens.new_tensor([0, *accumulate(kv_lens)])
                tail.max_seq_len = max(kv_lens)
                tail.max_seqlen_kv = dsa.max_seqlen_kv.new_full(dsa.max_seqlen_kv.shape, tail.max_seq_len)
                tail.block_tables = [[plan.select_table(table) for table in tables] for tables in dsa.block_tables]

            query_lens = [lengths[seq] for seq in active]
            tail.seq_lens_q_host = query_lens
            tail.seq_lens_q = dsa.seq_lens_q.new_tensor(query_lens)
            tail.actual_seq_lengths_query = dsa.actual_seq_lengths_query.new_tensor([0, *accumulate(query_lens)])
            tail.max_query_len = max(query_lens)
            tail.max_seqlen_q = dsa.max_seqlen_q.new_full(dsa.max_seqlen_q.shape, tail.max_query_len)
            tail.input_positions = dsa.input_positions.index_select(0, rows)
            tail.start_pos = (tail.seq_lens - tail.seq_lens_q).to(torch.int32)
            # The source already wrote compressed KV for all requests. The
            # decoder only needs compact SWA write slots for its live queries.
            full_rows = sum(dsa.seq_lens_q_host)
            slot_views: dict[int, torch.Tensor] = {}
            tail.slot_mappings = []
            for layer_slots in dsa.slot_mappings:
                slots = []
                for mapping in layer_slots:
                    key = id(mapping)
                    if key not in slot_views:
                        selected = mapping.index_select(0, rows) if mapping.numel() == full_rows else mapping
                        slot_views[key] = torch.full_like(selected, -1) if dummy else selected
                    slots.append(slot_views[key])
                tail.slot_mappings.append(slots)

        return plan

    @contextmanager
    def decoder_swa_replay(self, plan: _DecoderSwaReplayPlan) -> Iterator[DsaMetadata]:
        original = self._shared_runtime()
        self._v41_shared = replace(
            original,
            key_block_table=plan.select_table(original.key_block_table),
            index_block_table=plan.select_table(original.index_block_table),
            topk_idxs=None,
            candidates=None,
            candidate_blocks=None,
            compressed_indices=None,
            indexer_metadata={},
            compressed_mla_metadata={},
            swa_metadata=None,
            prefill_swa_pages={},
        )
        try:
            yield plan.metadata
        finally:
            self._v41_shared = original

    def _fused_swa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        mapping: _CompressedAttentionCacheMapping,
        shared: CSA2SharedRuntime,
        *,
        cache_written: bool = False,
    ) -> torch.Tensor:
        if q.size(0) == 0:
            return torch.zeros_like(q)
        cache = self._kv_caches[layer.layer_id].swa
        slots = _get_layer_cache_tensor(dsa.slot_mappings, layer.layer_id, mapping.ori_cache_idx)
        table = _get_layer_cache_tensor(dsa.block_tables, layer.layer_id, mapping.ori_cache_idx)
        if cache is None or slots is None or table is None:
            raise RuntimeError("DeepSeek-V4.1 fused SWA requires a paged cache and slot mapping")

        cache = cache.view(-1, cache.size(1), 1, self.head_dim)
        max_query_len = 1 if dsa.is_acl_graph else max(dsa.seq_lens_q_host)
        packed_prefill = max_query_len > 1 or bool(dsa.swa_history_floors)
        if packed_prefill:
            # A prefill chunk can reuse a ring page before its earliest
            # queries are finished with it. Read the live history first.
            attention_kv, attention_table, seq_lens, max_kv_len = self._pack_swa_prefill(kv, cache, table, dsa)
        else:
            if not cache_written:
                self._write_cache_rows(cache, slots, kv, graph=dsa.is_acl_graph)
            attention_kv, attention_table, seq_lens = cache, table, dsa.seq_lens
            max_kv_len = table.size(1) * cache.size(1) if dsa.is_acl_graph else dsa.max_seq_len
        cu_seqlens_q = dsa.actual_seq_lengths_query
        if shared.swa_metadata is None:
            shared.swa_metadata = torch.ops.xllm_ops.sparse_flash_mla_swa_metadata(
                cu_seqlens_q, seq_lens, q.size(1), q.size(2), max_query_len, max_kv_len, self.window_size
            )
        sinks = layer.attn_sink.float() if layer.attn_sink_loaded else None
        output = torch.ops.xllm_ops.sparse_flash_mla_swa(
            q,
            attention_kv,
            attention_table,
            cu_seqlens_q,
            seq_lens,
            sinks,
            shared.swa_metadata,
            self.scale,
            self.window_size,
        )
        if packed_prefill:
            slots, kv = shared.last_swa_writes(slots, kv, cache.numel() // self.head_dim)
            _scatter_by_slot(cache, slots, kv)
        return output

    def _fused_draft_swa(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        mapping: _CompressedAttentionCacheMapping,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        from xllm.python.kernels_npu.triton.draft_swa import draft_swa_metadata, pack_draft_swa

        width = self.dspark_block_size
        window = self.window_size
        span = window + width
        ring = self._kv_caches[layer.layer_id].swa
        metadata = shared.draft_swa
        if metadata is None:
            table = _get_layer_cache_tensor(dsa.block_tables, layer.layer_id, mapping.ori_cache_idx)
            block_size = ring.size(1)
            batch = dsa.seq_lens.numel()
            slots, positions, lengths, sequence_lengths, block_table = draft_swa_metadata(
                dsa.seq_lens, table, width, window, block_size
            )
            # Speculative rows must never overwrite the committed ring read by another query.
            cache = q.new_empty((batch, block_table.size(1) * block_size, self.head_dim))
            topk_width = positions.size(-1)
            tiling = torch.ops.xllm_ops.sparse_flash_mla_metadata(
                dsa.actual_seq_lengths_query,
                sequence_lengths,
                None,
                None,
                q.size(1),
                q.size(2),
                width,
                span,
                0,
                0,
                0,
                window,
                lengths,
                topk_width,
            )
            metadata = _Csa2DraftSwaMetadata(slots, positions, lengths, sequence_lengths, block_table, tiling, cache)
            shared.draft_swa = metadata

        pack_draft_swa(ring, kv, metadata.history_slots, metadata.cache)
        sinks = layer.attn_sink.float() if layer.attn_sink_loaded else None
        return torch.ops.xllm_ops.sparse_flash_mla(
            q,
            metadata.cache.view(-1, ring.size(1), 1, self.head_dim),
            metadata.block_table,
            dsa.actual_seq_lengths_query,
            metadata.sequence_lengths,
            None,
            None,
            None,
            None,
            None,
            sinks,
            metadata.tiling,
            self.scale,
            0,
            window,
            metadata.positions,
            metadata.lengths,
        )

    def _fused_compressed_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        mapping: _CompressedAttentionCacheMapping,
        cmp_indices: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        swa = self._kv_caches[layer.layer_id].swa
        swa_table = _get_layer_cache_tensor(dsa.block_tables, layer.layer_id, mapping.ori_cache_idx)
        cmp_kv, cmp_table = shared.key_cache, shared.key_block_table
        if swa is None or swa_table is None or cmp_kv is None or cmp_table is None:
            raise RuntimeError("Fused compressed attention requires paged SWA and compressed KV")
        swa = swa.view(-1, swa.size(1), 1, self.head_dim)
        cmp_kv = cmp_kv.view(-1, cmp_kv.size(1), 1, self.head_dim)
        ratio = layer.mode.compress_ratio
        q_lens = dsa.seq_lens_q_host
        if dsa.is_acl_graph:
            swa_lens = dsa.seq_lens
            max_swa_len = swa_table.size(1) * swa.size(1)
        else:
            # Keep SWA history alive until all queries in this chunk have read it.
            swa, swa_table, swa_lens, max_swa_len = self._pack_swa_prefill(kv, swa, swa_table, dsa)
        indexer_metadata = shared.indexer_metadata[ratio]
        cmp_lens, cmp_residual = indexer_metadata.committed, indexer_metadata.residual
        cu_seqlens_q = dsa.actual_seq_lengths_query
        metadata = shared.compressed_mla_metadata.get(ratio)
        if metadata is None:
            metadata = torch.ops.xllm_ops.sparse_flash_mla_metadata(
                cu_seqlens_q,
                swa_lens,
                cmp_lens,
                cmp_residual,
                q.size(1),
                q.size(2),
                max(q_lens),
                max_swa_len,
                self._committed_capacity(dsa, ratio, dsa.seq_lens),
                cmp_indices.size(-1),
                ratio,
                self.window_size,
            )
            shared.compressed_mla_metadata[ratio] = metadata
        sinks = layer.attn_sink.float() if layer.attn_sink_loaded else None
        return torch.ops.xllm_ops.sparse_flash_mla(
            q,
            swa,
            swa_table,
            cu_seqlens_q,
            swa_lens,
            cmp_kv,
            cmp_table,
            cmp_indices,
            cmp_lens,
            cmp_residual,
            sinks,
            metadata,
            self.scale,
            ratio,
            self.window_size,
        )

    def _pack_swa_prefill(
        self,
        kv: torch.Tensor,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        dsa: DsaMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        block_size = cache.size(1)
        use_fused_pack = kv.device.type in ("npu", "privateuseone") and self.head_dim == 512 and cache.is_contiguous()
        if use_fused_pack:
            from xllm.python.kernels_npu.triton.prefill_swa import pack_prefill_swa

            shared = self._shared_runtime()
            pages = shared.prefill_swa_pages.get(block_size)
            if pages is not None:
                packed = pack_prefill_swa(kv, cache, block_table, pages.descriptors)
                return packed, pages.table, pages.lengths, pages.max_length
        q_lens = dsa.seq_lens_q_host
        history_ends = [max(0, length - query_len) for length, query_len in zip(dsa.seq_lens_kv_host, q_lens)]
        floors = dsa.swa_history_floors or (0,) * len(q_lens)
        history_lens = [
            min(max(0, end - floor), self.window_size - 1) for end, floor in zip(history_ends, floors, strict=True)
        ]
        kv_lens = [
            past + min(query_len, length) for past, query_len, length in zip(history_lens, q_lens, dsa.seq_lens_kv_host)
        ]
        block_counts = [(length + block_size - 1) // block_size for length in kv_lens]
        table_width = max(block_counts)
        rows: list[list[int]] = []
        descriptors = [(0, 0, 0, 0, 0, 0)]  # Page zero supplies padding for the attention block table.
        block_start = 1
        token_start = 0
        for seq, (query_len, past, end, length, blocks) in enumerate(
            zip(q_lens, history_lens, history_ends, kv_lens, block_counts)
        ):
            rows.append(list(range(block_start, block_start + blocks)) + [0] * (table_width - blocks))
            if use_fused_pack:
                descriptors.extend(
                    (seq, page * block_size, end - past, past, token_start, length) for page in range(blocks)
                )
            block_start += blocks
            token_start += query_len

        table = torch.tensor(rows, dtype=torch.int32, device=cache.device)
        seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=cache.device)
        max_length = max(kv_lens)
        if use_fused_pack:
            pages = _Csa2PrefillSwaPages(
                torch.tensor(descriptors, dtype=torch.int32, device=cache.device), table, seq_lens, max_length
            )
            shared.prefill_swa_pages[block_size] = pages
            packed = pack_prefill_swa(kv, cache, block_table, pages.descriptors)
        else:
            packed = cache.new_zeros((block_start, block_size, 1, self.head_dim))
            cache_rows = cache.view(-1, self.head_dim)
            row_offsets = torch.arange(self.window_size - 1, device=cache.device)
            token_start = 0
            for seq, (query_len, past, end, length, blocks) in enumerate(
                zip(q_lens, history_lens, history_ends, kv_lens, block_counts)
            ):
                if blocks:
                    first = rows[seq][0]
                    dst = packed[first : first + blocks].view(-1, self.head_dim)
                    if past:
                        positions = row_offsets[:past] + (end - past)
                        dst[:past].copy_(
                            self._gather_swa_positions(cache_rows, block_table, seq, positions, block_size)
                        )
                    dst[past:length].copy_(kv[token_start : token_start + length - past])
                token_start += query_len
        return packed, table, seq_lens, max_length

    def _run_compressor(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        layer_cache: Any,
        mapping: _CompressedAttentionCacheMapping,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> None:
        ratio = layer.mode.compress_ratio
        layer_id = layer.layer_id
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        kv_state = layer_cache.compress_kv_state
        score_state = layer_cache.compress_score_state
        kv_state_flat = kv_state.view(-1, layer_cache.swa.size(-1)) if kv_state is not None else None
        score_state_flat = score_state.view(-1, layer_cache.swa.size(-1)) if score_state is not None else None

        decode = None
        swa_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.ori_cache_idx)
        if (
            self.uses_a3_cache_precision
            and dsa.is_acl_graph
            and q_lens
            and all(width == 1 for width in q_lens)
            and (
                ratio == 1
                or (
                    swa_table is not None
                    and swa_slot is not None
                    and kv_state_flat is not None
                    and score_state_flat is not None
                )
            )
        ):
            group = self.caches_info[layer_id][mapping.ori_cache_idx].group_id if ratio == 2 else -1
            metadata_key = (ratio, group)
            decode = shared.decode_compressor_metadata.get(metadata_key)
            if decode is None:
                decode = self._decode_compressor_metadata(
                    ctx, ctx_lens, ratio, swa_table, swa_slot, layer_cache.swa.size(1) if ratio == 2 else 0
                )
                shared.decode_compressor_metadata[metadata_key] = decode
            latent, scores = layer.compress_kv_score(ctx.x)
            if ratio == 2:
                latent = self._pool_decode_c2(layer, ctx, decode, latent, scores, kv_state_flat, score_state_flat)
            else:
                latent = layer.compress_rmsnorm(latent)
            all_starts, all_slot_rows, all_slot_seqs, all_slot_mask = (
                decode.starts,
                decode.rows,
                decode.owners,
                decode.committed,
            )
        else:
            compressed = self._compress_chunks(
                layer, ctx, dsa, mapping, q_lens, ctx_lens, swa_slot, kv_state_flat, score_state_flat
            )
            latent, all_starts, all_slot_rows, all_slot_seqs, all_slot_mask = compressed

        index_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        key_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.cmp_cache_idx)

        def cache_slots(cache_idx: int, table: torch.Tensor | None, cache: torch.Tensor) -> torch.Tensor:
            if decode is not None and dsa.query_aligned_token_slots:
                slots = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, cache_idx)
                if slots is not None:
                    return slots
            return _token_cache_slots(table, cache, all_slot_rows, all_slot_seqs, all_slot_mask)

        if latent is not None:
            if layer.indexer is not None and layer.indexer.owns_k:
                if (
                    self.uses_a3_cache_precision
                    and self.index_head_dim == 128
                    and ctx.cos_sin_cache.size(-1) == 64
                    and latent.dtype == torch.bfloat16
                    and latent.size(0) <= 65536
                ):
                    cos_sin = ctx.cos_sin_cache.index_select(0, all_starts).float()
                    index_k = layer.indexer_key(latent, cos_sin)
                else:
                    index_k = layer.indexer_key(latent)
                    index_k = apply_interleaved_rope(index_k, all_starts, ctx.cos_sin_cache)
                index_scale = None
                if self.uses_a3_cache_precision:
                    quant_input = index_k.bfloat16() if index_k.dtype == torch.float32 else index_k
                    codes, scale = kernels.dynamic_quant(quant_input)
                    if layer_cache.indexer_scale is None:
                        # Reference scoring uses the dequantized model-dtype key.
                        index_k = codes.float() * scale.half().float().unsqueeze(-1)
                    else:
                        index_k = codes
                        index_scale = scale.half().unsqueeze(-1)
                else:
                    index_k = fp4_act_qdq_e8m0(index_k, 32)
                if layer_cache.index is not None:
                    index_slots = cache_slots(mapping.index_cache_idx, index_block_table, layer_cache.index)
                    self._write_cache_rows(layer_cache.index, index_slots, index_k, graph=decode is not None)
                    if index_scale is not None:
                        self._write_cache_rows(
                            layer_cache.indexer_scale, index_slots, index_scale, graph=decode is not None
                        )
            rotated = self._apply_rope(latent, all_starts, ctx.cos_sin_cache, dsa, shared)
            if not self.uses_a3_cache_precision:
                rotated = fp4_act_qdq_e4m3_scale(rotated, 16)
            if layer_cache.key is not None:
                key_slots = cache_slots(mapping.cmp_cache_idx, key_block_table, layer_cache.key)
                self._write_cache_rows(layer_cache.key, key_slots, rotated, graph=decode is not None)
        shared.key_cache = layer_cache.key
        shared.key_block_table = key_block_table
        shared.index_cache = layer_cache.index
        shared.index_scale = layer_cache.indexer_scale
        shared.index_block_table = index_block_table
        shared.compress_ratio = ratio

    @staticmethod
    def _decode_compressor_metadata(
        ctx: Csa2LayerContext,
        lengths: torch.Tensor,
        ratio: int,
        swa_table: torch.Tensor | None,
        swa_slots: torch.Tensor | None,
        block_size: int,
    ) -> _Csa2DecodeCompressorMetadata:
        owners = torch.arange(lengths.numel(), device=lengths.device)
        positions = lengths.to(torch.int64) - 1
        live = lengths > 0
        if ratio == 1:
            return _Csa2DecodeCompressorMetadata(positions, ctx.positions.to(torch.int64), owners, live)

        rows = positions // 2
        starts = (rows * 2).clamp(0, ctx.cos_sin_cache.size(0) - 1)
        committed = live & (positions % 2 == 1)
        park = live & ~committed
        park_slots = torch.where(park, swa_slots, -1)

        previous_positions = (positions - 1).clamp_min(0)
        columns = (previous_positions // block_size) % swa_table.size(1)
        blocks = swa_table.gather(1, columns.unsqueeze(1)).squeeze(1).to(torch.int64).clamp_min(0)
        previous_slots = blocks * block_size + previous_positions % block_size

        # Fixed verification and packed verification both order consecutive
        # tokens of a request together. An equal physical slot alone is
        # insufficient when the request crosses the SWA ring boundary.
        preceding = (owners - 1).clamp_min(0)
        previous_in_batch = (
            committed
            & (owners > 0)
            & (positions.index_select(0, preceding) == previous_positions)
            & (park_slots.index_select(0, preceding) == previous_slots)
        )
        # Nonnegative entries address the state ring. The previous query's
        # index (owners - 1) is stored as -(index + 1) when it is in this step.
        previous_rows = torch.where(previous_in_batch, -owners, previous_slots).to(torch.int32)
        return _Csa2DecodeCompressorMetadata(rows, starts, owners, committed, park_slots, previous_rows)

    def _pool_decode_c2(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        decode: _Csa2DecodeCompressorMetadata,
        kv: torch.Tensor,
        scores: torch.Tensor,
        kv_state: torch.Tensor,
        score_state: torch.Tensor,
    ) -> torch.Tensor:
        if (
            self.uses_a3_cache_precision
            and self.head_dim == 512
            and ctx.x.dtype == torch.bfloat16
            and kv.size(0) <= 65536
            and kv_state.size(0) <= 1 << 21
        ):
            output = torch.ops.xllm_ops.v41_c2_decode_pool_norm(
                kv, scores, kv_state, score_state, decode.previous_rows, layer.cmp_norm.weight, layer.cmp_norm.eps
            )
        else:
            previous = decode.previous_rows.to(torch.int64)
            from_batch = (previous < 0).unsqueeze(1)
            preceding = (-previous - 1).clamp_min(0)
            previous_slots = previous.clamp_min(0)
            previous_kv = torch.where(
                from_batch, kv.index_select(0, preceding).to(kv_state.dtype), kv_state.index_select(0, previous_slots)
            )
            previous_scores = torch.where(
                from_batch,
                scores.index_select(0, preceding).to(score_state.dtype),
                score_state.index_select(0, previous_slots),
            )
            members = torch.stack((previous_kv, kv), dim=1)
            member_scores = torch.stack((previous_scores, scores), dim=1)
            output = self._pool_c2(layer, ctx, members, member_scores)
        # Every read of the old partial group must finish before publishing
        # this forward's new state into the ring.
        self._write_cache_rows(kv_state, decode.park_slots, kv, graph=True)
        self._write_cache_rows(score_state, decode.park_slots, scores, graph=True)
        return output

    def _pool_c2(self, layer: Any, ctx: Csa2LayerContext, members: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        if (
            self.uses_a3_cache_precision
            and self.head_dim == 512
            and ctx.x.dtype == torch.bfloat16
            and members.size(0) <= 65536
        ):
            return torch.ops.xllm_ops.v41_c2_pool_norm(members, scores, layer.cmp_norm.weight, layer.cmp_norm.eps)
        pooled = (members * scores.softmax(dim=1)).sum(dim=1)
        return layer.compress_rmsnorm(pooled.to(ctx.x.dtype))

    def _compress_chunks(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata,
        mapping: _CompressedAttentionCacheMapping,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        swa_slot: torch.Tensor | None,
        kv_state_flat: torch.Tensor | None,
        score_state_flat: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        ratio = layer.mode.compress_ratio
        layer_id = layer.layer_id
        latent_rows: list[torch.Tensor] = []
        c2_members: list[torch.Tensor] = []
        c2_scores: list[torch.Tensor] = []
        c2_projections = layer.compress_kv_score(ctx.x) if ratio == 2 and not dsa.compressed_write_floors else None
        group_starts: list[torch.Tensor] = []
        slot_rows: list[torch.Tensor] = []
        slot_seqs: list[torch.Tensor] = []
        slot_masks: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            start = token_offset
            token_offset += q_len
            if dsa.compressed_write_floors:
                # Shared compressed groups are already complete at this
                # boundary. Project and publish only the uncached suffix.
                query_start = dsa.seq_lens_kv_host[seq] - q_len
                skip = min(q_len, max(0, dsa.compressed_write_floors[seq] - query_start))
                start += skip
                q_len -= skip
                if q_len == 0:
                    continue
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            # Padded graph rows (ctx <= 0) stay fully inert: no parking, no
            # commit, so a garbage block-table entry can never be written.
            seq_valid = ctx_len > 0
            x_seq = ctx.x[start:token_offset]
            rows = torch.arange(q_len, device=ctx.x.device)
            # An aligned chunk provides both members of every C2 group. Other
            # chunks can straddle a boundary and need the parked state.
            complete_pairs = (
                ratio == 2
                and not dsa.is_acl_graph
                and not dsa.compressed_write_floors
                and q_len > 0
                and q_len % 2 == 0
                and len(dsa.seq_lens_kv_host) == len(q_lens)
                and dsa.seq_lens_kv_host[seq] >= q_len
                and dsa.seq_lens_kv_host[seq] % 2 == 0
            )
            capacity = q_len // ratio if complete_pairs else q_len // ratio + 1
            g = torch.arange(capacity, device=ctx.x.device)
            group_idx = torch.div(prev_ctx, ratio, rounding_mode="floor") + g
            commit_mask = g < (
                torch.div(ctx_len, ratio, rounding_mode="floor") - torch.div(prev_ctx, ratio, rounding_mode="floor")
            )
            commit_mask = commit_mask & seq_valid
            if ratio == 1:
                kv_seq, _ = layer.compress_kv_score(x_seq)
                latent_rows.append(layer.compress_rmsnorm(kv_seq))
                group_starts.append(ctx.positions[start:token_offset].to(torch.int64))
                rows_segment = group_idx[:q_len]
                mask_segment = commit_mask[:q_len]
            else:
                if c2_projections is None:
                    kv_seq, score_seq = layer.compress_kv_score(x_seq)
                else:
                    c2_kv, c2_wgate = c2_projections
                    kv_seq = c2_kv[start:token_offset]
                    score_seq = c2_wgate[start:token_offset]
                # Pool the groups committed by this forward; a member inside
                # the chunk reads the fresh projection, one before the chunk
                # start reads its parked copy from the state ring.
                if complete_pairs:
                    member_kv = kv_seq.view(-1, ratio, kv_seq.size(-1))
                    member_score = score_seq.view(-1, ratio, score_seq.size(-1))
                else:
                    member_pos = group_idx.unsqueeze(-1) * ratio + torch.arange(ratio, device=ctx.x.device)
                    in_chunk = (member_pos >= prev_ctx).unsqueeze(-1)
                    member_row = (member_pos - prev_ctx).clamp(0, q_len - 1)
                    parked_slot = self._swa_position_slots(
                        dsa, layer_id, mapping.ori_cache_idx, seq, member_pos.clamp_min(0)
                    )
                    if kv_state_flat is not None and score_state_flat is not None:
                        member_kv = torch.where(in_chunk, kv_seq[member_row], kv_state_flat[parked_slot])
                        member_score = torch.where(in_chunk, score_seq[member_row], score_state_flat[parked_slot])
                    else:
                        member_kv = kv_seq[member_row]
                        member_score = score_seq[member_row]
                # Read old partial-group members before a long chunk wraps
                # its new state onto the same ring slots.
                pos = prev_ctx + rows
                park_mask = ((pos + 1) % ratio != 0) & seq_valid
                if (
                    swa_slot is not None
                    and swa_slot.numel() > 0
                    and kv_state_flat is not None
                    and score_state_flat is not None
                ):
                    slot_seq = swa_slot[start:token_offset]
                    park_slot = torch.where(park_mask, slot_seq, -1)
                    park_kv, park_score = kv_seq, score_seq
                    if not dsa.is_acl_graph:
                        park_slot, park_rows = last_ring_writes(park_slot, rows, slot_capacity=kv_state_flat.size(0))
                        park_kv, park_score = kv_seq[park_rows], score_seq[park_rows]
                    _scatter_by_slot(kv_state_flat, park_slot, park_kv)
                    _scatter_by_slot(score_state_flat, park_slot, park_score)
                c2_members.append(member_kv)
                c2_scores.append(member_score)
                # Garbage rows (beyond this sequence's commit count) keep a
                # valid non-negative rope position; the slot mask keeps them
                # unwritten. The lower clamp also catches fully padded rows
                # (ctx_len <= 0 -> group_idx = -1), which stay inert via the
                # mask but still flow through apply_interleaved_rope.
                group_starts.append((group_idx * ratio).clamp(0, ctx.cos_sin_cache.size(0) - 1))
                rows_segment = group_idx
                mask_segment = commit_mask
            slot_rows.append(rows_segment)
            slot_seqs.append(torch.full_like(rows_segment, seq))
            slot_masks.append(mask_segment)

        if c2_members:
            latent_rows.append(self._pool_c2(layer, ctx, torch.cat(c2_members, dim=0), torch.cat(c2_scores, dim=0)))
        if not latent_rows:
            return None, None, None, None, None
        return (
            torch.cat(latent_rows, dim=0),
            torch.cat(group_starts, dim=0),
            torch.cat(slot_rows, dim=0),
            torch.cat(slot_seqs, dim=0),
            torch.cat(slot_masks, dim=0),
        )

    def _run_indexer(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata | None,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor | None:
        shared.topk_idxs = None
        shared.compressed_indices = None
        layer_id = layer.layer_id
        ratio = shared.compress_ratio
        if ratio <= 0 or shared.index_cache is None:
            raise RuntimeError(
                f"DeepSeek-V4.1 indexer for layer {layer_id} has no shared index cache; "
                "its group kv source must run first"
            )
        if shared.index_block_table is None:
            return None
        index_cache = shared.index_cache
        block_table = shared.index_block_table
        block_size = index_cache.size(1)
        if ctx.qr_quantized is None:
            assert ctx.qr is not None
            q_idx = layer.indexer_query(ctx.qr)
        else:
            q_idx = layer.indexer_query(ctx.qr, ctx.qr_quantized)
        q_idx = self._apply_rope(q_idx, ctx.positions, ctx.cos_sin_cache, dsa, shared)
        weights = layer.indexer_weights(ctx.x)
        if shared.index_scale is not None and dsa is not None and (dsa.is_acl_graph or dsa.max_seq_len > 0):
            return self._run_device_length_indexer(layer_id, q_idx, weights, dsa, shared)
        capacity = self._committed_capacity(dsa, ratio, ctx_lens)
        if self.uses_a3_cache_precision:
            q_idx = _a3_indexer_query_qdq(q_idx)
            weights = weights.half()
        else:
            q_idx = fp4_act_qdq_e8m0(q_idx, 32)

        candidates_per_seq: list[torch.Tensor | None] | None = None
        if layer_id == self.candidate_source_layer_id:
            candidates_per_seq = [None] * len(q_lens)
        rows_per_seq: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            q_seq = q_idx[token_offset : token_offset + q_len]
            w_seq = weights[token_offset : token_offset + q_len]
            positions_seq = ctx.positions[token_offset : token_offset + q_len].to(torch.int64)
            ctx_len = ctx_lens[seq]
            committed = torch.div(ctx_len, ratio, rounding_mode="floor")
            if capacity <= 0:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
                token_offset += q_len
                continue
            index_k = self._gather_cache_rows(index_cache, block_table, seq, capacity, block_size)
            if shared.index_scale is not None:
                scale = self._gather_cache_rows(
                    shared.index_scale.unsqueeze(-1), block_table, seq, capacity, block_size
                )
                index_k = (index_k.float() * scale.float()).to(self.dtype)
            # A compressed group is visible to query p once p has passed its
            # last token: visible(p) = (p + 1) // ratio;
            # rows past this sequence's committed count (or past the graph's
            # capacity bound) are padding and can never be picked.
            visible = torch.div(positions_seq + 1, ratio, rounding_mode="floor")
            t_range = torch.arange(capacity, device=q_idx.device)
            # The compressed picks address the concatenated
            # [window_kv ; compress_kv] tensor. The window is a fixed-capacity
            # [ring candidates ; chunk] block, so the compressed region starts
            # at a constant column and no ctx-length term remains.
            offset = self.window_size + q_len
            row_tiles: list[torch.Tensor] = []
            pool_tiles: list[torch.Tensor] = []
            for tile_start in range(0, q_len, _INDEXER_QUERY_TILE):
                tile_end = min(tile_start + _INDEXER_QUERY_TILE, q_len)
                score = _indexer_score(q_seq[tile_start:tile_end], index_k, w_seq[tile_start:tile_end])
                tile_visible = visible[tile_start:tile_end]
                score.masked_fill_(t_range >= tile_visible.unsqueeze(-1), float("-inf"))
                score.masked_fill_(t_range >= committed.unsqueeze(-1), float("-inf"))
                if layer_id == self.candidate_source_layer_id:
                    pool_tiles.append(
                        _select_candidate_blocks(
                            score,
                            tile_visible,
                            self.candidate_topk_blocks,
                            self.candidate_block_size,
                        )
                    )
                elif 0 <= self.candidate_source_layer_id < layer_id:
                    if shared.candidates is None or seq >= len(shared.candidates):
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} requires the candidate pool "
                            f"from layer {self.candidate_source_layer_id}"
                        )
                    pool = shared.candidates[seq]
                    if pool is None:
                        raise RuntimeError(
                            f"DeepSeek-V4.1 reindex layer {layer_id} has no candidate pool for sequence {seq}"
                        )
                    score.masked_fill_(~pool[tile_start:tile_end], float("-inf"))
                row_tiles.append(_indexer_topk(score, tile_visible, self.index_topk, offset))
            if pool_tiles and candidates_per_seq is not None:
                candidates_per_seq[seq] = torch.cat(pool_tiles, dim=0)
            if row_tiles:
                rows_per_seq.append(row_tiles[0] if len(row_tiles) == 1 else torch.cat(row_tiles, dim=0))
            else:
                rows_per_seq.append(torch.full((q_len, 0), -1, dtype=torch.int32, device=q_idx.device))
            token_offset += q_len

        seq_count = len(q_lens)
        s_max = max(q_lens) if q_lens else 0
        k_width = min(self.index_topk, capacity) if capacity > 0 else 0
        topk = torch.full((seq_count, s_max, k_width), -1, dtype=torch.int32, device=q_idx.device)
        for seq, rows in enumerate(rows_per_seq):
            if rows.numel() > 0:
                topk[seq, : rows.size(0), : rows.size(1)] = rows
        shared.topk_idxs = topk
        if candidates_per_seq is not None:
            shared.candidates = candidates_per_seq
        return topk

    def _run_device_length_indexer(
        self,
        layer_id: int,
        query: torch.Tensor,
        weights: torch.Tensor,
        dsa: DsaMetadata,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        ratio = shared.compress_ratio
        query_offsets = dsa.actual_seq_lengths_query
        max_query_len = max(dsa.seq_lens_q_host)
        key, key_scale, block_table = shared.index_cache, shared.index_scale, shared.index_block_table
        metadata = shared.indexer_metadata.get(ratio)
        if metadata is None:
            lengths = dsa.seq_lens
            committed = torch.div(lengths, ratio, rounding_mode="floor")
            residual = lengths.remainder(ratio) if ratio != 1 else None
            tiling = torch.ops.xllm_ops.quant_lightning_indexer_v2_metadata(
                query_offsets,
                committed,
                residual,
                self.index_n_heads,
                self.index_head_dim,
                self.index_topk,
                max_query_len,
                block_table.size(1) * key.size(1),
                ratio,
            )
            metadata = _Csa2IndexerMetadata(committed, residual, tiling)
            shared.indexer_metadata[ratio] = metadata
        if (
            self.uses_a3_cache_precision
            and query.dtype == torch.bfloat16
            and query.is_contiguous()
            and query.size(0) > 0
            and (not dsa.is_acl_graph or query.size(0) <= 4096)
            and query.size(1) in (32, 64)
            and query.size(2) == 128
        ):
            if query.size(0) <= 4096:
                query_codes, query_scale = torch.ops.xllm_ops.v41_index_query_quant(query)
            else:
                # The kernel addresses at most 4096 token rows per launch.
                quantized = [torch.ops.xllm_ops.v41_index_query_quant(part) for part in query.split(4096)]
                query_codes = torch.cat([codes for codes, _ in quantized])
                query_scale = torch.cat([scales for _, scales in quantized])
        else:
            query_codes, query_scale = _a3_quantize_indexer_query(query)
        is_source = layer_id == self.candidate_source_layer_id
        is_consumer = 0 <= self.candidate_source_layer_id < layer_id
        if is_consumer and shared.candidate_blocks is None:
            raise RuntimeError(f"DeepSeek-V4.1 reindex layer {layer_id} requires the graph candidate blocks")
        selected, candidates = torch.ops.xllm_ops.quant_lightning_indexer_v2(
            query_codes,
            key,
            weights.half(),
            query_scale,
            key_scale,
            query_offsets,
            metadata.committed,
            metadata.residual,
            block_table,
            metadata.tiling,
            shared.candidate_blocks if is_consumer else None,
            self.index_topk,
            max_query_len,
            ratio,
            1 if is_source else 2 if is_consumer else 3,
            self.candidate_topk_blocks,
            self.candidate_block_size,
        )
        if is_source:
            shared.candidate_blocks = candidates
        # Native MLA uses the same compressed-cache coordinates as the QLI.
        shared.compressed_indices = selected.clamp_min_(-1)
        return shared.compressed_indices

    def _reference_topk_indices(self, selected: torch.Tensor, q_lens: list[int]) -> torch.Tensor:
        if max(q_lens) == 1:
            return torch.where(selected >= 0, selected + self.window_size + 1, -1)
        rows = selected.squeeze(1).split(q_lens)
        picks = [
            torch.where(indices >= 0, indices + self.window_size + q_len, -1)
            for indices, q_len in zip(rows, q_lens, strict=True)
        ]
        return torch.nn.utils.rnn.pad_sequence(picks, batch_first=True, padding_value=-1)

    def _csa2_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        layer: Any,
        dsa: DsaMetadata,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        topk: torch.Tensor | None,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor:
        mode = layer.mode
        layer_id = layer.layer_id
        ratio = mode.compress_ratio
        win = self.window_size
        head_dim = q.size(-1)
        # An absent sink must add zero to the normalizer, not exp(0).
        sinks = layer.attn_sink.to(torch.float32) if getattr(layer, "attn_sink_loaded", False) else None
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, ratio)
        swa_flat = layer_cache.swa.view(-1, head_dim) if layer_cache.swa is not None else None
        swa_bt = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.ori_cache_idx)
        swa_bs = layer_cache.swa.size(1) if layer_cache.swa is not None else win

        capacity = 0
        if q_lens and ratio > 0 and topk is not None and shared.key_cache is not None:
            capacity = self._committed_capacity(dsa, ratio, ctx_lens)
        if capacity > 0 and dsa.is_acl_graph and all(n == 1 for n in q_lens):
            return self._compressed_decode_attention(
                q, kv, ctx_lens, topk[:, 0], layer_cache.swa, swa_bt, shared, ratio, capacity, sinks
            )
        compressed_rows = torch.arange(capacity, device=q.device) if capacity > 0 else None

        history_floors = dsa.swa_history_floors
        out = torch.empty_like(q)
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            q_seq = q[token_offset : token_offset + q_len]
            kv_seq = kv[token_offset : token_offset + q_len]
            ring_pos = (prev_ctx - win) + torch.arange(win, device=q.device)
            if swa_flat is not None and swa_bt is not None:
                ring_rows = self._gather_swa_positions(swa_flat, swa_bt, seq, ring_pos.clamp_min(0), swa_bs)
            else:
                ring_rows = q.new_zeros((win, kv_seq.size(-1)))
            # Unwritten rows can contain NaN/Inf. Score masking alone still
            # lets zero probabilities multiply those values in the reduction.
            # The gather owns this storage; masking does not change the cache.
            history_floor = history_floors[seq] if history_floors else 0
            ring_rows.masked_fill_((ring_pos < history_floor).unsqueeze(-1), 0)
            kv_seq = torch.where(ctx_len > 0, kv_seq, 0)
            n_w = ring_rows.size(0) + kv_seq.size(0)
            positions_seq = prev_ctx + torch.arange(q_len, device=q.device)
            col_pos = torch.cat((ring_pos, positions_seq), dim=0)
            cols = torch.arange(n_w, device=q.device)
            window_idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)
            if history_floor:
                window_idxs = window_idxs.masked_fill(col_pos < history_floor, -1)
            if capacity > 0:
                shared_key_cache = shared.key_cache
                shared_key_table = shared.key_block_table
                if shared_key_cache is None or shared_key_table is None:
                    raise RuntimeError(
                        f"DeepSeek-V4.1 layer {layer_id} has no shared key cache for its "
                        f"compressed rows; its group kv source must run first"
                    )
                picks = topk[seq][:q_len]
                committed = torch.div(ctx_len, ratio, rounding_mode="floor")
                compress_kv = self._gather_cache_rows(
                    shared_key_cache,
                    shared_key_table,
                    seq,
                    capacity,
                    shared_key_cache.size(1),
                )
                valid_rows = compressed_rows < committed
                compress_kv.masked_fill_(~valid_rows.unsqueeze(-1), 0)
                kv_cat = torch.cat((ring_rows, kv_seq, compress_kv), dim=0)
                idxs = torch.cat((window_idxs, picks), dim=-1)
            else:
                kv_cat = torch.cat((ring_rows, kv_seq), dim=0)
                idxs = window_idxs
            kv_cat_f = kv_cat.to(torch.float32)
            row_mask = _picked_row_mask(idxs, kv_cat.size(0))

            for tile_start in range(0, q_len, _ATTENTION_QUERY_TILE):
                tile_end = min(tile_start + _ATTENTION_QUERY_TILE, q_len)
                scores = torch.einsum("shd,nd->shn", q_seq[tile_start:tile_end].to(torch.float32), kv_cat_f)
                scores = scores * self.scale
                scores = scores.masked_fill(~row_mask[tile_start:tile_end].unsqueeze(1), float("-inf"))
                row_max = scores.amax(dim=-1).clamp_min(_NEG_INF_FLOOR)
                probs = torch.exp(scores - row_max.unsqueeze(-1))
                sum_exp = probs.sum(dim=-1)
                if sinks is not None:
                    sum_exp = sum_exp + torch.exp(sinks - row_max)
                tile_out = torch.einsum("shn,nd->shd", probs, kv_cat_f)
                # Empty-DP queries have no visible values and no required sink.
                tile_out = tile_out / sum_exp.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)
                out[token_offset + tile_start : token_offset + tile_end] = tile_out.to(q.dtype)
            token_offset += q_len
        return out

    def _compressed_decode_attention(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        ctx_lens: torch.Tensor,
        topk: torch.Tensor,
        swa_cache: torch.Tensor | None,
        swa_table: torch.Tensor | None,
        shared: CSA2SharedRuntime,
        ratio: int,
        capacity: int,
        sinks: torch.Tensor | None,
    ) -> torch.Tensor:
        key_cache, key_table = shared.key_cache, shared.key_block_table
        if key_cache is None or key_table is None:
            raise RuntimeError("DeepSeek-V4.1 compressed decode needs the source key cache and page table")
        batch, _, head_dim = q.shape
        owners = torch.arange(batch, device=q.device).unsqueeze(1)
        lengths = ctx_lens.reshape(-1, 1)
        # The current query occupies one position of the causal SWA window.
        # Its oldest cached row is outside the window and need not be gathered.
        ring_pos = lengths - 1 + torch.arange(1 - self.window_size, 0, device=q.device)
        live_ring = ring_pos >= 0
        if swa_cache is None or swa_table is None:
            ring = kv.new_zeros((batch, self.window_size - 1, head_dim))
        else:
            slots = _cache_slots(swa_table, owners, ring_pos.clamp_min(0), swa_cache.size(1))
            ring = swa_cache.view(-1, head_dim).index_select(0, slots.flatten()).view(batch, -1, head_dim)

        # The indexer returns distinct compressed rows after the SWA span in
        # the reference layout (window + query). Invalid rows use a safe slot.
        selected = topk.to(torch.int64) - (self.window_size + 1)
        committed = torch.div(lengths, ratio, rounding_mode="floor")
        live_keys = (selected >= 0) & (selected < capacity) & (selected < committed)
        positions = torch.where(live_keys, selected, 0)
        slots = _cache_slots(key_table, owners, positions, key_cache.size(1))
        compressed = key_cache.view(-1, head_dim).index_select(0, slots.flatten()).view(batch, -1, head_dim)
        valid = torch.cat((live_ring, lengths > 0, live_keys), dim=1)
        values = torch.cat((ring, kv.unsqueeze(1), compressed), dim=1).float()
        # Unwritten cache rows may contain NaNs, even when scores are masked.
        values.masked_fill_(~valid.unsqueeze(-1), 0)
        scores = torch.bmm(q.float(), values.transpose(1, 2)) * self.scale
        scores.masked_fill_(~valid.unsqueeze(1), float("-inf"))
        row_max = scores.amax(dim=-1).clamp_min(_NEG_INF_FLOOR)
        probs = torch.exp(scores - row_max.unsqueeze(-1))
        normalizer = probs.sum(dim=-1)
        if sinks is not None:
            normalizer = normalizer + torch.exp(sinks - row_max)
        output = torch.bmm(probs, values)
        return (output / normalizer.clamp_min(torch.finfo(torch.float32).tiny).unsqueeze(-1)).to(q.dtype)

    def _committed_capacity(
        self,
        dsa: DsaMetadata | None,
        ratio: int,
        ctx_lens: torch.Tensor,
    ) -> int:
        if ratio <= 0:
            return 0
        if bool(getattr(dsa, "is_acl_graph", False)):
            bucketed_capacity = int(getattr(dsa, "graph_token_capacity", 0) or 0)
            if bucketed_capacity > 0:
                return max(bucketed_capacity // ratio, 1)
            if self.graph_token_capacity_granularity > 0:
                return max(self.graph_token_capacity_granularity // ratio, 1)
            raise RuntimeError(
                "DeepSeek-V4.1 ACL graph capture needs a static committed-row capacity, but the "
                "serving entry did not derive one (XLLM_V41_GRAPH_CAPACITY_GRANULARITY is unset and "
                "no max_model_len was provided). The engine should have derived a default at "
                "construction; a missing value means the deployment entry point passed no "
                "max_model_len."
            )
        if dsa is not None and dsa.seq_lens_kv_host:
            # Eager metadata owns the exact host lengths from the same forward.
            # Reading lengths back would synchronize the NPU at every layer.
            return max(max(dsa.seq_lens_kv_host) // ratio, 0)
        if ctx_lens.numel() == 0:
            return 0
        return max(int(torch.div(ctx_lens.max(), ratio, rounding_mode="floor").item()), 0)

    def _gather_cache_rows(
        self,
        cache: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        count: int,
        block_size: int,
    ) -> torch.Tensor:
        rows = torch.arange(count, device=cache.device)
        slots = _cache_slots(block_table, seq, rows, block_size)
        return cache.view(-1, cache.size(-1)).index_select(0, slots)

    def _gather_swa_positions(
        self,
        swa_flat: torch.Tensor,
        block_table: torch.Tensor,
        seq: int,
        positions: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        return swa_flat.index_select(0, _cache_slots(block_table, seq, positions, block_size))

    def _swa_position_slots(
        self,
        dsa: DsaMetadata,
        layer_id: int,
        swa_cache_idx: int,
        seq: int,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, swa_cache_idx)
        swa = self._kv_caches[layer_id].swa
        block_size = swa.size(1) if swa is not None else self.window_size
        if block_table is None:
            return positions % block_size
        return _cache_slots(block_table, seq, positions, block_size)

    def _shared_runtime(self) -> CSA2SharedRuntime:
        if self._v41_shared is None:
            self._v41_shared = CSA2SharedRuntime()
        return self._v41_shared
