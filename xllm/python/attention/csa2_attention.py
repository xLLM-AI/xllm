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
from dataclasses import dataclass
from typing import Any

import torch

from scripts.logger import logger
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
    build_cache_specs_v41,
    last_ring_writes,
)
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


@dataclass
class _SwaGroupMetadata:
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    block_size: int
    cache: torch.Tensor


@dataclass
class Csa2LayerContext:
    x: torch.Tensor
    qr: torch.Tensor
    positions: torch.Tensor
    cos_sin_cache: torch.Tensor


@dataclass
class CSA2SharedRuntime:
    key_cache: torch.Tensor | None = None
    key_block_table: torch.Tensor | None = None
    index_cache: torch.Tensor | None = None
    index_block_table: torch.Tensor | None = None
    compress_ratio: int = 0
    topk_idxs: torch.Tensor | None = None
    candidates: list[torch.Tensor | None] | None = None


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
        acl_graph_enabled: bool = False,
    ) -> None:
        self.caches_info, self.group_infos = build_cache_specs_v41(
            compress_ratios,
            kv_source_layer_ids,
            window_size,
            n_layers,
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

        self.index_source_layer_ids = sorted({int(source) for source in index_source_layer_ids})
        self.candidate_source_layer_id = int(candidate_source_layer_id)
        self.candidate_topk_blocks = int(candidate_topk_blocks)
        self.candidate_block_size = int(candidate_block_size)
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

        if acl_graph_enabled and not self.moe_graph_capturable:
            logger.warning(
                f"DeepSeek-V4.1 quant_mode={quant_mode} serves the torch MoE, which no ACL graph "
                "can capture; decode-graph admission is disabled and decode stays eager"
            )
        if acl_graph_enabled and self.moe_graph_capturable and self.graph_token_capacity_granularity <= 0:
            raise RuntimeError(
                "DeepSeek-V4.1 ACL-graph serving needs a static committed-row capacity, but "
                "XLLM_V41_GRAPH_CAPACITY_GRANULARITY is unset/empty and max_model_len was not "
                "provided to derive one. Set the env var or provide max_position_embeddings, or "
                "disable graph execution."
            )

        self._kv_caches: list[Any] = []
        self._metadata: Any | None = None
        self._v41_shared: CSA2SharedRuntime | None = None

    def reset_forward(self, metadata: Any | None = None) -> None:
        super().reset_forward(metadata)
        self._v41_shared = None

    def _build_dsa_metadata_for_forward(self, metadata: Any) -> DsaMetadata:
        dsa = super()._build_dsa_metadata_for_forward(metadata)
        if bool(getattr(metadata, "is_dummy", False)):
            # V4's synthetic tiling context is not initialized history for tensor attention.
            dsa.seq_lens.zero_()
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
        mode = layer.mode
        layer_id = layer.layer_id
        layer_cache = self._kv_caches[layer_id]
        mapping = self._resolve_cache_mapping(layer_id, mode.compress_ratio)
        ctx = layer._csa2_ctx
        if ctx is None:
            raise RuntimeError(f"DeepSeek-V4.1 layer {layer_id} ran without its CSA2 context")
        if dsa.is_acl_graph:
            # Decode-graph contract: one token row per sequence. The host list
            # is a capture-stable constant, so never sync-read the device
            # tensor inside the captured stream (aclrtMemcpy on a captured
            # stream fails with 107030).
            q_lens = [1] * dsa.seq_lens_q.numel()
        else:
            q_lens = dsa.seq_lens_q_host
        ctx_lens = dsa.seq_lens

        shared = self._shared_runtime()
        if mode.is_full:
            self._run_compressor(layer, ctx, dsa, layer_cache, mapping, q_lens, ctx_lens, shared)
        if mode.is_full or mode.is_reindex:
            topk = self._run_indexer(layer, ctx, dsa, q_lens, ctx_lens, shared)
        elif mode.is_reuse:
            topk = shared.topk_idxs
            if topk is None and dsa.block_tables:
                raise RuntimeError(
                    f"DeepSeek-V4.1 reuse layer {layer_id} has no shared top-k; its group source must run first"
                )
        else:
            topk = None
        # Current-chunk KV is read directly; preserve the ring history until
        # all queries have consumed it, even when the chunk wraps the ring.
        output = self._csa2_attention(q, kv, layer, dsa, q_lens, ctx_lens, topk, shared)
        swa_slot = _get_layer_cache_tensor(dsa.slot_mappings, layer_id, mapping.ori_cache_idx)
        if layer_cache.swa is not None and swa_slot is not None and swa_slot.numel() > 0:
            if not dsa.is_acl_graph:
                swa_slot, kv = last_ring_writes(swa_slot, kv, slot_capacity=layer_cache.swa.numel() // kv.size(-1))
            _scatter_by_slot(layer_cache.swa, swa_slot, kv)
        return output

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

        latent_rows: list[torch.Tensor] = []
        group_starts: list[torch.Tensor] = []
        slot_rows: list[torch.Tensor] = []
        slot_seqs: list[torch.Tensor] = []
        slot_masks: list[torch.Tensor] = []
        token_offset = 0
        for seq, q_len in enumerate(q_lens):
            ctx_len = ctx_lens[seq]
            prev_ctx = ctx_len - q_len
            # Padded graph rows (ctx <= 0) stay fully inert: no parking, no
            # commit, so a garbage block-table entry can never be written.
            seq_valid = ctx_len > 0
            x_seq = ctx.x[token_offset : token_offset + q_len]
            rows = torch.arange(q_len, device=ctx.x.device)
            # Fixed commit capacity: a chunk of q_len tokens completes at most
            # q_len // ratio + 1 groups (the group straddling each boundary).
            capacity = q_len // ratio + 1
            g = torch.arange(capacity, device=ctx.x.device)
            group_idx = torch.div(prev_ctx, ratio, rounding_mode="floor") + g
            commit_mask = g < (
                torch.div(ctx_len, ratio, rounding_mode="floor") - torch.div(prev_ctx, ratio, rounding_mode="floor")
            )
            commit_mask = commit_mask & seq_valid
            if ratio == 1:
                kv_seq, _ = layer.compress_kv_score(x_seq)
                latent_rows.append(layer.compress_rmsnorm(kv_seq))
                group_starts.append(ctx.positions[token_offset : token_offset + q_len].to(torch.int64))
                rows_segment = group_idx[:q_len]
                mask_segment = commit_mask[:q_len]
            else:
                kv_seq, score_seq = layer.compress_kv_score(x_seq)
                # Pool the groups committed by this forward; a member inside
                # the chunk reads the fresh projection, one before the chunk
                # start reads its parked copy from the state ring.
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
                    slot_seq = swa_slot[token_offset : token_offset + q_len]
                    park_slot = torch.where(park_mask, slot_seq, -1)
                    park_kv, park_score = kv_seq, score_seq
                    if not dsa.is_acl_graph:
                        park_slot, park_rows = last_ring_writes(park_slot, rows, slot_capacity=kv_state_flat.size(0))
                        park_kv, park_score = kv_seq[park_rows], score_seq[park_rows]
                    _scatter_by_slot(kv_state_flat, park_slot, park_kv)
                    _scatter_by_slot(score_state_flat, park_slot, park_score)
                gate = member_score.softmax(dim=1)
                pooled = (member_kv * gate).sum(dim=1)
                # RMSNorm AFTER pooling, BEFORE RoPE (model.py:485).
                latent_rows.append(layer.compress_rmsnorm(pooled.to(ctx.x.dtype)))
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
            token_offset += q_len

        index_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.index_cache_idx)
        key_block_table = _get_layer_cache_tensor(dsa.block_tables, layer_id, mapping.cmp_cache_idx)
        if latent_rows:
            latent = torch.cat(latent_rows, dim=0)
            all_starts = torch.cat(group_starts, dim=0)
            all_slot_rows = torch.cat(slot_rows, dim=0)
            all_slot_seqs = torch.cat(slot_seqs, dim=0)
            all_slot_mask = torch.cat(slot_masks, dim=0)
            if layer.indexer is not None and layer.indexer.owns_k:
                index_k = layer.indexer_key(latent)
                index_k = apply_interleaved_rope(index_k, all_starts, ctx.cos_sin_cache)
                index_k = fp4_act_qdq_e8m0(index_k, 32)
                if layer_cache.index is not None:
                    index_slots = _token_cache_slots(
                        index_block_table,
                        layer_cache.index,
                        all_slot_rows,
                        all_slot_seqs,
                        all_slot_mask,
                    )
                    _scatter_by_slot(layer_cache.index, index_slots, index_k)
            rotated = apply_interleaved_rope(latent, all_starts, ctx.cos_sin_cache)
            rotated = fp4_act_qdq_e4m3_scale(rotated, 16)
            if layer_cache.key is not None:
                key_slots = _token_cache_slots(
                    key_block_table,
                    layer_cache.key,
                    all_slot_rows,
                    all_slot_seqs,
                    all_slot_mask,
                )
                _scatter_by_slot(layer_cache.key, key_slots, rotated)
        shared.key_cache = layer_cache.key
        shared.key_block_table = key_block_table
        shared.index_cache = layer_cache.index
        shared.index_block_table = index_block_table
        shared.compress_ratio = ratio

    def _run_indexer(
        self,
        layer: Any,
        ctx: Csa2LayerContext,
        dsa: DsaMetadata | None,
        q_lens: list[int],
        ctx_lens: torch.Tensor,
        shared: CSA2SharedRuntime,
    ) -> torch.Tensor | None:
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
        capacity = self._committed_capacity(dsa, ratio, ctx_lens)

        q_idx = layer.indexer_query(ctx.qr)
        q_idx = apply_interleaved_rope(q_idx, ctx.positions, ctx.cos_sin_cache)
        q_idx = fp4_act_qdq_e8m0(q_idx, 32)
        weights = layer.indexer_weights(ctx.x)

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
            # A compressed group is visible to query p once p has passed its
            # last token: visible(p) = (p + 1) // ratio (model.py:562-567);
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
        compressed_rows = torch.arange(capacity, device=q.device) if capacity > 0 else None

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
            ring_rows.masked_fill_((ring_pos < 0).unsqueeze(-1), 0)
            kv_seq = torch.where(ctx_len > 0, kv_seq, 0)
            n_w = ring_rows.size(0) + kv_seq.size(0)
            positions_seq = prev_ctx + torch.arange(q_len, device=q.device)
            col_pos = torch.cat((ring_pos, positions_seq), dim=0)
            cols = torch.arange(n_w, device=q.device)
            window_idxs = _window_visible_idxs(positions_seq, col_pos, cols, win)

            compress_kv = None
            if capacity > 0:
                shared_key_cache = shared.key_cache
                shared_key_table = shared.key_block_table
                if shared_key_cache is None or shared_key_table is None:
                    raise RuntimeError(
                        f"DeepSeek-V4.1 layer {layer_id} has no shared key cache for its "
                        f"compressed rows; its group kv source must run first"
                    )
                compress_kv = self._gather_cache_rows(
                    shared_key_cache,
                    shared_key_table,
                    seq,
                    capacity,
                    shared_key_cache.size(1),
                )
                committed = torch.div(ctx_len, ratio, rounding_mode="floor")
                valid_rows = compressed_rows < committed
                compress_kv.masked_fill_(~valid_rows.unsqueeze(-1), 0)
            if compress_kv is not None:
                kv_cat = torch.cat((ring_rows, kv_seq, compress_kv), dim=0)
                idxs = torch.cat((window_idxs, topk[seq][:q_len]), dim=-1)
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
