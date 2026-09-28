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
"""Fast torch implementation of kPool pooling (bit-exact equivalent to the original get_pooled_states).

Uses ``F.embedding`` (a small [B, P, rate] index) to replace the flattened
``gather(1, flat_idx)`` (a large [B, P*rate*D] int64 index): the set of
selected elements is identical (safe already clamped), only saving the
construction of and random access into a 4.19M-element index tensor.
Microbenchmark 2.50ms -> 0.05ms (B=1, T=32768, D=128, rate=4, npu:1).
This module stays pure torch with no xllm dependency, so unit tests can load it directly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _gather_rows(src: torch.Tensor, safe_indices: torch.Tensor) -> torch.Tensor:
    """src [B, T, D] (may be a strided view), safe_indices [B, P, R] -> [B, P, R, D]."""
    if src.shape[0] == 1:
        return F.embedding(safe_indices[0], src[0]).unsqueeze(0)
    return torch.stack([F.embedding(safe_indices[b], src[b]) for b in range(src.shape[0])])


def pooled_states(packed_states: torch.Tensor, key_valid: torch.Tensor, ape: torch.Tensor, head_dim: int, rate: int):
    """Bit-exact identical to the full output of Glm5NextIndexer.get_pooled_states."""
    keys, gate_scores, _ = torch.split(packed_states, [head_dim, head_dim, 1], dim=-1)
    batch_size, total_len = keys.shape[:2]
    device = keys.device
    first_key = torch.where(
        key_valid.any(-1),
        key_valid.to(torch.int32).argmax(-1),
        torch.full((batch_size,), total_len, dtype=torch.long, device=device),
    )
    n_pools = (total_len + rate - 1) // rate
    pool_offsets = torch.arange(n_pools, device=device) * rate
    slot_offsets = torch.arange(rate, device=device)
    pool_indices = first_key[:, None, None] + pool_offsets[None, :, None] + slot_offsets[None, None, :]
    slot_in_range = pool_indices < total_len
    safe_indices = pool_indices.clamp(0, total_len - 1)
    grouped_keys = _gather_rows(keys, safe_indices)  # [B, P, R, D]
    grouped_gate_scores = _gather_rows(gate_scores, safe_indices)
    slot_valid = (
        key_valid.to(torch.uint8)
        .gather(1, safe_indices.reshape(batch_size, -1))
        .reshape(batch_size, n_pools, rate)
        .to(torch.bool)
        & slot_in_range
    )
    pool_valid = slot_valid.all(-1)
    logits = grouped_gate_scores.float() + ape.float()[None, None]
    logits = logits.masked_fill(~slot_valid[..., None], float("-inf"))
    weights = torch.nan_to_num(logits.softmax(2)).to(grouped_keys.dtype)
    pool_keys = (weights * grouped_keys).sum(2)
    pool_indices = pool_indices.masked_fill(~slot_valid, -1)
    return pool_keys, pool_indices, pool_valid


# ---------------------------------------------------------------------------
# Paged pool cache: write-time incremental compression + direct read.
#
# Physical blocks reuse the token block table: pool logical block L covers
# pools [L*bs/kpool, (L+1)*bs/kpool) = tokens [L*bs, (L+1)*bs) = token logical
# block L, so pool_slot(p) = bt[s, p//pool_bs] * pool_bs + p%pool_bs (where
# pool_bs = bs//index_kpool) — linear addressing without engine-side
# pool-granular allocation. The 257-wide token-granular index cache is kept
# untouched: compression inputs are read from it, which also solves
# chunked-prefill pools spanning chunk boundaries (vllm's dual-cache design).
# ---------------------------------------------------------------------------


def alloc_pool_cache(index_cache: torch.Tensor, index_kpool: int) -> torch.Tensor:
    """Per-DSA-layer pool-granular cache: ``[num_blocks, bs//index_kpool, 1, head_dim]`` bf16 all-zeros.

    ``index_kpool`` is the pool compression ratio (tokens per pool slot, from
    ``text_config.index_kpool``); it must divide ``block_size`` evenly so the
    paged addressing ``bt[p // pool_bs] * pool_bs + p % pool_bs`` stays aligned.
    """
    bs = index_cache.shape[1]
    assert index_kpool > 0, "index_kpool must be > 0"
    assert bs % index_kpool == 0, (
        f"block_size {bs} is not a multiple of index_kpool {index_kpool}, pool addressing breaks"
    )
    head_dim = (index_cache.shape[-1] - 1) // 2
    return torch.zeros(
        index_cache.shape[0],
        bs // index_kpool,
        1,
        head_dim,
        dtype=torch.bfloat16,
        device=index_cache.device,
    )


def compress_completed_pools(
    index_cache: torch.Tensor,
    pool_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    head_dim: int,
    rate: int,
) -> None:
    """Write the compressed k of pools completed this step into the pool cache.

    ``block_table`` is a **single-sequence** row ``[1, n_logical_blocks]``;
    ``positions`` are the absolute positions of the tokens written this step
    for that sequence. The math is fully order-consistent with
    ``pooled_states()`` (fp32 softmax + bf16 per-product round), and the
    written value == the per-step recomputed value of the old path
    (bit-exact). Graph-safe: no data-dependent shapes; unfinished pools
    retain their original value via the write mask.
    """
    bs = index_cache.shape[1]
    pool_bs = pool_cache.shape[1]
    width = 2 * head_dim + 1
    flat = index_cache.reshape(-1, width)
    pool_flat = pool_cache.reshape(-1, head_dim)
    n_tok = positions.shape[0]
    rate_off = torch.arange(rate, device=positions.device)
    # Each token is the last token of its pool: pool p = pos // rate
    pos = positions.reshape(-1, 1)  # [N, 1]
    member_pos = pos - (rate - 1) + rate_off[None, :]  # [N, rate]
    done = ((pos + 1) % rate == 0) & (pos >= rate - 1)  # [N, 1]
    member_valid = done & (member_pos >= 0)
    # member token row: bt[t // bs] * bs + t % bs
    bt = block_table.reshape(-1)  # [nblk]
    blk_idx = (member_pos.clamp(min=0) // bs).clamp(max=bt.shape[0] - 1)
    slots = bt[blk_idx] * bs + member_pos.clamp(min=0) % bs  # [N, rate]
    rows = flat[slots.reshape(-1)].reshape(n_tok, rate, width)
    gate = rows[..., head_dim : 2 * head_dim].float()
    logits = gate + ape.float()[None]  # ape [rate, head_dim]
    logits = logits.masked_fill(~member_valid[..., None], float("-inf"))
    weights = torch.nan_to_num(logits.softmax(1)).to(torch.bfloat16)
    keys = rows[..., :head_dim].float()
    # mirrors pooled_states' (w * k).sum: per-product bf16 round + fp32 accumulation
    prod = (weights.float() * keys).to(torch.bfloat16).float()
    compressed = prod.sum(1).to(torch.bfloat16)  # [N, head_dim]
    # write slot: bt[p // pool_bs] * pool_bs + p % pool_bs
    pool_id = (pos // rate).reshape(-1)  # [N]
    pool_blk = (pool_id // pool_bs).clamp(max=bt.shape[0] - 1)
    pool_slots = bt[pool_blk] * pool_bs + pool_id % pool_bs
    pool_slots = pool_slots.clamp(0, pool_flat.shape[0] - 1)
    write = done.reshape(-1)
    if n_tok > 1:
        # Prefill chunk (eager): multiple tokens of the same pool all issue writes,
        # so repeated in-place assignment is non-deterministic (the old value of a
        # non-done slot written back would overwrite a done slot's compressed value);
        # first filter down to only completed pools. decode graph takes the n_tok==1
        # branch (no repeats, static shapes).
        sel = write.nonzero().flatten()
        pool_flat.index_copy_(0, pool_slots[sel], compressed[sel])
    else:
        # masked in-place write (graph static shapes): unfinished pools retain their original value
        pool_flat[pool_slots] = torch.where(write[:, None], compressed, pool_flat[pool_slots])


def read_pools(pool_cache: torch.Tensor, block_table: torch.Tensor, kv_lens: torch.Tensor, n_pools: int, rate: int):
    """Direct read of the pool cache -> ``(pool_keys [B,P,D], pool_indices [B,P,rate] int64,
    pool_valid [B,P] bool)``.

    ``pool_valid = 4(p+1) <= kv_len`` is equivalent to the old path's "4 slots all valid";
    the -1 mask of ``pool_indices`` reuses the old ``slot_valid`` semantics
    (``4p+s >= kv_len[b]`` -> set to -1, per sequence).
    """
    pool_bs = pool_cache.shape[1]
    head_dim = pool_cache.shape[-1]
    device = pool_cache.device
    B = block_table.shape[0]
    pool_flat = pool_cache.reshape(-1, head_dim)
    offs = torch.arange(n_pools, device=device)
    # [B, P] = bt[b, p // pool_bs] * pool_bs + p % pool_bs
    blk = (offs // pool_bs).clamp(max=block_table.shape[1] - 1)
    pool_slots = block_table[:, blk].clamp(min=0) * pool_bs + offs[None, :] % pool_bs
    pool_keys = pool_flat[pool_slots.reshape(-1)].reshape(B, n_pools, head_dim)
    slot_off = torch.arange(rate, device=device)
    member = offs[:, None] * rate + slot_off[None, :]  # [P, rate]
    # -1 mask aligns with the old slot_valid (key_valid & in_range): mask per sequence by kv_len
    member_b = member[None].expand(B, -1, -1)
    pool_indices = member_b.masked_fill(member_b >= kv_lens.reshape(-1, 1, 1), -1).contiguous()
    pool_valid = (offs[None, :] + 1) * rate <= kv_lens.reshape(-1, 1)
    return pool_keys, pool_indices, pool_valid
