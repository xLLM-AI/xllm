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
"""NPU Triton kernel for GLM-Next KPool compression on the decode path.

Mirrors the pure-torch ``compress_completed_pools(batched=True)`` semantics:
each program handles one token that completes a pool, gathers the ``rate``
member K/gate columns from the 257-wide index cache, computes
``softmax(gate + ape)`` (fp32) with the same per-product bf16 rounding as the
reference, and writes the pooled key into the paged pool cache. One Triton
launch replaces the torch gather/softmax/sum/write small-op flood.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _kpool_compress_decode_kernel(
    index_cache_ptr,
    pool_cache_ptr,
    block_table_ptr,
    positions_ptr,
    ape_ptr,
    bs,
    width,
    pool_bs,
    bt_stride_req,
    bt_stride_blk,
    bt_width,
    num_index_rows,
    num_pool_slots,
    pos_stride,
    ape_stride_row,
    ape_stride_dim,
    RATE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fuse KPool window gather + softmax + weighted sum + cache write."""
    t = tl.program_id(0)
    d_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM

    pos = tl.load(positions_ptr + t * pos_stride).to(tl.int64)
    done = ((pos + 1) % RATE == 0) & (pos >= RATE - 1)
    pool_id = pos // RATE
    req = t  # decode: num_tokens == n_seqs

    # The online-softmax update combines max and normalizer in the first pass;
    # the weighted numerator is accumulated in a second pass.
    # ``valid`` members contribute exp(x), invalid members (x == -inf) contribute
    # nothing; the safe-exp guards below keep early decode (pos < RATE-1) free
    # of NaN in the running normalizer.
    logit_max = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
    denom = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in tl.static_range(RATE):
        mpos = pos - (RATE - 1 - i)
        valid = mpos >= 0
        # Clamp mpos so blk_idx >= 0: for early decode (pos < RATE-1) the
        # negative member positions must not produce a negative floor-division
        # block index that would read before the block table (the index-cache
        # loads below are already masked by ``valid``).
        mpos_safe = tl.maximum(mpos, 0)
        blk_idx = tl.minimum(mpos_safe // bs, bt_width - 1)
        slot_off = mpos_safe % bs
        bt = tl.load(block_table_ptr + req * bt_stride_req + blk_idx * bt_stride_blk).to(tl.int64)
        # slot_off < bs, so bt == slot // bs and slot_off == slot % bs; the
        # intermediate slot/blk/off decomposition is an identity and is omitted.
        slot = bt * bs + slot_off
        slot = tl.where(slot < 0, slot + num_index_rows, slot)
        base = slot * width
        g = tl.load(
            index_cache_ptr + base + HEAD_DIM + d_offsets,
            mask=d_mask & valid & done,
            other=float("-inf"),
        ).to(tl.float32)
        ape_v = tl.load(ape_ptr + i * ape_stride_row + d_offsets * ape_stride_dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        x = g + ape_v  # -inf for invalid members
        m_new = tl.maximum(logit_max, x)
        # exp(logit_max - m_new) rescales the accumulated normalizer; exp(x -
        # m_new) is this member's contribution. Both are guarded so a leading
        # run of invalid members (x == -inf, logit_max == -inf) yields 0 rather
        # than exp(nan).
        rescale = tl.where(logit_max == float("-inf"), 0.0, tl.exp(logit_max - m_new))
        contrib = tl.where(x == float("-inf"), 0.0, tl.exp(x - m_new))
        denom = denom * rescale + contrib
        logit_max = m_new

    # ---- (normalized weight * key) numerator ----
    # Normalize BEFORE the bf16 round to match the torch reference:
    #   weights = softmax(...).to(bf16); prod = (weights.float()*k).to(bf16).float()
    num = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for i in tl.static_range(RATE):
        mpos = pos - (RATE - 1 - i)
        valid = mpos >= 0
        mpos_safe = tl.maximum(mpos, 0)
        blk_idx = tl.minimum(mpos_safe // bs, bt_width - 1)
        slot_off = mpos_safe % bs
        bt = tl.load(block_table_ptr + req * bt_stride_req + blk_idx * bt_stride_blk).to(tl.int64)
        # slot_off < bs, so bt == slot // bs and slot_off == slot % bs; the
        # intermediate slot/blk/off decomposition is an identity and is omitted.
        slot = bt * bs + slot_off
        slot = tl.where(slot < 0, slot + num_index_rows, slot)
        base = slot * width
        g = tl.load(
            index_cache_ptr + base + HEAD_DIM + d_offsets,
            mask=d_mask & valid & done,
            other=float("-inf"),
        ).to(tl.float32)
        k = tl.load(index_cache_ptr + base + d_offsets, mask=d_mask & valid & done, other=0.0).to(tl.float32)
        ape_v = tl.load(ape_ptr + i * ape_stride_row + d_offsets * ape_stride_dim, mask=d_mask, other=0.0).to(
            tl.float32
        )
        w_norm = tl.exp(g + ape_v - logit_max) / denom
        w_bf16 = w_norm.to(tl.bfloat16).to(tl.float32)
        prod = (w_bf16 * k).to(tl.bfloat16).to(tl.float32)
        num += prod

    compressed = num.to(tl.bfloat16)

    # ---- write to pool cache (only if this token completes a pool) ----
    # The pool cache is paged: map the logical pool block through the request's
    # block table to the physical block, matching the torch reference's
    # ``bt[p // pool_bs] * pool_bs + p % pool_bs``.
    pool_blk_logical = tl.minimum(tl.maximum(pool_id // pool_bs, 0), bt_width - 1)
    pool_off = pool_id % pool_bs
    pool_blk_physical = tl.load(block_table_ptr + req * bt_stride_req + pool_blk_logical * bt_stride_blk).to(tl.int64)
    out_slot = tl.minimum(tl.maximum(pool_blk_physical * pool_bs + pool_off, 0), num_pool_slots - 1)
    # Aliased completed slots read the same aligned pool members. Unfinished
    # rows never write, so they cannot overwrite a completed pool with old data.
    tl.store(pool_cache_ptr + out_slot * HEAD_DIM + d_offsets, compressed, mask=d_mask & done)


def compress_completed_pools_decode(
    index_cache: torch.Tensor,
    pool_cache: torch.Tensor,
    block_table: torch.Tensor,
    positions: torch.Tensor,
    ape: torch.Tensor,
    head_dim: int,
    rate: int,
) -> None:
    """Triton fast path for the decode ``batched=True`` branch.

    Semantics match the pure-torch ``compress_completed_pools`` exactly for
    the batched case: one token position per ``block_table`` row, and every
    token is the last token of a pool (``(pos + 1) % rate == 0``).
    Non-completing tokens are skipped via the ``done`` mask.

    ``index_cache`` is ``[num_blocks, block_size, num_kv_heads, width]`` with
    ``width == 2 * head_dim + 1`` (packed ``[k, gate, valid]``); the kernel
    indexes it as a flattened ``token-row`` view, so it must be contiguous.
    """
    num_tokens = positions.shape[0]
    if num_tokens == 0:
        return

    bs = index_cache.shape[1]  # index_cache token block_size
    pool_bs = pool_cache.shape[1]  # pool-granular block_size = bs // rate
    width = 2 * head_dim + 1

    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_tokens, triton.cdiv(head_dim, block_d))
    _kpool_compress_decode_kernel[grid](
        index_cache,
        pool_cache,
        block_table,
        positions,
        ape,
        bs,
        width,
        pool_bs,
        block_table.stride(0),
        block_table.stride(1),
        block_table.shape[1],
        index_cache.numel() // width,
        pool_cache.numel() // head_dim,
        positions.stride(0),
        ape.stride(0),
        ape.stride(1),
        RATE=rate,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
    )


__all__ = ["compress_completed_pools_decode"]
