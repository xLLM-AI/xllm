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

import triton
import triton.language as tl


@triton.jit
def _prefix(mask: tl.tensor, BN: tl.constexpr, C: tl.constexpr) -> tl.tensor:
    # Boolean dot products give exact local counts and avoid a serial scan.
    if BN >= 256:
        r = tl.arange(0, C)
        triangular = (r[:, None] <= r[None, :]).to(tl.bfloat16)
        blocks = mask.to(tl.float32).reshape(BN // C, C)
        local = tl.dot(blocks.to(tl.bfloat16), triangular)
        counts = tl.sum(blocks, 1)
        g = tl.arange(0, BN // C)
        before = tl.sum(tl.where(g[:, None] < g[None, :], counts[:, None], 0.0), 0)
        return (local + before[:, None]).reshape(BN)
    r = tl.arange(0, BN)
    return tl.sum(tl.where(r[:, None] <= r[None, :], mask[:, None].to(tl.float32), 0.0), 0)


@triton.jit
def kpool_complete(
    k: tl.tensor,
    gate: tl.tensor,
    ape: tl.tensor,
    hadamard: tl.tensor,
    cache: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    row_batch: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    table_stride: tl.int64,
    P: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
) -> None:
    """Complete independent pools before any request's old tail is overwritten."""
    i = tl.program_id(0)
    pos = tl.load(positions + i)
    if (pos >= 0) & (pos % P == P - 1):
        batch = tl.load(row_batch + i)
        start = tl.load(starts + batch)
        tail_id = tl.load(tail_ids + batch)
        r = tl.arange(0, BLOCK_P)
        d = tl.arange(0, BLOCK_D)
        source = i - P + 1 + r
        current = source >= start
        source_pos = tl.load(positions + source, (r < P) & current, -1)
        current = current & (source_pos >= 0)
        raw_k = tl.load(
            k + source[:, None] * k_stride + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & current[:, None],
            0,
        ).to(tl.float32)
        raw_g = tl.load(
            gate + source[:, None] * gate_stride + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & current[:, None],
            0,
        ).to(tl.float32)
        tail_r = (pos - P + 1 + r) % T
        old_k = tl.load(
            tail + (tail_id * 2 * T + tail_r[:, None]) * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & ~current[:, None],
            0,
        ).to(tl.float32)
        old_g = tl.load(
            tail + (tail_id * 2 * T + T + tail_r[:, None]) * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D) & ~current[:, None],
            0,
        ).to(tl.float32)
        a = tl.load(
            ape + r[:, None] * D + d[None, :],
            (r[:, None] < P) & (d[None, :] < D),
            0,
            cache_modifier=".ca",
        )
        logits = tl.where(r[:, None] < P, tl.where(current[:, None], raw_g, old_g) + a, -float("inf"))
        prob = tl.exp(logits - tl.max(logits, 0)[None, :])
        prob = prob / tl.sum(prob, 0)[None, :]
        pooled = tl.sum(tl.where(current[:, None], raw_k, old_k) * prob, 0)
        pooled = pooled.to(tl.bfloat16).to(tl.float32)
        matrix = tl.load(
            hadamard + d[:, None] * D + d[None, :],
            (d[:, None] < D) & (d[None, :] < D),
            0,
            cache_modifier=".ca",
        )
        rotated = tl.sum(matrix * pooled[None, :], 1)
        pool = pos // P
        block = tl.load(table + batch * table_stride + pool // POOL_BLOCK)
        slot = block * POOL_BLOCK + pool % POOL_BLOCK
        tl.store(cache + slot * D + d, rotated, d < D)


@triton.jit
def kpool_stash(
    k: tl.tensor,
    gate: tl.tensor,
    tail: tl.tensor,
    tail_ids: tl.tensor,
    positions: tl.tensor,
    starts: tl.tensor,
    k_stride: tl.int64,
    gate_stride: tl.int64,
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SINGLE_TOKEN: tl.constexpr,
) -> None:
    """Only the last writer of each circular slot survives a batch update."""
    batch = tl.program_id(0)
    start = tl.load(starts + batch)
    end = tl.load(starts + batch + 1)
    last = tl.load(positions + end - 1, end > start, -1)
    while (end > start) & (last < 0):
        end -= 1
        last = tl.load(positions + end - 1, end > start, -1)
    if end > start:
        tail_id = tl.load(tail_ids + batch)
        if (last >= 0) & (tail_id >= 0):
            d = tl.arange(0, BLOCK_D)
            if SINGLE_TOKEN:
                r = last % T
                mask = d < D
                kv = tl.load(k + (end - 1) * k_stride + d, mask, 0)
                gv = tl.load(gate + (end - 1) * gate_stride + d, mask, 0)
                tl.store(tail + (tail_id * 2 * T + r) * D + d, kv, mask)
                tl.store(tail + (tail_id * 2 * T + T + r) * D + d, gv, mask)
            else:
                r = tl.arange(0, BLOCK_P)
                source = end - 1 - (last % T - r + T) % T
                source_pos = tl.load(positions + source, (r < T) & (source >= start), -1)
                mask = (r[:, None] < T) & (d[None, :] < D) & (source_pos[:, None] >= 0)
                kv = tl.load(k + source[:, None] * k_stride + d[None, :], mask, 0)
                gv = tl.load(gate + source[:, None] * gate_stride + d[None, :], mask, 0)
                tl.store(tail + (tail_id * 2 * T + r[:, None]) * D + d[None, :], kv, mask)
                tl.store(tail + (tail_id * 2 * T + T + r[:, None]) * D + d[None, :], gv, mask)


@triton.jit
def kpool_rows(
    starts: tl.tensor,
    rows: tl.tensor,
    tokens: tl.int64,
    N: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
) -> None:
    token = tl.program_id(0) * BT + tl.arange(0, BT)
    req = tl.arange(0, BN)
    begin = tl.load(starts + req, req < N, 0x7FFFFFFFFFFFFFFF)
    row = tl.sum((token[:, None] >= begin[None, :]).to(tl.int32), 1) - 1
    tl.store(rows + token, row, token < tokens)


# Four-query KPool verification adapted from vLLM-MLU kpool_logits.py
# at commit 0029935c161e38d3979a67593bc5cacc1fffd939.


@triton.jit(do_not_specialize=["scale", "table_width"])
def score_verify_pools(
    query: tl.tensor,
    weights: tl.tensor,
    cache: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    query_starts: tl.tensor,
    output: tl.tensor,
    tokens: tl.int64,
    capacity: tl.int64,
    cache_blocks: tl.int64,
    table_width: tl.int64,
    output_stride: tl.int64,
    scale: tl.float32,
    BLOCK_POOLS: tl.constexpr,
    TILES_PER_PROGRAM: tl.constexpr,
) -> None:
    """Score a request's at-most-four live queries against each shared key tile.

    The caller selects this path for T4 verify batches; replay may shorten or
    mask individual requests. Q/cache are contiguous BF16, weights/output FP32,
    with H=32, D=128 and four compressed pools per physical cache page.
    """
    request_id = tl.program_id(0)
    begin = tl.load(query_starts + request_id).to(tl.int64)
    end = tl.load(query_starts + request_id + 1).to(tl.int64)
    t = tl.arange(0, 4)
    h = tl.arange(0, 32)
    d = tl.arange(0, 128)
    query_ids = begin + t
    query_valid = (query_ids < end) & (query_ids < tokens)
    request = tl.load(rows + begin, (begin < end) & (begin < tokens), 0).to(tl.int64)
    request_valid = (request >= 0) & (request < tl.num_programs(0))
    safe_request = tl.where(request_valid, request, 0)
    lengths = tl.load(positions + query_ids, query_valid, -1).to(tl.int64) + 1
    live_pools = tl.minimum(tl.maximum(lengths, 0) // 4, tl.minimum(capacity, table_width * 4))
    query_values = tl.load(
        query + query_ids[None, :, None] * 4096 + h[:, None, None] * 128 + d[None, None, :],
        query_valid[None, :, None],
        0.0,
    ).reshape(128, 128)
    head_weights = tl.load(
        weights + query_ids[None, :] * 32 + h[:, None],
        query_valid[None, :],
        0.0,
    ).to(tl.float32)
    first_tile = tl.program_id(1) * TILES_PER_PROGRAM
    for tile in range(first_tile, first_tile + TILES_PER_PROGRAM):
        pools = tile * BLOCK_POOLS + tl.arange(0, BLOCK_POOLS)
        logical_pages = tile * (BLOCK_POOLS // 4) + tl.arange(0, BLOCK_POOLS // 4)
        page_live = (logical_pages < table_width) & request_valid
        pages = tl.load(
            table + safe_request * table_width + logical_pages,
            page_live,
            -1,
        ).to(tl.int64)
        page_live &= (pages >= 0) & (pages < cache_blocks)
        safe_pages = tl.where(page_live, pages, 0)
        keys = tl.load(
            cache + safe_pages[:, None] * 512 + tl.arange(0, 512)[None, :],
            cache_blocks > 0,
            0.0,
            cache_modifier=".cg",
        ).reshape(BLOCK_POOLS, 128)
        dots = tl.dot(query_values, tl.trans(keys), allow_tf32=False).to(tl.float32)
        dots = dots.reshape(32, 4, BLOCK_POOLS)
        weighted = tl.maximum(dots * scale, 0.0) * head_weights[:, :, None]
        scores = tl.sum(weighted, 0)
        pool_valid = tl.broadcast_to(page_live[:, None], (BLOCK_POOLS // 4, 4)).reshape(BLOCK_POOLS)
        valid = pool_valid[None, :] & (pools[None, :] < live_pools[:, None])
        tl.store(
            output + query_ids[:, None] * output_stride + pools[None, :],
            tl.where(valid, scores, -float("inf")),
            query_valid[:, None] & (pools[None, :] < capacity),
        )
        if request_id == tl.num_programs(0) - 1:
            for padding_row in range(end, tokens):
                tl.store(
                    output + padding_row * output_stride + pools,
                    -float("inf"),
                    pools < capacity,
                )
