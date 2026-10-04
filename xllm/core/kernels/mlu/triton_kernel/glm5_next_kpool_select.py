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
import triton.language.extra.mlu as mlu

from xllm.core.kernels.mlu.triton_kernel.glm5_next_kpool import _prefix


@triton.jit
def score(
    q: tl.tensor,
    weights: tl.tensor,
    keys: tl.tensor,
    table: tl.tensor,
    positions: tl.tensor,
    rows: tl.tensor,
    out: tl.tensor,
    q_stride: tl.int64,
    q_head_stride: tl.int64,
    w_stride: tl.int64,
    table_stride: tl.int64,
    out_stride: tl.int64,
    N: tl.int64,
    scale: tl.float32,
    H: tl.constexpr,
    D: tl.constexpr,
    P: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    BH: tl.constexpr,
    BN: tl.constexpr,
    PAGED: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    completed = (tl.load(positions + row) + 1) // P
    valid = (n < N) & (n < completed)
    result = tl.full((BN,), -float("inf"), tl.float32)
    if tl.program_id(1) * BN < completed:
        if PAGED:
            batch = tl.load(rows + row)
            if BN % POOL_BLOCK == 0:
                page = tl.arange(0, BN // POOL_BLOCK)
                page_offset = tl.arange(0, POOL_BLOCK)
                page_start = tl.program_id(1) * BN + page * POOL_BLOCK
                page_base = tl.program_id(1) * (BN // POOL_BLOCK)
                block = tl.load(
                    table + batch * table_stride + page_base + page,
                    page_start < N,
                    0,
                )
                slots = (block[:, None] * POOL_BLOCK + page_offset[None, :]).reshape(BN)
            else:
                block = tl.load(table + batch * table_stride + n // POOL_BLOCK, valid, 0)
                slots = block * POOL_BLOCK + n % POOL_BLOCK
        else:
            slots = n
        k = tl.load(keys + slots[None, :] * D + d[:, None], valid[None, :], 0)
        result = tl.full((BN,), 0, tl.float32)
        for base in range(tl.cdiv(H, BH)):
            heads = base * BH + h
            query = tl.load(
                q + row * q_stride + heads[:, None] * q_head_stride + d[None, :],
                heads[:, None] < H,
                0,
            )
            w = tl.load(weights + row * w_stride + heads, heads < H, 0)
            dots = tl.dot(query, k).to(tl.float32)
            result += tl.sum(tl.maximum(dots * scale, 0) * w[:, None], 0)
    tl.store(out + row * out_stride + n, tl.where(valid, result, -float("inf")), n < N)


@triton.jit
def gather_cache(
    cache: tl.tensor,
    table: tl.tensor,
    keys: tl.tensor,
    request: tl.int64,
    N: tl.int64,
    table_stride: tl.int64,
    D: tl.constexpr,
    POOL_BLOCK: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    d = tl.arange(0, D)
    block = tl.load(
        table + request * table_stride + n // POOL_BLOCK,
        n // POOL_BLOCK < tl.cdiv(N, POOL_BLOCK),
        0,
    )
    slot = tl.maximum(block, 0) * POOL_BLOCK + n % POOL_BLOCK
    slot = tl.max_contiguous(slot, POOL_BLOCK)
    v = tl.load(cache + slot[:, None] * D + d[None, :], n[:, None] < N, 0)
    tl.store(keys + n[:, None] * D + d[None, :], v, n[:, None] < N)


@triton.jit
def select_topk_streaming(
    scores: tl.tensor,
    output: tl.tensor,
    N: tl.int64,
    in_stride: tl.int64,
    K: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    offsets = tl.arange(0, BN)
    bins = tl.arange(0, 16)
    # Resolve the kth ordered FP32 key in eight passes with bounded scratch.
    prefix = tl.full((), 0, tl.uint32)
    remaining = tl.minimum(K, N)
    digit = 0
    done = False
    while (digit < 8) & ~done:
        shift = 28 - 4 * digit
        hist = tl.full((16,), 0, tl.int32)
        for start in range(tl.cdiv(N, BN)):
            i = start * BN + offsets
            v = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
            bits = tl.where(v == 0, 0, tl.where(v != v, 0x7FC00000, v.to(tl.uint32, bitcast=True))).to(tl.uint32)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
            # The current prefix contains only bits more significant than this digit.
            same = (ordered >> (shift + 4).to(tl.uint64)) == (prefix >> (shift + 4).to(tl.uint64))
            if digit == 0:
                same = tl.full((BN,), True, tl.int1)
            bucket = tl.where(same & (i < N), ((ordered >> shift) & 15).to(tl.int32), 16)
            hist += tl.histogram(bucket, 16)
        tail = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        chosen = tl.max(tl.where(tail >= remaining, bins, -1), 0)
        remaining -= tl.sum(tl.where(bins > chosen, hist, 0), 0)
        prefix |= chosen.to(tl.uint32) << shift
        chosen_count = tl.sum(tl.where(bins == chosen, hist, 0), 0)
        done = chosen_count == remaining
        digit += 1
    written = 0
    for start in range(tl.cdiv(N, BN)):
        i = start * BN + offsets
        v = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
        bits = tl.where(v == 0, 0, tl.where(v != v, 0x7FC00000, v.to(tl.uint32, bitcast=True))).to(tl.uint32)
        ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32)
        equal = (ordered == prefix) & (i < N)
        selected = ((ordered > prefix) | (equal & (_wide_prefix(equal, BN).to(tl.int32) <= remaining))) & (i < N)
        rank = _wide_prefix(selected, BN).to(tl.int32) - 1
        value = tl.where((ordered > 0x007FFFFF) & (ordered < 0xFF800000), i, -1)
        tl.store(output + row * K + written + rank, value, selected)
        written += tl.sum(selected.to(tl.int32), 0)
        remaining -= tl.minimum(remaining, tl.sum(equal.to(tl.int32), 0))
    for start in range(tl.cdiv(K, BN)):
        i = start * BN + offsets
        tl.store(output + row * K + i, -1, (i >= N) & (i < K))


@triton.jit
def _wide_prefix(mask: tl.tensor, BN: tl.constexpr) -> tl.tensor:
    return _prefix(mask, BN, 128 if BN >= 2048 else (32 if BN >= 512 else 16))


@triton.jit
def select_topk(
    scores: tl.tensor,
    output: tl.tensor,
    N: tl.int64,
    in_stride: tl.int64,
    K: tl.constexpr,
    BN: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    i = tl.arange(0, BN)
    raw = tl.load(scores + row * in_stride + i, i < N, -float("inf"))
    v = tl.where(raw != raw, float("inf"), raw)
    # Nonfinite winners occupy top-k slots but are returned as padding.
    finite = (tl.abs(v) != float("inf")) & (i < N)
    infinities = tl.sum(((v == float("inf")) & (i < N)).to(tl.float32), 0).to(tl.int32)
    finite_count = tl.sum(finite.to(tl.float32), 0).to(tl.int32)
    threshold = tl.full((), -float("inf"), tl.float32)
    if infinities >= K:
        threshold = tl.full((), float("inf"), tl.float32)
    elif finite_count + infinities > K:
        low = tl.min(tl.where(finite, v, float("inf")), 0)
        high = tl.max(tl.where(finite, v, -float("inf")), 0)
        count = tl.full((), 0, tl.int32)
        target = K - infinities
        while (low < high) & (count != target):
            pivot = low * 0.5 + high * 0.5
            pivot = tl.where(pivot <= low, high, pivot)
            count = tl.sum(((v >= pivot) & finite).to(tl.float32), 0).to(tl.int32)
            low = tl.where(count >= target, pivot, low)
            bits = pivot.to(tl.uint32, bitcast=True)
            ordered = tl.where((bits & 0x80000000) != 0, ~bits, bits ^ 0x80000000).to(tl.uint32) - 1
            down = (
                tl.where((ordered & 0x80000000) != 0, ordered ^ 0x80000000, ~ordered)
                .to(tl.uint32)
                .to(tl.float32, bitcast=True)
            )
            high = tl.where(count < target, down, high)
        threshold = low
    greater = (v > threshold) & (i < N)
    equal = (v == threshold) & (i < N)
    remaining = K - tl.sum(greater.to(tl.float32), 0)
    selected = greater | (equal & (_wide_prefix(equal, BN) <= remaining))
    rank = _wide_prefix(selected, BN).to(tl.int32) - 1
    value = tl.where(finite, i, -1)
    BK: tl.constexpr = triton.next_power_of_2(K)
    compact = mlu.scatter(tl.full((BK,), -1, tl.int32), value, rank, selected)
    j = tl.arange(0, BK)
    tl.store(output + row * K + j, compact, j < K)
