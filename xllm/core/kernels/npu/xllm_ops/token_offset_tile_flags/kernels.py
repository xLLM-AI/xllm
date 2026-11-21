# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton Ascend tile predicates; no temporary global-memory tensors."""

import triton
import triton.language as tl


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def tile_flags(
    offsets: tl.tensor,
    indptr: tl.tensor,
    lengths: tl.tensor,
    output: tl.tensor,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    LANES: tl.constexpr,
    GROUP: tl.constexpr,
) -> None:
    base = tl.program_id(0) * GROUP
    relative = tl.arange(0, GROUP)
    valid = relative < TOTAL - base
    ids = tl.where(valid, base + relative, 0)
    row = ids // K
    tile = ids % K
    length = tl.load(lengths + row, mask=valid, other=0)
    start = tl.load(indptr + row, mask=valid, other=0)
    full = valid & (tile < length // T)
    begin = tl.where(full, tile, 0) * T
    address = tl.where(full, start + begin, 0)
    first = tl.load(offsets + address, mask=full & (address < N), other=0)
    lane = tl.arange(0, LANES)
    index = address[:, None] + tl.minimum(lane[None, :], T - 1)
    active = full[:, None] & (lane[None, :] < T) & (index < N)
    data = tl.load(offsets + index, mask=active, other=0)
    bad = (lane[None, :] < T) & ((data - first[:, None]) != lane[None, :])
    result = full & (first % T == 0) & (tl.sum(bad.to(tl.int32), axis=1) == 0)
    tl.store(output + ids, result.to(tl.uint8), mask=valid)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def tile_flags_row(
    offsets: tl.tensor,
    indptr: tl.tensor,
    lengths: tl.tensor,
    output: tl.tensor,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    LANES: tl.constexpr,
    GROUP: tl.constexpr,
) -> None:
    chunks: tl.constexpr = (K + GROUP - 1) // GROUP
    row = tl.program_id(0) // chunks
    tile_base = (tl.program_id(0) % chunks) * GROUP
    relative = tl.arange(0, GROUP)
    valid = relative < K - tile_base
    tile = tl.where(valid, tile_base + relative, 0)
    length = tl.load(lengths + row)
    start = tl.load(indptr + row)
    count = length // T
    full = valid & (tile < count)
    address = tl.where(full, start + tl.where(full, tile, 0) * T, 0)
    first = tl.load(offsets + address, mask=full & (address < N), other=0)
    linear = tl.arange(0, GROUP * LANES)
    active = (linear // T < K - tile_base) & (linear // T < count - tile_base)
    safe_base = tl.where(tile_base < count, tile_base, 0) * T
    if N <= 2147483647 - GROUP * LANES:
        index = start + safe_base + linear
    else:
        index = start.to(tl.int64) + safe_base + linear
    flat = tl.load(offsets + index, mask=active, other=0)
    data = tl.reshape(flat, (GROUP, LANES))
    lane = tl.arange(0, LANES)
    bad = (data - first[:, None]) != lane[None, :]
    result = full & (first % T == 0) & (tl.sum(bad.to(tl.int32), axis=1) == 0)
    tl.store(output + row * K + tile, result.to(tl.uint8), mask=valid)


@triton.jit(do_not_specialize_on_alignment=[0, 1, 2])
def tile_flags_large(
    offsets: tl.tensor,
    indptr: tl.tensor,
    lengths: tl.tensor,
    output: tl.tensor,
    N: tl.constexpr,
    TOTAL: tl.constexpr,
    T: tl.constexpr,
    K: tl.constexpr,
    LANES: tl.constexpr,
    GROUP: tl.constexpr,
) -> None:
    # Avoid the compiler's degenerate [1,4096] grouped-reduction path.
    q = tl.program_id(0)
    row = q // K
    tile = q % K
    length = tl.load(lengths + row)
    start = tl.load(indptr + row)
    full = tile < length // T
    # Validate fullness before multiplication. Active addresses are <=N-1;
    # inactive tiles use index zero, so even huge unused K cannot overflow.
    begin = tl.where(full, tile, 0) * T
    address = tl.where(full, start + begin, 0)
    first = tl.load(offsets + address, mask=full & (address < N), other=0)
    lane = tl.arange(0, LANES)
    safe_lane = tl.minimum(lane, T - 1)
    data = tl.load(offsets + address + safe_lane, mask=full & (lane < T), other=0)
    bad = (lane < T) & ((data - first) != lane)
    result = full & (first % T == 0) & (tl.sum(bad.to(tl.int32), axis=0) == 0)
    tl.store(output + q, result.to(tl.uint8))
