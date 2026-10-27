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
def tmo_dcp_decode_kernel(
    SRC: tl.tensor,
    DST: tl.tensor,
    L: tl.tensor,
    S: tl.tensor,
    R: tl.constexpr,
    B: tl.constexpr,
    TRANSPOSE: tl.constexpr,
    G: tl.constexpr,
) -> None:
    first_tile = B - 1 - tl.program_id(0)
    preout = tl.load(SRC + first_tile * G * 512 + tl.arange(0, G * 512))
    for index in range(tl.program_id(0), B, tl.num_programs(0)):
        tile = B - 1 - index
        heads = tile * G + tl.arange(0, G)
        output = tl.reshape(preout, (G, 512)).to(tl.float32)
        lookahead = preout
        next_tile = tile - tl.num_programs(0)
        if next_tile >= 0:
            lookahead = tl.load(SRC + next_tile * G * 512 + tl.arange(0, G * 512))
        lptr = tl.make_block_ptr(L, (4, B * 64), (B * 64, 1), (0, tile * G), (4, G), (1, 0))
        raw = tl.load(lptr)
        clean = tl.where(raw < float("inf"), raw, -float("inf"))
        maximum = tl.maximum(tl.max(clean, 0), -3.4028234663852886e38)
        total = tl.sum(tl.exp(clean - maximum[None, :]), 0)
        global_lse = tl.log(total) + maximum
        local = tl.load(L + R * B * 64 + heads)
        diff = local - global_lse
        diff = tl.where(diff < float("inf"), diff, -float("inf"))
        factor = tl.exp(diff)
        padding = tl.load(S + tile) < 0
        result = (output * factor[:, None]).to(tl.bfloat16)
        result = tl.where((padding | (factor == 0))[:, None], tl.full((), 0, tl.bfloat16), result)
        if TRANSPOSE:
            dst_offsets = (tl.arange(0, G)[:, None] * B + tile) * 512 + tl.arange(0, 512)[None, :]
        else:
            dst_offsets = (tile * G + tl.arange(0, G)[:, None]) * 512 + tl.arange(0, 512)[None, :]
        tl.store(DST + dst_offsets, result)
        preout = lookahead


@triton.jit
def tmo_dcp_correct_attn_out_kernel(
    SRC: tl.tensor,
    DST: tl.tensor,
    L: tl.tensor,
    S: tl.tensor,
    rank: int,
    n: int,
    v: int,
    sb: int,
    sh: int,
    sv: int,
    db: int,
    dh: int,
    dv: int,
    ln: int,
    lb: int,
    lh: int,
    VN: tl.constexpr,
    NN: tl.constexpr,
    BASE_E: tl.constexpr,
) -> None:
    b = tl.program_id(0)
    h = tl.program_id(1)
    ns = tl.arange(0, NN)
    lse = tl.load(L + ns * ln + b * lb + h * lh, ns < n, other=-float("inf")).to(tl.float32)
    clean = tl.where(lse < float("inf"), lse, -float("inf"))
    maximum = tl.max(clean, 0)
    maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
    shifted = clean - maximum
    if BASE_E:
        global_lse = tl.log(tl.sum(tl.exp(shifted), 0)) + maximum
    else:
        global_lse = tl.log2(tl.sum(tl.exp2(shifted), 0)) + maximum
    local = tl.load(L + rank * ln + b * lb + h * lh).to(tl.float32)
    diff = local - global_lse
    diff = tl.where(diff < float("inf"), diff, -float("inf"))
    if BASE_E:
        factor = tl.exp(diff)
    else:
        factor = tl.exp2(diff)
        # Preserve nonzero subnormal weights on MLU (including NaN/Inf outputs).
        subnormal = tl.exp2(diff + 126.0) * 1.1754943508222875e-38
        factor = tl.where(diff < -126.0, subnormal, factor)
    dims = tl.arange(0, VN)
    value = tl.load(SRC + b * sb + h * sh + dims * sv, dims < v, other=0).to(tl.float32)
    result = (value * factor).to(DST.dtype.element_ty)
    zero = (tl.load(S + b) < 0) | (factor == 0)
    result = tl.where(zero, tl.full((), 0, DST.dtype.element_ty), result)
    tl.store(DST + b * db + h * dh + dims * dv, result, dims < v)


@triton.jit
def tmo_dcp_prefill_kernel(
    output_ptr: tl.tensor,
    L: tl.tensor,
    S: tl.tensor,
    B: int,
    H: tl.constexpr,
    R: tl.constexpr,
    G: tl.constexpr,
) -> None:
    for tile in range(
        tl.program_id(0) * tl.cdiv(tl.cdiv(B * H, G), tl.num_programs(0)),
        tl.minimum(
            (tl.program_id(0) + 1) * tl.cdiv(tl.cdiv(B * H, G), tl.num_programs(0)),
            tl.cdiv(B * H, G),
        ),
    ):
        rows = tile * G + tl.arange(0, G)
        ns = tl.arange(0, 4)
        lse = tl.load(
            L + ns[:, None] * (B * H) + rows[None, :],
            rows[None, :] < B * H,
            other=-float("inf"),
        )
        lse = tl.where(~(lse < float("inf")), -float("inf"), lse)
        maximum = tl.max(lse, 0)
        maximum = tl.where(maximum == -float("inf"), 0.0, maximum)
        shifted = lse - maximum[None, :]
        total = tl.sum(tl.exp(shifted), 0)
        global_lse = tl.log(total) + maximum
        local = tl.load(L + R * (B * H) + rows, rows < B * H, other=-float("inf"))
        diff = local - global_lse
        diff = tl.where(~(diff < float("inf")), -float("inf"), diff)
        factor = tl.exp(diff)
        if H == 64:
            padding = tl.full((G,), tl.load(S + tile * G // H) < 0, tl.int1)
        else:
            padding = tl.load(S + rows // H, rows < B * H, other=-1) < 0
        d = tl.arange(0, 512)
        offsets = rows[:, None] * 512 + d[None, :]
        output = tl.load(output_ptr + offsets, rows[:, None] < B * H, other=0).to(tl.float32)
        output = (output * factor[:, None]).to(tl.bfloat16)
        output = tl.where(padding[:, None] | (factor[:, None] == 0), tl.cast(0, tl.bfloat16), output)
        tl.store(output_ptr + offsets, output, rows[:, None] < B * H)
