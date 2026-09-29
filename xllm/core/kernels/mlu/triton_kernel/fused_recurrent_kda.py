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

# Local implementation of sequence/head/value parallel KDA recurrence. The
# independent-tile scheduling follows vLLM's GLM5Next KDA implementation at
# commit 58ad1f3b8973b23943107b51230d594050b42ec3, with MLU value tiles.
# The arithmetic follows xLLM's fused_sigmoid_gating_delta_rule_update.py,
# derived from vLLM and flash-linear-attention (Songlin Yang, Yu Zhang).
# This entry handles both regular decode and ragged speculative verification.
# Each sequence uses its device CU boundaries and accepted checkpoint; BV is
# selected by the C++ dispatcher without changing those recurrence semantics.

import triton
import triton.language as tl


@triton.jit(do_not_specialize=["n"])
def fused_recurrent_kda_kernel(
    q_ptr: tl.tensor,
    k_ptr: tl.tensor,
    v_ptr: tl.tensor,
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    a_log_ptr: tl.tensor,
    dt_bias_ptr: tl.tensor,
    initial_state_ptr: tl.tensor,
    final_state_ptr: tl.tensor,
    output_ptr: tl.tensor,
    cu_seqlens_ptr: tl.tensor,
    state_indices_ptr: tl.tensor,
    accepted_tokens_ptr: tl.tensor,
    n: tl.int32,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    STRIDE_INDICES_SEQ: tl.constexpr,
    STRIDE_INDICES_TOK: tl.constexpr,
    SCALE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    SPEC: tl.constexpr,
    INPLACE: tl.constexpr,
    BV: tl.constexpr,
    BK: tl.constexpr,
) -> None:
    nv: tl.constexpr = triton.cdiv(DV, BV)
    tile = tl.program_id(0)
    seq = tile // (HV * nv)
    head = tile // nv % HV
    value_tile = tile % nv
    kh = head // (HV // H)
    kk = tl.arange(0, BK)
    vv = value_tile * BV + tl.arange(0, BV)
    mask = (vv[:, None] < DV) & (kk[None, :] < DK)
    bos = tl.load(cu_seqlens_ptr + seq)
    eos = tl.load(cu_seqlens_ptr + seq + 1)
    if bos == eos:
        return
    accepted = 0
    if SPEC:
        accepted = tl.load(accepted_tokens_ptr + seq) - 1
    slot = tl.load(state_indices_ptr + seq * STRIDE_INDICES_SEQ + accepted * STRIDE_INDICES_TOK).to(tl.int64)
    if slot <= 0:
        return
    state_offset = head * DV * DK + vv[:, None] * DK + kk[None, :]
    state = tl.load(initial_state_ptr + slot * HV * DV * DK + state_offset, mask=mask, other=0).to(tl.float32)
    bias = tl.load(dt_bias_ptr + head * DK + kk, mask=kk < DK, other=0).to(tl.float32)
    a_scale = tl.exp(tl.load(a_log_ptr + head).to(tl.float32))
    query_ptr = q_ptr + (bos * H + kh) * DK + kk
    key_ptr = k_ptr + (bos * H + kh) * DK + kk
    gate_ptr = a_ptr + (bos * HV + head) * DK + kk
    beta_ptr = b_ptr + bos * HV + head
    value_ptr = v_ptr + (bos * HV + head) * DV + vv
    out_ptr = output_ptr + (bos * HV + head) * DV + vv
    for token in range(bos, eos):
        # Small value tiles keep per-token address generation.
        if BV == 32:
            query_ptr = q_ptr + (token * H + kh) * DK + kk
            key_ptr = k_ptr + (token * H + kh) * DK + kk
            gate_ptr = a_ptr + (token * HV + head) * DK + kk
            beta_ptr = b_ptr + token * HV + head
            value_ptr = v_ptr + (token * HV + head) * DV + vv
            out_ptr = output_ptr + (token * HV + head) * DV + vv
        q = tl.load(query_ptr, mask=kk < DK, other=0).to(tl.float32)
        k = tl.load(key_ptr, mask=kk < DK, other=0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
        a = tl.load(gate_ptr, mask=kk < DK, other=0).to(tl.float32)
        gate = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a + bias)))
        beta = tl.sigmoid(tl.load(beta_ptr).to(tl.float32))
        value = tl.load(value_ptr, mask=vv < DV, other=0).to(tl.float32)
        state = state * gate[None, :]
        delta = (value - tl.sum(state * k[None, :], 1)) * beta
        state = state + delta[:, None] * k[None, :]
        output = tl.sum(state * q[None, :], 1)
        tl.store(out_ptr, output, mask=vv < DV)
        if INPLACE:
            final_slot = tl.load(state_indices_ptr + seq * STRIDE_INDICES_SEQ + (token - bos) * STRIDE_INDICES_TOK).to(
                tl.int64
            )
            if final_slot > 0:
                tl.store(final_state_ptr + final_slot * HV * DV * DK + state_offset, state, mask=mask)
        else:
            tl.store(final_state_ptr + token * HV * DV * DK + state_offset, state, mask=mask)
        if BV != 32:
            query_ptr += H * DK
            key_ptr += H * DK
            gate_ptr += HV * DK
            beta_ptr += HV
            value_ptr += HV * DV
            out_ptr += HV * DV
