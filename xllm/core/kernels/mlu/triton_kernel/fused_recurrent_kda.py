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
# Persistent programs process contiguous head tasks with full value tiles.
# Four-token verification preloads inputs and uses matrix-vector projections;
# device CU boundaries and accepted checkpoints retain the decode semantics.

import triton
import triton.language as tl


@triton.jit
def _update_tile(
    flat: tl.tensor,
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
    full: tl.constexpr = (BV == DV) and (BK == DK)
    seq = flat // (HV * nv)
    head = flat // nv % HV
    value_tile = flat % nv
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
    if full:
        state = tl.load(initial_state_ptr + slot * HV * DV * DK + state_offset).to(tl.float32)
        bias = tl.load(dt_bias_ptr + head * DK + kk).to(tl.float32)
    else:
        state = tl.load(initial_state_ptr + slot * HV * DV * DK + state_offset, mask=mask, other=0).to(tl.float32)
        bias = tl.load(dt_bias_ptr + head * DK + kk, mask=kk < DK, other=0).to(tl.float32)
    a_scale = tl.exp(tl.load(a_log_ptr + head).to(tl.float32))
    query_ptr = q_ptr + (bos * H + kh) * DK + kk
    key_ptr = k_ptr + (bos * H + kh) * DK + kk
    gate_ptr = a_ptr + (bos * HV + head) * DK + kk
    beta_ptr = b_ptr + bos * HV + head
    value_ptr = v_ptr + (bos * HV + head) * DV + vv
    out_ptr = output_ptr + (bos * HV + head) * DV + vv
    if full:
        if (eos - bos) == 4:
            _four_token_tail(
                state,
                state_offset,
                query_ptr,
                key_ptr,
                value_ptr,
                gate_ptr,
                beta_ptr,
                final_state_ptr,
                out_ptr,
                state_indices_ptr,
                seq,
                bos,
                a_scale,
                bias,
                H,
                HV,
                DK,
                DV,
                STRIDE_INDICES_SEQ,
                STRIDE_INDICES_TOK,
                SCALE,
                LOWER_BOUND,
                INPLACE,
            )
            return
    for token in range(bos, eos):
        if BV == 32:
            query_ptr = q_ptr + (token * H + kh) * DK + kk
            key_ptr = k_ptr + (token * H + kh) * DK + kk
            gate_ptr = a_ptr + (token * HV + head) * DK + kk
            beta_ptr = b_ptr + token * HV + head
            value_ptr = v_ptr + (token * HV + head) * DV + vv
            out_ptr = output_ptr + (token * HV + head) * DV + vv
        if full:
            q = tl.load(query_ptr).to(tl.float32)
            k = tl.load(key_ptr).to(tl.float32)
        else:
            q = tl.load(query_ptr, mask=kk < DK, other=0).to(tl.float32)
            k = tl.load(key_ptr, mask=kk < DK, other=0).to(tl.float32)
        q = q * tl.rsqrt(tl.sum(q * q, 0) + 1.0e-6) * SCALE
        k = k * tl.rsqrt(tl.sum(k * k, 0) + 1.0e-6)
        if full:
            a = tl.load(gate_ptr).to(tl.float32)
        else:
            a = tl.load(gate_ptr, mask=kk < DK, other=0).to(tl.float32)
        gate = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a + bias)))
        beta = tl.sigmoid(tl.load(beta_ptr).to(tl.float32))
        if full:
            value = tl.load(value_ptr).to(tl.float32)
        else:
            value = tl.load(value_ptr, mask=vv < DV, other=0).to(tl.float32)
        state = state * gate[None, :]
        delta = (value - tl.sum(state * k[None, :], 1)) * beta
        state = state + delta[:, None] * k[None, :]
        if INPLACE:
            final_slot = tl.load(state_indices_ptr + seq * STRIDE_INDICES_SEQ + (token - bos) * STRIDE_INDICES_TOK).to(
                tl.int64
            )
            if final_slot > 0:
                tl.store(
                    final_state_ptr + final_slot * HV * DV * DK + state_offset,
                    state,
                    mask=None if full else mask,
                    cache_modifier=".cg",
                )
        else:
            tl.store(
                final_state_ptr + token * HV * DV * DK + state_offset,
                state,
                mask=None if full else mask,
                cache_modifier=".cg",
            )
        output = tl.sum(state * q[None, :], 1)
        tl.store(out_ptr, output, mask=None if full else vv < DV)
        if BV != 32:
            query_ptr += H * DK
            key_ptr += H * DK
            gate_ptr += HV * DK
            beta_ptr += HV
            value_ptr += HV * DV
            out_ptr += HV * DV


@triton.jit
def _four_token_tail(
    state: tl.tensor,
    state_offset: tl.tensor,
    query_ptr: tl.tensor,
    key_ptr: tl.tensor,
    value_ptr: tl.tensor,
    gate_ptr: tl.tensor,
    beta_ptr: tl.tensor,
    final_state_ptr: tl.tensor,
    out_ptr: tl.tensor,
    state_indices_ptr: tl.tensor,
    seq: tl.tensor,
    bos: tl.tensor,
    a_scale: tl.tensor,
    bias: tl.tensor,
    H: tl.constexpr,
    HV: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    STRIDE_INDICES_SEQ: tl.constexpr,
    STRIDE_INDICES_TOK: tl.constexpr,
    SCALE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    INPLACE: tl.constexpr,
) -> None:
    """Preload four tokens before their sequential FP32 state updates."""
    q0 = tl.load(query_ptr).to(tl.float32)
    k0 = tl.load(key_ptr).to(tl.float32)
    a0 = tl.load(gate_ptr).to(tl.float32)
    b0 = tl.load(beta_ptr).to(tl.float32)
    v0 = tl.load(value_ptr).to(tl.float32)
    q1 = tl.load(query_ptr + H * DK).to(tl.float32)
    k1 = tl.load(key_ptr + H * DK).to(tl.float32)
    a1 = tl.load(gate_ptr + HV * DK).to(tl.float32)
    b1 = tl.load(beta_ptr + HV).to(tl.float32)
    v1 = tl.load(value_ptr + HV * DV).to(tl.float32)
    q2 = tl.load(query_ptr + 2 * H * DK).to(tl.float32)
    k2 = tl.load(key_ptr + 2 * H * DK).to(tl.float32)
    a2 = tl.load(gate_ptr + 2 * HV * DK).to(tl.float32)
    b2 = tl.load(beta_ptr + 2 * HV).to(tl.float32)
    v2 = tl.load(value_ptr + 2 * HV * DV).to(tl.float32)
    q3 = tl.load(query_ptr + 3 * H * DK).to(tl.float32)
    k3 = tl.load(key_ptr + 3 * H * DK).to(tl.float32)
    a3 = tl.load(gate_ptr + 3 * HV * DK).to(tl.float32)
    b3 = tl.load(beta_ptr + 3 * HV).to(tl.float32)
    v3 = tl.load(value_ptr + 3 * HV * DV).to(tl.float32)
    if INPLACE:
        so = state_indices_ptr + seq * STRIDE_INDICES_SEQ
        s0 = tl.load(so).to(tl.int64)
        s1 = tl.load(so + STRIDE_INDICES_TOK).to(tl.int64)
        s2 = tl.load(so + 2 * STRIDE_INDICES_TOK).to(tl.int64)
        s3 = tl.load(so + 3 * STRIDE_INDICES_TOK).to(tl.int64)
    q0 = q0 * (tl.rsqrt(tl.sum(q0 * q0, 0) + 1.0e-6) * SCALE)
    q1 = q1 * (tl.rsqrt(tl.sum(q1 * q1, 0) + 1.0e-6) * SCALE)
    q2 = q2 * (tl.rsqrt(tl.sum(q2 * q2, 0) + 1.0e-6) * SCALE)
    q3 = q3 * (tl.rsqrt(tl.sum(q3 * q3, 0) + 1.0e-6) * SCALE)
    k0 = k0 * tl.rsqrt(tl.sum(k0 * k0, 0) + 1.0e-6)
    k1 = k1 * tl.rsqrt(tl.sum(k1 * k1, 0) + 1.0e-6)
    k2 = k2 * tl.rsqrt(tl.sum(k2 * k2, 0) + 1.0e-6)
    k3 = k3 * tl.rsqrt(tl.sum(k3 * k3, 0) + 1.0e-6)
    g0 = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a0 + bias)))
    g1 = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a1 + bias)))
    g2 = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a2 + bias)))
    g3 = tl.exp(LOWER_BOUND * tl.sigmoid(a_scale * (a3 + bias)))
    bt0 = tl.sigmoid(b0)
    bt1 = tl.sigmoid(b1)
    bt2 = tl.sigmoid(b2)
    bt3 = tl.sigmoid(b3)
    st = state * g0[None, :]
    rk0, rq0 = tl.split(tl.dot(st, tl.join(k0, q0), input_precision="ieee"))
    d = (v0 - rk0) * bt0
    state = tl.dot(d[:, None], k0[None, :], acc=st, input_precision="ieee")
    if INPLACE:
        if s0 > 0:
            tl.store(final_state_ptr + s0 * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    else:
        tl.store(final_state_ptr + bos * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    tl.store(out_ptr, rq0 + d * tl.sum(k0 * q0, 0))
    st = state * g1[None, :]
    rk1, rq1 = tl.split(tl.dot(st, tl.join(k1, q1), input_precision="ieee"))
    d = (v1 - rk1) * bt1
    state = tl.dot(d[:, None], k1[None, :], acc=st, input_precision="ieee")
    if INPLACE:
        if s1 > 0:
            tl.store(final_state_ptr + s1 * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    else:
        tl.store(final_state_ptr + (bos + 1) * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    tl.store(out_ptr + HV * DV, rq1 + d * tl.sum(k1 * q1, 0))
    st = state * g2[None, :]
    rk2, rq2 = tl.split(tl.dot(st, tl.join(k2, q2), input_precision="ieee"))
    d = (v2 - rk2) * bt2
    state = tl.dot(d[:, None], k2[None, :], acc=st, input_precision="ieee")
    if INPLACE:
        if s2 > 0:
            tl.store(final_state_ptr + s2 * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    else:
        tl.store(final_state_ptr + (bos + 2) * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    tl.store(out_ptr + 2 * HV * DV, rq2 + d * tl.sum(k2 * q2, 0))
    st = state * g3[None, :]
    rk3, rq3 = tl.split(tl.dot(st, tl.join(k3, q3), input_precision="ieee"))
    d = (v3 - rk3) * bt3
    state = tl.dot(d[:, None], k3[None, :], acc=st, input_precision="ieee")
    if INPLACE:
        if s3 > 0:
            tl.store(final_state_ptr + s3 * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    else:
        tl.store(final_state_ptr + (bos + 3) * HV * DV * DK + state_offset, state, cache_modifier=".cg")
    tl.store(out_ptr + 3 * HV * DV, rq3 + d * tl.sum(k3 * q3, 0))


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
    total_tasks = n * HV * nv
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    per_program = (total_tasks + num_programs - 1) // num_programs
    begin = pid * per_program
    end = tl.minimum(begin + per_program, total_tasks)
    if (end - begin) == 4:
        for i in tl.static_range(4):
            _update_tile(
                begin + i,
                q_ptr,
                k_ptr,
                v_ptr,
                a_ptr,
                b_ptr,
                a_log_ptr,
                dt_bias_ptr,
                initial_state_ptr,
                final_state_ptr,
                output_ptr,
                cu_seqlens_ptr,
                state_indices_ptr,
                accepted_tokens_ptr,
                H,
                HV,
                DK,
                DV,
                STRIDE_INDICES_SEQ,
                STRIDE_INDICES_TOK,
                SCALE,
                LOWER_BOUND,
                SPEC,
                INPLACE,
                BV,
                BK,
            )
    else:
        for flat in range(begin, end):
            _update_tile(
                flat,
                q_ptr,
                k_ptr,
                v_ptr,
                a_ptr,
                b_ptr,
                a_log_ptr,
                dt_bias_ptr,
                initial_state_ptr,
                final_state_ptr,
                output_ptr,
                cu_seqlens_ptr,
                state_indices_ptr,
                accepted_tokens_ptr,
                H,
                HV,
                DK,
                DV,
                STRIDE_INDICES_SEQ,
                STRIDE_INDICES_TOK,
                SCALE,
                LOWER_BOUND,
                SPEC,
                INPLACE,
                BV,
                BK,
            )


# Four-token KDA verification adapted from vllm_mlu kda_decode_chunk.py
# at commit 0029935c161e38d3979a67593bc5cacc1fffd939.
# Copyright (C) 2025-2026 Cambricon.


@triton.jit
def _token_decay(
    raw_gate: tl.tensor,
    gate_bias: tl.tensor,
    gate_scale: tl.tensor,
    vg_base: tl.tensor,
    rk: tl.tensor,
    tok: tl.constexpr,
    length: tl.int32,
    HV: tl.constexpr,
    K: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
) -> tl.tensor:
    g_t = tl.load(raw_gate + vg_base + tok * HV * K + rk, mask=tok < length, other=0).to(tl.float32)
    return tl.where(
        tok < length,
        tl.exp(LOWER_BOUND * tl.sigmoid(gate_scale * (g_t + gate_bias))),
        1.0,
    )


@triton.jit
def _snapshot_store(snap: tl.tensor, sidx: tl.tensor, snap_bp: tl.tensor, active: tl.tensor) -> None:
    if (sidx > 0) & (active != 0):
        tl.store(snap_bp, snap)


@triton.jit(do_not_specialize=["N", "TOTAL_TOKENS"])
def fused_chunk_kda_decode_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    raw_gate: tl.tensor,
    raw_beta: tl.tensor,
    output: tl.tensor,
    state: tl.tensor,
    cu_seqlens: tl.tensor,
    state_indices: tl.tensor,
    accepted_tokens: tl.tensor,
    a_log: tl.tensor,
    gate_bias: tl.tensor,
    scale: tl.float32,
    N: tl.int32,
    TOTAL_TOKENS: tl.int32,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
    stride_state_slot: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
) -> None:
    pid = tl.program_id(0)
    workers = tl.num_programs(0)
    value_tiles = tl.cdiv(V, BV)
    _rt4 = tl.arange(0, 4)
    eye = tl.where(_rt4[:, None] == _rt4[None, :], 1.0, 0.0)
    head_tile = pid % (HV * value_tiles)
    first_n = pid // (HV * value_tiles)
    seq_step = workers // (HV * value_tiles)
    i_v = head_tile % value_tiles
    i_hv = head_tile // value_tiles
    i_h = i_hv // (HV // H)
    rk = tl.arange(0, K)
    rt = tl.arange(0, 4)
    rv = i_v * BV + tl.arange(0, BV)
    gate_scale = tl.exp(tl.load(a_log + i_h, cache_modifier=".ca").to(tl.float32))
    g_bias = tl.load(gate_bias + i_h * K + rk, cache_modifier=".ca").to(tl.float32)
    bos = tl.load(cu_seqlens + first_n)
    eos = tl.load(cu_seqlens + first_n + 1)
    first_column = tl.load(accepted_tokens + first_n) - 1
    state_idx = tl.load(
        state_indices + first_n * stride_indices_seq + tl.minimum(tl.maximum(first_column, 0), 3) * stride_indices_tok
    )
    state_idx = tl.where((first_column >= 0) & (first_column < 4), state_idx, 0)
    for i_n in range(first_n, N, seq_step):
        next_n = tl.minimum(i_n + seq_step, N - 1)
        bos_next = tl.load(cu_seqlens + next_n)
        eos_next = tl.load(cu_seqlens + next_n + 1)
        next_column = tl.load(accepted_tokens + next_n) - 1
        state_idx_next = tl.load(
            state_indices + next_n * stride_indices_seq + tl.minimum(tl.maximum(next_column, 0), 3) * stride_indices_tok
        )
        state_idx_next = tl.where((next_column >= 0) & (next_column < 4), state_idx_next, 0)
        length = eos - bos
        active = ((state_idx > 0) & (length > 0)).to(tl.int32)
        state_slot = state_idx * active
        qk_base = (bos * H + i_h) * K
        vg_base = (bos * HV + i_hv) * V
        q_kt = tl.load(
            q + qk_base + rk[:, None] + rt[None, :] * (H * K),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        k_kt = tl.load(
            k + qk_base + rk[:, None] + rt[None, :] * (H * K),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        v_vt = tl.load(
            v + vg_base + rv[:, None] + rt[None, :] * (HV * V),
            mask=rt[None, :] < length,
            other=0,
        ).to(tl.float32)
        beta_t = tl.load(raw_beta + bos * HV + i_hv + rt * HV, mask=rt < length, other=0).to(tl.float32)
        q_inv = 1.0 / tl.sqrt(tl.sum(q_kt * q_kt, axis=0) + 1e-06)
        k_inv = 1.0 / tl.sqrt(tl.sum(k_kt * k_kt, axis=0) + 1e-06)
        qn = q_kt * (q_inv * scale)[None, :]
        kn = k_kt * k_inv[None, :]
        dec0 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 0, length, HV, K, LOWER_BOUND)
        dec1 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 1, length, HV, K, LOWER_BOUND)
        dec2 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 2, length, HV, K, LOWER_BOUND)
        dec3 = _token_decay(raw_gate, g_bias, gate_scale, vg_base, rk, 3, length, HV, K, LOWER_BOUND)
        lam0 = dec0
        lam1 = lam0 * dec1
        lam2 = lam1 * dec2
        lam3 = lam2 * dec3
        lam = tl.where(
            rt[None, :] == 0,
            lam0[:, None],
            tl.where(
                rt[None, :] == 1,
                lam1[:, None],
                tl.where(rt[None, :] == 2, lam2[:, None], lam3[:, None]),
            ),
        )
        qs = lam * qn
        kb = lam * kn
        km = kn / lam
        state_load_bp = tl.make_block_ptr(
            base=state + state_slot * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        recurrent = tl.load(state_load_bp).to(tl.float32)
        km4 = tl.trans(km)
        sidx_vec = tl.load(state_indices + i_n * stride_indices_seq + rt * stride_indices_tok)
        sid0 = tl.sum(tl.where(rt == 0, sidx_vec, 0), axis=0)
        sid1 = tl.sum(tl.where(rt == 1, sidx_vec, 0), axis=0)
        sid2 = tl.sum(tl.where(rt == 2, sidx_vec, 0), axis=0)
        sid3 = tl.sum(tl.where(rt == 3, sidx_vec, 0), axis=0)
        bp0 = tl.make_block_ptr(
            base=state + tl.where(sid0 > 0, sid0, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp1 = tl.make_block_ptr(
            base=state + tl.where(sid1 > 0, sid1, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp2 = tl.make_block_ptr(
            base=state + tl.where(sid2 > 0, sid2, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        bp3 = tl.make_block_ptr(
            base=state + tl.where(sid3 > 0, sid3, 0) * stride_state_slot + i_hv * V * K,
            shape=(V, K),
            strides=(K, 1),
            offsets=(i_v * BV, 0),
            block_shape=(BV, K),
            order=(1, 0),
        )
        ut_qk_t = tl.dot(km4, qs, input_precision="tf32")
        ut_qk_t = tl.where(rt[None, :] >= rt[:, None], ut_qk_t, 0.0)
        ut_kk = tl.dot(tl.trans(kb), km, input_precision="ieee")
        beta_s = tl.where(rt < length, tl.sigmoid(beta_t), 0.0)
        lower = tl.where(rt[None, :] < rt[:, None], beta_s[:, None] * ut_kk, 0.0)
        lower2 = tl.dot(lower, lower, input_precision="ieee")
        lower3 = tl.dot(lower2, lower, input_precision="ieee")
        tri_inv = eye - lower + lower2 - lower3
        tri_inv_t = tl.trans(tri_inv)
        s0_kb = tl.dot(recurrent, kb, input_precision="ieee")
        s0_qs = tl.dot(recurrent, qs, input_precision="tf32")
        rhs = (v_vt - s0_kb) * beta_s[None, :]
        delta = tl.dot(rhs, tri_inv_t, input_precision="ieee")
        out_vt = s0_qs + tl.dot(delta, ut_qk_t, input_precision="tf32")
        # The MLU backend requires a uniform guard for this store.
        if active != 0:
            tl.store(
                output + (bos * HV + i_hv) * V + rv[:, None] + rt[None, :] * (HV * V),
                out_vt.to(output.dtype.element_ty),
                mask=rt[None, :] < length,
            )
        d0 = tl.where(rt[None, :] <= 0, delta, 0.0)
        a0 = tl.dot(d0, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam0[None, :] * a0, sid0, bp0, active & (length > 0))
        d1 = tl.where(rt[None, :] <= 1, delta, 0.0)
        a1 = tl.dot(d1, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam1[None, :] * a1, sid1, bp1, active & (length > 1))
        d2 = tl.where(rt[None, :] <= 2, delta, 0.0)
        a2 = tl.dot(d2, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam2[None, :] * a2, sid2, bp2, active & (length > 2))
        a3 = tl.dot(delta, km4, acc=recurrent, input_precision="ieee")
        _snapshot_store(lam3[None, :] * a3, sid3, bp3, active & (length > 3))
        bos = bos_next
        eos = eos_next
        state_idx = state_idx_next
