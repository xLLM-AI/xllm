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

# This kernel is adapted from vLLM fused_sigmoid_gating.py.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# GLM5 KDA specialization: four-head tiles, FP32 recurrent state, and
# joint key/query projections. C++ selects this path before execution.
# NULL initial slots leave the prezeroed output and state pool untouched.
# NULL destinations suppress only the checkpoint store, never the recurrence.

import triton
import triton.language as tl


@triton.jit
def _glm_kda_window(
    A_log: tl.tensor,
    a: tl.tensor,
    b: tl.tensor,
    dt_bias: tl.tensor,
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    o: tl.tensor,
    h0: tl.tensor,
    ht: tl.tensor,
    scale: float,
    state_idxs: tl.tensor,
    i_t_inits: tl.tensor,
    cu_seqlens_: tl.tensor,
    o_k: tl.tensor,
    o_v: tl.tensor,
    i_hv_block: tl.tensor,
    i_n: tl.tensor,
    numN: tl.tensor,
    T: tl.tensor,
    KDA_GATE_LOWER_BOUND: float,
    H: tl.constexpr,
    HV: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    FACTORED_REDUCE: tl.constexpr,
    BLOCK_QUERY_LEN: tl.constexpr,
) -> None:
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    # State stays resident across windows, including NULL checkpoint boundaries.
    local_t = tl.arange(0, BLOCK_QUERY_LEN)
    first_head = i_hv_block * BLOCK_HV
    last_head = tl.minimum(first_head + BLOCK_HV, HV)
    for sequence in range(numN):
        if IS_VARLEN:
            bos = cu_seqlens_[sequence]
            eos = cu_seqlens_[sequence + 1]
        else:
            bos = (i_n + sequence) * T
            eos = bos + T
        if IS_SPEC_DECODING:
            accepted_index = i_t_inits[sequence] - 1
        else:
            accepted_index = 0
        initial_slot = state_idxs[sequence, accepted_index].to(tl.int64)
        valid_state = initial_slot > 0
        if valid_state:
            for head in range(first_head, last_head):
                q_head = head // (HV // H)
                state_base = initial_slot * stride_init_state_token
                state = tl.load(
                    h0 + state_base + head * V * K + o_v[:, None] * K + o_k[None, :], mask=mask_h, other=0
                ).to(tl.float32)
                bias = tl.load(dt_bias + head * K + o_k, mask=mask_k, other=0).to(tl.float32)
                rate = tl.exp(tl.load(A_log + head).to(tl.float32))
                for window_start in range(0, eos - bos, BLOCK_QUERY_LEN):
                    token = bos + window_start + local_t
                    valid_t = token < eos
                    qs = tl.load(
                        q + token[:, None] * H * K + q_head * K + o_k[None, :],
                        mask=valid_t[:, None] & mask_k[None, :],
                        other=0,
                    ).to(tl.float32)
                    ks = tl.load(
                        k + token[:, None] * H * K + q_head * K + o_k[None, :],
                        mask=valid_t[:, None] & mask_k[None, :],
                        other=0,
                    ).to(tl.float32)
                    vs = tl.load(
                        v + token[:, None] * HV * V + head * V + o_v[None, :],
                        mask=valid_t[:, None] & mask_v[None, :],
                        other=0,
                    ).to(tl.float32)
                    qs *= tl.rsqrt(tl.sum(qs * qs, 1) + 1e-06)[:, None]
                    ks *= tl.rsqrt(tl.sum(ks * ks, 1) + 1e-06)[:, None]
                    qs *= scale
                    raw = tl.load(
                        a + token[:, None] * HV * K + head * K + o_k[None, :],
                        mask=valid_t[:, None] & mask_k[None, :],
                        other=0,
                    ).to(tl.float32)
                    xs = raw + bias[None, :]
                    gates = tl.exp(KDA_GATE_LOWER_BOUND * tl.sigmoid(rate * xs))
                    betas = tl.sigmoid(tl.load(b + token * HV + head, mask=valid_t, other=0).to(tl.float32))
                    outputs = tl.zeros((BLOCK_QUERY_LEN, BV), tl.float32)
                    for local_index in range(tl.minimum(BLOCK_QUERY_LEN, eos - bos - window_start)):
                        query = qs[local_index, :]
                        key = ks[local_index, :]
                        state *= gates[local_index, :][None, :]
                        if FACTORED_REDUCE:
                            projected = tl.sum(tl.sum((state * key[None, :]).reshape((BV, 4, BK // 4)), 1), 1)
                        else:
                            projected = tl.sum(state * key[None, :], 1)
                        delta = (vs[local_index, :] - projected) * betas[local_index]
                        state += delta[:, None] * key[None, :]
                        if FACTORED_REDUCE:
                            outputs[local_index, :] = tl.sum(
                                tl.sum((state * query[None, :]).reshape((BV, 4, BK // 4)), 1), 1
                            )
                        else:
                            outputs[local_index, :] = tl.sum(state * query[None, :], 1)
                        global_index = window_start + local_index
                        if INPLACE_FINAL_STATE:
                            destination = state_idxs[sequence, global_index].to(tl.int64)
                            valid_destination = destination > 0
                            destination = tl.maximum(destination, 0)
                        else:
                            destination = (bos + global_index).to(tl.int64)
                            valid_destination = True
                        tl.store(
                            ht
                            + destination * stride_final_state_token
                            + head * V * K
                            + o_v[:, None] * K
                            + o_k[None, :],
                            state,
                            mask=mask_h & valid_destination,
                        )
                    tl.store(
                        o + token[:, None] * HV * V + head * V + o_v[None, :],
                        outputs,
                        mask=valid_t[:, None] & mask_v[None, :],
                    )


@triton.jit
def fused_glm_kda_update_kernel(
    A_log: tl.tensor,
    a: tl.tensor,
    b: tl.tensor,
    dt_bias: tl.tensor,
    beta: float,
    threshold: float,
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    o: tl.tensor,
    h0: tl.tensor,
    ht: tl.tensor,
    cu_seqlens: tl.tensor,
    ssm_state_indices: tl.tensor,
    num_accepted_tokens: tl.tensor,
    scale: float,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    KDA_USE_SAFE_GATE: tl.constexpr,
    KDA_GATE_LOWER_BOUND: float,
    SPLIT_HV: tl.constexpr,
    BLOCK_N: tl.constexpr = 4,
    BLOCK_QUERY_LEN: tl.constexpr = 4,
    FACTORED_REDUCE: tl.constexpr = False,
    BOUNDED_QUERY: tl.constexpr = False,
) -> None:
    tl.static_assert(
        IS_KDA
        and SPLIT_HV
        and USE_INITIAL_STATE
        and IS_CONTINUOUS_BATCHING
        and USE_QK_L2NORM_IN_KERNEL
        and KDA_USE_SAFE_GATE
        and (BLOCK_N == 1)
        and (BLOCK_HV == 4)
        and (H == 8)
        and (HV == 8)
        and (BK == 128)
        and (BV == 128)
        and (K == 128)
        and (V == 128)
    )
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    NK, NV = (triton.cdiv(K, BK), triton.cdiv(V, BV))
    NUM_HV_BLOCKS: tl.constexpr = triton.cdiv(HV, BLOCK_HV)
    TOTAL_BLOCKS = NK * NV * N * NUM_HV_BLOCKS
    num_percore = TOTAL_BLOCKS // num_jobs
    num_actual_per_core = num_percore
    remainder = TOTAL_BLOCKS % num_jobs
    num_actual_per_core += (pid < remainder).to(tl.int64)
    core_start = pid
    o_T = tl.arange(0, BLOCK_QUERY_LEN * BLOCK_N)
    HEADS_PER_Q: tl.constexpr = HV // H
    BH: tl.constexpr = BLOCK_HV // HEADS_PER_Q
    id = tl.zeros([], tl.int32)
    while id < num_actual_per_core:
        tl.static_assert(BLOCK_N == 1)
        flat_pid = id * num_jobs + core_start
        i_hv_block = flat_pid // (NK * N * NV)
        flat_tile_pid = flat_pid % (NK * N * NV)
        i_k = flat_tile_pid % NK
        i_n = flat_tile_pid // NK % N
        i_v = flat_tile_pid // (NK * N)
        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        numN = tl.minimum(BLOCK_N, num_actual_per_core - id)
        numN = tl.minimum(numN, N - i_n)
        end_N = i_n + numN
        rangeN = i_n + tl.arange(0, BLOCK_N)
        rangeS = i_n + tl.arange(0, BLOCK_N + 1)
        mask_N = (rangeN < end_N) & (rangeN < N)
        mask_S = (rangeS < end_N + 1) & (rangeS <= N)
        if IS_VARLEN:
            cu_seqlens_ = tl.load(cu_seqlens + rangeS, mask=mask_S).to(tl.int32)
        state_idxs = tl.zeros((BLOCK_N, 1), tl.int32)
        i_t_inits = tl.zeros((BLOCK_N,), tl.int32)
        cu_seqlens_ = tl.zeros((BLOCK_N + 1,), tl.int32) if not IS_VARLEN else cu_seqlens_
        state_idxs = tl.load(
            ssm_state_indices
            + (rangeN * stride_indices_seq)[:, None]
            + tl.arange(0, triton.next_power_of_2(stride_indices_seq))[None, :],
            mask=mask_N[:, None]
            & (tl.arange(0, triton.next_power_of_2(stride_indices_seq))[None, :] < stride_indices_seq),
            other=0,
        ).to(tl.int32)
        if IS_SPEC_DECODING:
            i_t_inits = tl.load(num_accepted_tokens + rangeN, mask=mask_N)
        if IS_VARLEN:
            max_block_query_len = cu_seqlens_[numN] - cu_seqlens_[0]
            start_T = cu_seqlens_[0]
        else:
            max_block_query_len = numN * T
            start_T = i_n * T
        if IS_KDA and (not BOUNDED_QUERY):
            _glm_kda_window(
                A_log,
                a,
                b,
                dt_bias,
                q,
                k,
                v,
                o,
                h0,
                ht,
                scale,
                state_idxs,
                i_t_inits,
                cu_seqlens_,
                o_k,
                o_v,
                i_hv_block,
                i_n,
                numN,
                T,
                KDA_GATE_LOWER_BOUND,
                H,
                HV,
                BK,
                BV,
                K,
                V,
                BLOCK_HV,
                IS_SPEC_DECODING,
                IS_VARLEN,
                INPLACE_FINAL_STATE,
                stride_init_state_token,
                stride_final_state_token,
                FACTORED_REDUCE,
                BLOCK_QUERY_LEN,
            )
        else:
            mask_T = o_T < max_block_query_len
            mask_k = o_k < K
            mask_v = o_v < V
            mask_h = mask_v[:, None] & mask_k[None, :]
            hv_start_lo = i_hv_block * BLOCK_HV
            hv_start_hi = hv_start_lo + BLOCK_HV
            b_offsets = hv_start_lo + tl.arange(0, BLOCK_HV)
            mask_ab = mask_T[:, None] & (b_offsets[None, :] < HV)
            b_vals = tl.load(b + (start_T + o_T)[:, None] * HV + b_offsets[None, :], mask=mask_ab).to(tl.float32)
            b_beta = tl.sigmoid(b_vals)
            b_beta = tl.trans(b_beta)
            for hv_start in range(hv_start_lo, hv_start_hi, BLOCK_HV):
                o_h_block = tl.arange(0, BLOCK_HV // (HV // H)) + hv_start // (HV // H)
                o_hv_block = tl.arange(0, BLOCK_HV) + hv_start
                mask_hb = o_h_block < H
                mask_hvb = o_hv_block < HV
                p_qs = (
                    q + (start_T + 0 + o_T)[:, None, None] * (H * K) + o_h_block[None, :, None] * K + o_k[None, None, :]
                )
                p_ks = (
                    k + (start_T + 0 + o_T)[:, None, None] * (H * K) + o_h_block[None, :, None] * K + o_k[None, None, :]
                )
                p_vs = (
                    v
                    + (start_T + 0 + o_T)[:, None, None] * (HV * V)
                    + o_hv_block[None, :, None] * V
                    + o_v[None, None, :]
                )
                mask_qks = mask_k[None, None, :] & mask_T[:, None, None] & mask_hb[None, :, None]
                mask_hvs = mask_v[None, None, :] & mask_T[:, None, None] & mask_hvb[None, :, None]
                qs = tl.load(p_qs, mask=mask_qks).to(tl.float32)
                ks = tl.load(p_ks, mask=mask_qks).to(tl.float32)
                vs = tl.load(p_vs, mask=mask_hvs).to(tl.float32)
                qs = qs.reshape((BLOCK_QUERY_LEN * BLOCK_N * BH, BK))
                ks = ks.reshape((BLOCK_QUERY_LEN * BLOCK_N * BH, BK))
                qs = qs * tl.rsqrt(tl.sum(qs * qs, 1) + 1e-06)[:, None]
                ks = ks * tl.rsqrt(tl.sum(ks * ks, 1) + 1e-06)[:, None]
                qs = qs * scale
                qs = qs.reshape((BLOCK_QUERY_LEN * BLOCK_N, BH, BK))
                ks = ks.reshape((BLOCK_QUERY_LEN * BLOCK_N, BH, BK))
                gate_mask = mask_k[None, None, :] & mask_T[:, None, None] & mask_hvb[None, :, None]
                a_vals = tl.load(
                    a
                    + (start_T + 0 + o_T)[:, None, None] * (HV * K)
                    + o_hv_block[None, :, None] * K
                    + o_k[None, None, :],
                    mask=gate_mask,
                ).to(tl.float32)
                dt_bias_b = tl.load(dt_bias + o_hv_block[:, None] * BK + tl.arange(0, BK)[None, :])
                A_log_b = tl.load(A_log + o_hv_block).to(tl.float32)
                xs = a_vals + dt_bias_b[None, :, :]
                log_gate = KDA_GATE_LOWER_BOUND * tl.sigmoid(tl.exp(A_log_b)[None, :, None] * xs)
                b_gs = tl.exp(log_gate)
                qs = tl.permute(qs, (1, 0, 2))
                ks = tl.permute(ks, (1, 0, 2))
                vs = tl.permute(vs, (1, 0, 2))
                b_gs = tl.permute(b_gs, (1, 0, 2))
                query_key_products = tl.sum(qs * ks, 2)
                cum_tokens = tl.zeros([], tl.int64)
                b_os = tl.zeros((BLOCK_HV, BLOCK_QUERY_LEN * BLOCK_N, BV), tl.float32)
                p_os = (
                    o
                    + (start_T + 0 + o_T)[:, None, None] * (HV * V)
                    + o_hv_block[None, :, None] * V
                    + o_v[None, None, :]
                )
                token_valid = tl.zeros((BLOCK_QUERY_LEN * BLOCK_N,), dtype=tl.int1)
                for i_n_off in range(0, numN, 1):
                    if IS_VARLEN:
                        bos, eos = (cu_seqlens_[i_n_off], cu_seqlens_[i_n_off + 1])
                        num_tokens = eos - bos
                    else:
                        bos = (i_n + i_n_off) * T
                        eos = bos + T
                        num_tokens = T
                    num_tokens_real = num_tokens
                    valid_state = True
                    state_idx = 0
                    if IS_SPEC_DECODING:
                        i_t_init = i_t_inits[i_n_off] - 1
                    else:
                        i_t_init = 0
                    state_idx = state_idxs[i_n_off, i_t_init].to(tl.int64)
                    valid_state = state_idx > 0
                    state_idx = tl.where(valid_state, state_idx, 0)
                    num_tokens = tl.where(valid_state, num_tokens, 0)
                    if num_tokens == 0:
                        pass
                    else:
                        for local_i_hv in range(0, BLOCK_HV, 1):
                            gi_hv = hv_start + local_i_hv
                            i_h = local_i_hv // HEADS_PER_Q
                            b_h = tl.zeros([BV, BK], dtype=tl.float32)
                            p_h0 = h0 + state_idx * stride_init_state_token
                            p_h0 = p_h0 + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            b_h += tl.load(p_h0, mask=mask_h & valid_state, other=0)
                            for pair_start in range(0, num_tokens, 4):
                                i_t_raw = pair_start + 0
                                if i_t_raw < num_tokens:
                                    i_t = i_t_raw + cum_tokens
                                    b_q = qs[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_k = ks[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_v = vs[local_i_hv, i_t, :]
                                    b_h *= b_gs[local_i_hv, i_t, :][None, :]
                                    columns = tl.arange(0, 2)
                                    rhs = tl.where(
                                        columns[:, None] == 0,
                                        b_k[None, :],
                                        tl.where(columns[:, None] == 1, b_q[None, :], 0.0),
                                    )
                                    projections = tl.dot(rhs, tl.trans(b_h), allow_tf32=False)
                                    projection_k = projections[0, :]
                                    projection_q = projections[1, :]
                                    b_v = b_v - projection_k
                                    b_v *= b_beta[local_i_hv, i_t]
                                    state_0 = b_h + b_v[:, None] * b_k[None, :]
                                    b_h = state_0
                                    b_os[local_i_hv, i_t, :] = projection_q + b_v * query_key_products[i_h, i_t]
                                    if INPLACE_FINAL_STATE:
                                        final_state_idx = state_idxs[i_n_off, i_t_raw].to(tl.int64)
                                        valid_final_state = final_state_idx > 0
                                        final_state_idx = tl.where(valid_final_state, final_state_idx, 0)
                                        p_ht = ht + final_state_idx * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_final_state)
                                    else:
                                        p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                                i_t_raw = pair_start + 1
                                if i_t_raw < num_tokens:
                                    i_t = i_t_raw + cum_tokens
                                    b_q = qs[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_k = ks[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_v = vs[local_i_hv, i_t, :]
                                    b_h *= b_gs[local_i_hv, i_t, :][None, :]
                                    columns = tl.arange(0, 2)
                                    rhs = tl.where(
                                        columns[:, None] == 0,
                                        b_k[None, :],
                                        tl.where(columns[:, None] == 1, b_q[None, :], 0.0),
                                    )
                                    projections = tl.dot(rhs, tl.trans(b_h), allow_tf32=False)
                                    projection_k = projections[0, :]
                                    projection_q = projections[1, :]
                                    b_v = b_v - projection_k
                                    b_v *= b_beta[local_i_hv, i_t]
                                    state_1 = b_h + b_v[:, None] * b_k[None, :]
                                    b_h = state_1
                                    b_os[local_i_hv, i_t, :] = projection_q + b_v * query_key_products[i_h, i_t]
                                    if INPLACE_FINAL_STATE:
                                        final_state_idx = state_idxs[i_n_off, i_t_raw].to(tl.int64)
                                        valid_final_state = final_state_idx > 0
                                        final_state_idx = tl.where(valid_final_state, final_state_idx, 0)
                                        p_ht = ht + final_state_idx * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_final_state)
                                    else:
                                        p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                                i_t_raw = pair_start + 2
                                if i_t_raw < num_tokens:
                                    i_t = i_t_raw + cum_tokens
                                    b_q = qs[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_k = ks[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_v = vs[local_i_hv, i_t, :]
                                    b_h *= b_gs[local_i_hv, i_t, :][None, :]
                                    columns = tl.arange(0, 2)
                                    rhs = tl.where(
                                        columns[:, None] == 0,
                                        b_k[None, :],
                                        tl.where(columns[:, None] == 1, b_q[None, :], 0.0),
                                    )
                                    projections = tl.dot(rhs, tl.trans(b_h), allow_tf32=False)
                                    projection_k = projections[0, :]
                                    projection_q = projections[1, :]
                                    b_v = b_v - projection_k
                                    b_v *= b_beta[local_i_hv, i_t]
                                    state_2 = b_h + b_v[:, None] * b_k[None, :]
                                    b_h = state_2
                                    b_os[local_i_hv, i_t, :] = projection_q + b_v * query_key_products[i_h, i_t]
                                    if INPLACE_FINAL_STATE:
                                        final_state_idx = state_idxs[i_n_off, i_t_raw].to(tl.int64)
                                        valid_final_state = final_state_idx > 0
                                        final_state_idx = tl.where(valid_final_state, final_state_idx, 0)
                                        p_ht = ht + final_state_idx * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_final_state)
                                    else:
                                        p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                                i_t_raw = pair_start + 3
                                if i_t_raw < num_tokens:
                                    i_t = i_t_raw + cum_tokens
                                    b_q = qs[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_k = ks[i_h, i_t, i_k * BK : i_k * BK + BK]
                                    b_v = vs[local_i_hv, i_t, :]
                                    b_h *= b_gs[local_i_hv, i_t, :][None, :]
                                    columns = tl.arange(0, 2)
                                    rhs = tl.where(
                                        columns[:, None] == 0,
                                        b_k[None, :],
                                        tl.where(columns[:, None] == 1, b_q[None, :], 0.0),
                                    )
                                    projections = tl.dot(rhs, tl.trans(b_h), allow_tf32=False)
                                    projection_k = projections[0, :]
                                    projection_q = projections[1, :]
                                    b_v = b_v - projection_k
                                    b_v *= b_beta[local_i_hv, i_t]
                                    state_3 = b_h + b_v[:, None] * b_k[None, :]
                                    b_h = state_3
                                    b_os[local_i_hv, i_t, :] = projection_q + b_v * query_key_products[i_h, i_t]
                                    if INPLACE_FINAL_STATE:
                                        final_state_idx = state_idxs[i_n_off, i_t_raw].to(tl.int64)
                                        valid_final_state = final_state_idx > 0
                                        final_state_idx = tl.where(valid_final_state, final_state_idx, 0)
                                        p_ht = ht + final_state_idx * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_final_state)
                                    else:
                                        p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                                        p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                    token_valid = token_valid | (o_T >= cum_tokens) & (o_T < cum_tokens + num_tokens_real) & valid_state
                    cum_tokens += num_tokens_real
                tl.store(
                    p_os,
                    tl.permute(b_os, (1, 0, 2)).to(p_os.dtype.element_ty),
                    mask=mask_hvs & token_valid[:, None, None],
                )
        id += numN.to(tl.int32)
