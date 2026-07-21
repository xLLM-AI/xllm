# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
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

import triton
import triton.language as tl


@triton.jit()
def tmo_fused_sigmoid_gating_delta_rule_update_kernel(
    A_log,
    a,
    b,
    dt_bias,
    beta,
    threshold,
    q,
    k,
    v,
    o,
    h0,
    ht,
    cu_seqlens,
    ssm_state_indices,
    num_accepted_tokens,
    scale,
    N: tl.int64,  # num of sequences
    T: tl.int64,  # num of tokens
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    INPLACE_FINAL_STATE: tl.constexpr,  # whether to store final state inplace
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
    MAX_N: tl.constexpr = 1024,
    BLOCK_N: tl.constexpr = 4,
    BLOCK_QUERY_LEN: tl.constexpr = 4,
    BLOCK_HV: tl.constexpr = 4,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    # Pre-load per-head scalars
    rangeN = tl.arange(0, MAX_N)
    rangeS = tl.arange(0, MAX_N + 1)
    mask_N = rangeN < N
    mask_S = rangeS < N + 1
    cu_seqlens_ = tl.load(cu_seqlens + rangeS, mask=mask_S).to(tl.int32)
    state_idxs = tl.load(ssm_state_indices + (rangeN * stride_indices_seq)[:, None] + tl.arange(0, stride_indices_seq)[None,:],
                          mask=mask_N[:, None]).to(tl.int32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t_inits = tl.load(num_accepted_tokens + rangeN, mask=mask_N)

    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NHV = triton.cdiv(HV, BLOCK_HV)
    TOTAL_BLOCKS = NK * NV * NHV * N
    num_percore = TOTAL_BLOCKS//num_jobs
    num_acturepercore = num_percore
    if pid == num_jobs-1:
        num_acturepercore=TOTAL_BLOCKS-num_percore*(num_jobs-1)

    ones = tl.full((1,BK),1,tl.float32)
    neg_ones = tl.full((1,BK),-1,tl.float32)

    o_hv_offsets = tl.arange(0, BLOCK_HV)
    o_T = tl.arange(0, BLOCK_QUERY_LEN*BLOCK_N)
    for id in range(0, num_acturepercore, BLOCK_N):
        flat_pid = id + num_percore*pid
        i_k = flat_pid % NK
        i_v = (flat_pid // NK) % NV
        i_hv_block = (flat_pid // (NK * NV)) % NHV
        i_n = flat_pid // (NK * NV * NHV)

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)
        hv_start = i_hv_block * BLOCK_HV
        o_hv = hv_start + o_hv_offsets
        mask_hv = o_hv < HV
        o_h = o_hv // (HV // H)

        A_logs = tl.load(A_log + o_hv, mask=mask_hv, other=0).to(tl.float32)
        if not IS_KDA:
            a_vals = tl.load(a + (rangeN * HV)[:, None] + o_hv[None, :],
                             mask=mask_N[:, None] & mask_hv[None, :],
                             other=0) # [MAX_N, BLOCK_HV]
            dt_bias_vals = tl.load(dt_bias + o_hv, mask=mask_hv, other=0) # [BLOCK_HV]
            xs = a_vals+dt_bias_vals[None, :]
            softplus_xs = tl.where(
                beta * xs <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * xs)), xs
            ) # [MAX_N, BLOCK_HV]
            b_gs = tl.exp(-tl.exp(A_logs)[None, :] * softplus_xs) # [MAX_N, BLOCK_HV]
        else:
            # a_vals = 0 #p_a = a + (bos * HV + i_hv) * K + o_k  # T,hv,k
            dt_bias_vals = tl.load(dt_bias + o_hv[:, None] * K + o_k[None, :],
                                   mask=mask_hv[:, None] & (o_k < K)[None, :],
                                   other=0) # [BLOCK_HV,BK]

        b_vals = tl.load(b + (rangeN * HV)[:, None] + o_hv[None, :],
                         mask=mask_N[:, None] & mask_hv[None, :],
                         other=0).to(tl.float32) # [MAX_N, BLOCK_HV]
        # compute beta_output = sigmoid(b)
        b_beta = tl.sigmoid(b_vals) # [MAX_N, BLOCK_HV]

        if IS_VARLEN:
            if pid==num_jobs-1 and num_acturepercore%BLOCK_N!=0 and num_acturepercore/BLOCK_N == id:
                max_block_query_len = cu_seqlens_[i_n + num_acturepercore%BLOCK_N]-cu_seqlens_[i_n]
                end_N = i_n + num_acturepercore%BLOCK_N
            else:
                max_block_query_len = cu_seqlens_[i_n + BLOCK_N]-cu_seqlens_[i_n]
                end_N = i_n + BLOCK_N
            all = T
            start_T = cu_seqlens_[i_n]
        else:
            max_block_query_len = T*BLOCK_N
            all = B * T
            start_T = i_n * T
            end_N = i_n + BLOCK_N

        # for b_token in range(0,tl.cdiv(max_block_query_len/BLOCK_QUERY_LEN),BLOCK_QUERY_LEN):
        mask_T = o_T<max_block_query_len

        p_qs = q + (start_T+0+o_T)[:,None,None]*(H*K) + o_h[None,:,None]*K + o_k[None,None,:]
        p_ks = k + (start_T+0+o_T)[:,None,None]*(H*K) + o_h[None,:,None]*K + o_k[None,None,:]
        p_vs = v + (start_T+0+o_T)[:,None,None]*(HV*V) + o_hv[None,:,None]*V + o_v[None,None,:]
        mask_k = o_k < K
        mask_ks = mask_k[None,None,:] & mask_T[:,None,None] & mask_hv[None,:,None]
        mask_v = o_v < V
        mask_vs = mask_v[None,None,:] & mask_T[:,None,None] & mask_hv[None,:,None]
        mask_h = mask_v[:, None] & mask_k[None, :]
        qs = tl.load(p_qs,mask=mask_ks).to(tl.float32) # [BLOCK_QUERY_LEN*BLOCK_N, BLOCK_HV, BK]
        ks = tl.load(p_ks,mask=mask_ks).to(tl.float32) # [BLOCK_QUERY_LEN*BLOCK_N, BLOCK_HV, BK]
        vs = tl.load(p_vs,mask=mask_vs).to(tl.float32) # [BLOCK_QUERY_LEN*BLOCK_N, BLOCK_HV, BV]
        if USE_QK_L2NORM_IN_KERNEL:
            qs = qs * (tl.rsqrt(tl.sum(qs * qs, axis=-1, keep_dims=True) + 1e-6))
            ks = ks * (tl.rsqrt(tl.sum(ks * ks, axis=-1, keep_dims=True) + 1e-6))
        qs = qs * scale

        if IS_KDA:
            a_vals = tl.load(a + (start_T+0+o_T)[:,None,None]*(HV*K) + o_hv[None,:,None]*K + o_k[None,None,:],mask=mask_ks) # [BLOCK_QUERY_LEN*BLOCK_N, HV, BK]
            xs = a_vals + dt_bias_vals[None, :, :] # [BLOCK_QUERY_LEN*BLOCK_N, HV, BK]
            softplus_xs = tl.where(
                beta * xs <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * xs)), xs
            )
            b_gs = -tl.exp(A_logs)[None, :, None] * softplus_xs

        b_os = tl.empty_like(vs)
        p_os = o + (start_T+0+o_T)[:,None,None]*(HV*V) + o_hv[None,:,None]*V + o_v[None,None,:]
        for i_n in range(i_n, end_N, 1):
            if IS_VARLEN:
                bos, eos = (
                    cu_seqlens_[i_n],cu_seqlens_[i_n+1]
                )
                num_tokens = eos - bos
            else:
                bos, eos = i_n * T, i_n * T + T
                num_tokens = T
            seq_token_offset = bos - start_T

            if num_tokens == 0:
                # no tokens to process for this sequence
                pass
            else:
                for i_hv_offset in range(0,BLOCK_HV,1):
                    i_hv = hv_start + i_hv_offset
                    i_h = i_hv // (HV // H)
                    valid_hv = i_hv < HV

                    # p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
                    b_h = tl.zeros([BV, BK], dtype=tl.float32)
                    if USE_INITIAL_STATE:
                        if IS_CONTINUOUS_BATCHING:
                            if IS_SPEC_DECODING:
                                i_t_init = i_t_inits[i_n] - 1
                            else:
                                i_t_init = 0
                            # Load state index and check for invalid entries
                            state_idx = state_idxs[i_n, i_t_init]
                            # Skip if state index is invalid (NULL_BLOCK_ID=0)
                            valid_state = state_idx > 0
                            state_idx = tl.where(valid_state, state_idx, 0)
                            num_tokens = tl.where(valid_state, num_tokens, 0)
                            p_h0 = h0 + state_idx * stride_init_state_token
                            p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            b_h += tl.load(p_h0, mask=mask_h & valid_state & valid_hv, other=0)
                        else:
                            p_h0 = h0 + bos * HV * V * K
                            p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            b_h += tl.load(p_h0, mask=mask_h & valid_hv, other=0)

                    for i_t_raw in range(0, num_tokens):
                        i_t = i_t_raw + seq_token_offset
                        b_q = qs[i_t,i_hv_offset,i_k*BK:i_k*BK+BK]
                        b_k = ks[i_t,i_hv_offset,i_k*BK:i_k*BK+BK]
                        b_v = vs[i_t,i_hv_offset,i_v*BV:i_v*BV+BV]

                        # [BV, BK]
                        if not IS_KDA:
                            b_h *= b_gs[i_n, i_hv_offset]
                        else:
                            b_h *= (b_gs[i_t,i_hv_offset, :])[None, :]
                        # [BV]
                        b_v += tl.dot(neg_ones,(b_h * b_k[None, :]).trans(), allow_tf32=False)
                        b_v *= b_beta[i_n, i_hv_offset]
                        # [BV, BK]
                        b_h += tl.dot(b_v.reshape([BV,1]),ones,allow_tf32=False) * b_k[None, :]
                        # [BV]
                        b_os[i_t,i_hv_offset,i_v*BV:i_v*BV+BV] = (tl.dot(ones,(b_h * b_q[None, :]).trans(),allow_tf32=False)).reshape([BV,])
                        # b_o = (tl.dot(ones,(b_h * b_q[None, :]).trans(),allow_tf32=False)).reshape([BV,])
                        # tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                        # keep the states for multi-query tokens
                        if INPLACE_FINAL_STATE:
                            # Load state index and check for invalid entries
                            final_state_idx = state_idxs[i_n, i_t_raw]
                            # Only store if state index is valid (not NULL_BLOCK_ID=0)
                            if final_state_idx > 0:
                                p_ht = ht + final_state_idx * stride_final_state_token
                                p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_hv)
                        else:
                            p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                            p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
                            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_hv)

        tl.store(p_os, b_os.to(p_os.dtype.element_ty), mask=mask_vs)
