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

import triton
import triton.language as tl


# @triton.jit(do_not_specialize=["N", "T"])
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
    BLOCK_HV: tl.constexpr,  # HV tile size: load BLOCK_HV V heads and the corresponding Q/K heads each time
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
    KDA_USE_SAFE_GATE: tl.constexpr,
    KDA_GATE_LOWER_BOUND,
    SPLIT_HV: tl.constexpr,
    BLOCK_N: tl.constexpr = 4,  # N-dimension tile size. It should distribute N evenly across all cores; larger values are preferable
    BLOCK_QUERY_LEN: tl.constexpr = 4,  # Token tile size per sequence. It must equal T, the maximum token count of one sequence
    FACTORED_REDUCE: tl.constexpr = False,
) -> None:
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)

    if not IS_KDA:
        rangeV = tl.arange(0, HV)
        # xLLM may pass bf16 A_log, while MLU tl.exp requires fp32 input.
        A_logs = tl.load(A_log + rangeV).to(tl.float32)
        dt_bias_vals = tl.load(dt_bias + rangeV)  # [HV,]

    NK, NV = triton.cdiv(K, BK), triton.cdiv(V, BV)
    NUM_HV_BLOCKS: tl.constexpr = triton.cdiv(HV, BLOCK_HV)
    if SPLIT_HV:
        TOTAL_BLOCKS = NK * NV * N * NUM_HV_BLOCKS
    else:
        TOTAL_BLOCKS = NK * NV * N
    num_percore = TOTAL_BLOCKS // num_jobs
    core_start = num_percore * pid
    num_acturepercore = num_percore
    if IS_KDA:
        # Distribute the remainder across cores instead of serializing it on
        # the final core when the active batch is not a multiple of the grid.
        remainder = TOTAL_BLOCKS % num_jobs
        num_acturepercore += (pid < remainder).to(tl.int64)
        core_start += tl.minimum(pid, remainder)
    elif pid == num_jobs - 1:
        num_acturepercore = TOTAL_BLOCKS - num_percore * (num_jobs - 1)

    o_T = tl.arange(0, BLOCK_QUERY_LEN * BLOCK_N)
    HEADS_PER_Q: tl.constexpr = HV // H
    BH: tl.constexpr = BLOCK_HV // HEADS_PER_Q  # Number of Q/K heads corresponding to each HV tile

    id = tl.zeros([], tl.int32)
    while id < num_acturepercore:
        flat_pid = id + core_start
        if SPLIT_HV:
            i_hv_block = flat_pid // (NK * N * NV)
            flat_tile_pid = flat_pid % (NK * N * NV)
        else:
            i_hv_block = 0
            flat_tile_pid = flat_pid
        # Keep N in the low bits of flat: take BLOCK_N consecutive jobs along N within the same (K, V) tile column.
        # This fixes C3-a (missing tiles, duplicates, or races when NV > 1). K/V tiles occupy the high bits. NK is always 1, so i_k is always 0.
        i_k = flat_tile_pid % NK
        i_n = (flat_tile_pid // NK) % N
        i_v = flat_tile_pid // (NK * N)  # V-tile high bits

        o_k = i_k * BK + tl.arange(0, BK)
        o_v = i_v * BV + tl.arange(0, BV)

        # The sequence count for this core segment is the smaller of its remaining jobs and BLOCK_N.
        numN = tl.minimum(BLOCK_N, num_acturepercore - id)
        # Clamp again to N - i_n: when a group crosses i_n = N - 1, it enters the next V tile as flat % N wraps to 0.
        # Without this clamp, i_v differs within one group and the starting i_v is incorrectly forced on every job when NV > 1.
        # After clamping, all i_n values in [i_n, N) belong to one V tile; the next iteration starts at i_n = 0 in the next V tile.
        numN = tl.minimum(numN, N - i_n)
        end_N = i_n + numN

        rangeN = i_n + tl.arange(0, BLOCK_N)
        rangeS = i_n + tl.arange(0, BLOCK_N + 1)
        mask_N = (rangeN < end_N) & (rangeN < N)
        mask_S = (rangeS < end_N + 1) & (rangeS <= N)
        if IS_VARLEN:
            cu_seqlens_ = tl.load(cu_seqlens + rangeS, mask=mask_S).to(tl.int32)

        if USE_INITIAL_STATE:
            if IS_CONTINUOUS_BATCHING:
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
            # The global position of the first token in this segment is cu_seqlens_[0].
            max_block_query_len = cu_seqlens_[numN] - cu_seqlens_[0]
            start_T = cu_seqlens_[0]
        else:
            max_block_query_len = numN * T
            start_T = i_n * T

        # for b_token in range(0,tl.cdiv(max_block_query_len/BLOCK_QUERY_LEN),BLOCK_QUERY_LEN):
        mask_T = o_T < max_block_query_len

        mask_k = o_k < K
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]

        if SPLIT_HV:
            hv_start_lo = i_hv_block * BLOCK_HV
            hv_start_hi = hv_start_lo + BLOCK_HV
            b_offsets = hv_start_lo + tl.arange(0, BLOCK_HV)
            mask_ab = mask_T[:, None] & (b_offsets[None, :] < HV)
            b_vals = tl.load(b + (start_T + o_T)[:, None] * HV + b_offsets[None, :], mask=mask_ab).to(tl.float32)
            b_beta = tl.sigmoid(b_vals)  # [BL*BN, BLOCK_HV]
        else:
            hv_start_lo = 0
            hv_start_hi = HV
            mask_ab = mask_T[:, None] & (tl.arange(0, HV)[None, :] < HV)
            b_vals = tl.load(b + (start_T + o_T)[:, None] * HV + tl.arange(0, HV)[None, :], mask=mask_ab).to(
                tl.float32
            )  # [BL*BN, HV]
            b_beta = tl.sigmoid(b_vals)  # [BL*BN, HV]
        if not IS_KDA:
            a_vals = tl.load(a + (start_T + o_T)[:, None] * HV + tl.arange(0, HV)[None, :], mask=mask_ab).to(
                tl.float32
            )  # [BL*BN, HV]
            xs = a_vals + dt_bias_vals[None, :]  # [BL*BN, HV]
            softplus_xs = tl.where(
                beta * xs <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * xs)), xs
            )  # [BL*BN, HV]
            b_gs = tl.exp(
                -tl.exp(A_logs)[None, :] * softplus_xs
            )  # [BL*BN, HV], applied as b_h *= b_gs[i_t, gi_hv] in the inner loop

        # Tile HV by loading BLOCK_HV V heads and the corresponding BH Q/K heads per iteration.
        for hv_start in range(hv_start_lo, hv_start_hi, BLOCK_HV):
            # The arange endpoint must be a literal constexpr; add the runtime hv_start offset separately.
            o_h_block = tl.arange(0, BLOCK_HV // (HV // H)) + (hv_start // (HV // H))  # [BH]
            o_hv_block = tl.arange(0, BLOCK_HV) + hv_start  # [BLOCK_HV]
            mask_hb = o_h_block < H
            mask_hvb = o_hv_block < HV

            p_qs = q + (start_T + 0 + o_T)[:, None, None] * (H * K) + o_h_block[None, :, None] * K + o_k[None, None, :]
            p_ks = k + (start_T + 0 + o_T)[:, None, None] * (H * K) + o_h_block[None, :, None] * K + o_k[None, None, :]
            p_vs = (
                v + (start_T + 0 + o_T)[:, None, None] * (HV * V) + o_hv_block[None, :, None] * V + o_v[None, None, :]
            )
            mask_qks = mask_k[None, None, :] & mask_T[:, None, None] & mask_hb[None, :, None]
            mask_hvs = mask_v[None, None, :] & mask_T[:, None, None] & mask_hvb[None, :, None]
            qs = tl.load(p_qs, mask=mask_qks).to(tl.float32)  # [BLOCK_QUERY_LEN*BLOCK_N, BH, BK]
            ks = tl.load(p_ks, mask=mask_qks).to(tl.float32)  # [BLOCK_QUERY_LEN*BLOCK_N, BH, BK]
            vs = tl.load(p_vs, mask=mask_hvs).to(tl.float32)  # [BLOCK_QUERY_LEN*BLOCK_N, BLOCK_HV, BV]
            qs = qs.reshape((BLOCK_QUERY_LEN * BLOCK_N * BH, BK))
            ks = ks.reshape((BLOCK_QUERY_LEN * BLOCK_N * BH, BK))
            if USE_QK_L2NORM_IN_KERNEL:
                qs = qs * tl.rsqrt(tl.sum(qs * qs, 1) + 1e-6)[:, None]
                ks = ks * tl.rsqrt(tl.sum(ks * ks, 1) + 1e-6)[:, None]
            qs = qs * scale
            qs = qs.reshape((BLOCK_QUERY_LEN * BLOCK_N, BH, BK))
            ks = ks.reshape((BLOCK_QUERY_LEN * BLOCK_N, BH, BK))

            if IS_KDA:
                # Load a/dt_bias/A_log by o_hv_block and compute b_gs [BL*BN, BLOCK_HV, BK].
                mask_aks = mask_k[None, None, :] & mask_T[:, None, None] & mask_hvb[None, :, None]
                a_vals = tl.load(
                    a
                    + (start_T + 0 + o_T)[:, None, None] * (HV * K)
                    + o_hv_block[None, :, None] * K
                    + o_k[None, None, :],
                    mask=mask_aks,
                ).to(tl.float32)  # [BL*BN, BLOCK_HV, BK]
                dt_bias_b = tl.load(dt_bias + o_hv_block[:, None] * BK + tl.arange(0, BK)[None, :])  # [BLOCK_HV, BK]
                A_log_b = tl.load(A_log + o_hv_block).to(tl.float32)  # [BLOCK_HV]
                xs = a_vals + dt_bias_b[None, :, :]  # [BL*BN, BLOCK_HV, BK]
                if KDA_USE_SAFE_GATE:
                    log_gate = KDA_GATE_LOWER_BOUND * tl.sigmoid(tl.exp(A_log_b)[None, :, None] * xs)
                    b_gs = tl.exp(log_gate)
                else:
                    softplus_xs = tl.where(beta * xs <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * xs)), xs)
                    b_gs = tl.exp(-tl.exp(A_log_b)[None, :, None] * softplus_xs)

            cum_tokens = tl.zeros(
                [], tl.int64
            )  # Cumulative tokens across N, always using the actual num_tokens regardless of valid_state
            b_os = tl.empty_like(vs)
            p_os = (
                o + (start_T + 0 + o_T)[:, None, None] * (HV * V) + o_hv_block[None, :, None] * V + o_v[None, None, :]
            )
            # token_valid[i] is true when token i belongs to a valid sequence (state_idx > 0) and should be written.
            # Tokens in invalid sequences remain false and are not stored, leaving the wrapper-prezeroed output at zero.
            token_valid = tl.zeros((BLOCK_QUERY_LEN * BLOCK_N,), dtype=tl.int1)
            for i_n_off in range(0, numN, 1):
                if IS_VARLEN:
                    bos, eos = (cu_seqlens_[i_n_off], cu_seqlens_[i_n_off + 1])
                    num_tokens = eos - bos
                else:
                    bos = (i_n + i_n_off) * T
                    eos = bos + T
                    num_tokens = T
                num_tokens_real = num_tokens  # Actual token count used to advance cum_tokens

                # valid_state indicates whether this sequence has state_idx > 0; invalid sequences skip computation and writeback.
                valid_state = True
                state_idx = 0
                if USE_INITIAL_STATE and IS_CONTINUOUS_BATCHING:
                    if IS_SPEC_DECODING:
                        i_t_init = i_t_inits[i_n_off] - 1
                    else:
                        i_t_init = 0
                    # Load state index and check for invalid entries
                    state_idx = state_idxs[i_n_off, i_t_init]
                    # Skip if state index is invalid (NULL_BLOCK_ID=0)
                    valid_state = state_idx > 0
                    state_idx = tl.where(valid_state, state_idx, 0)
                    num_tokens = tl.where(valid_state, num_tokens, 0)  # Skip this sequence when state_idx <= 0

                if num_tokens == 0:
                    # no tokens to process for this sequence
                    pass
                else:
                    for local_i_hv in range(0, BLOCK_HV, 1):
                        gi_hv = hv_start + local_i_hv  # Global HV index for b_gs/b_beta/h0/ht
                        i_h = local_i_hv // HEADS_PER_Q  # Q/K head index within the tile

                        # p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
                        b_h = tl.zeros([BV, BK], dtype=tl.float32)
                        if USE_INITIAL_STATE:
                            if IS_CONTINUOUS_BATCHING:
                                p_h0 = h0 + state_idx * stride_init_state_token
                                p_h0 = p_h0 + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                b_h += tl.load(p_h0, mask=mask_h & valid_state, other=0)
                            else:
                                p_h0 = h0 + bos * HV * V * K
                                p_h0 = p_h0 + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                b_h += tl.load(p_h0, mask=mask_h, other=0)

                        for i_t_raw in range(0, num_tokens):
                            i_t = i_t_raw + cum_tokens
                            b_q = qs[i_t, i_h, i_k * BK : i_k * BK + BK]
                            b_k = ks[i_t, i_h, i_k * BK : i_k * BK + BK]
                            b_v = vs[i_t, local_i_hv, :]

                            # [BV, BK]
                            if not IS_KDA:
                                b_h *= b_gs[i_t, gi_hv]
                            else:
                                b_h *= (b_gs[i_t, local_i_hv, :])[None, :]
                            # [BV]
                            if FACTORED_REDUCE:
                                b_v -= tl.sum(tl.sum((b_h * b_k[None, :]).reshape((BV, 4, BK // 4)), 1), 1)
                            else:
                                b_v -= tl.sum(b_h * b_k[None, :], 1)
                            if SPLIT_HV:
                                b_v *= b_beta[i_t, local_i_hv]
                            else:
                                b_v *= b_beta[i_t, gi_hv]
                            # [BV, BK]
                            if FACTORED_REDUCE:
                                b_h = (
                                    b_h.reshape((BV, 4, BK // 4))
                                    + b_v[:, None, None] * b_k.reshape((4, BK // 4))[None, :, :]
                                ).reshape((BV, BK))
                            else:
                                b_h += b_v[:, None] * b_k[None, :]
                            # [BV]
                            if FACTORED_REDUCE:
                                b_os[i_t, local_i_hv, :] = tl.sum(
                                    tl.sum((b_h * b_q[None, :]).reshape((BV, 4, BK // 4)), 1), 1
                                )
                            else:
                                b_os[i_t, local_i_hv, :] = tl.sum(b_h * b_q[None, :], 1)

                            # keep the states for multi-query tokens
                            if INPLACE_FINAL_STATE:
                                final_state_idx = state_idxs[i_n_off, i_t_raw]
                                # Only store if state index is valid (not NULL_BLOCK_ID=0)
                                valid_final_state = final_state_idx > 0
                                final_state_idx = tl.where(valid_final_state, final_state_idx, 0)
                                p_ht = ht + final_state_idx * stride_final_state_token
                                p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h & valid_final_state)
                            else:
                                p_ht = ht + (bos + i_t_raw) * stride_final_state_token
                                p_ht = p_ht + gi_hv * V * K + o_v[:, None] * K + o_k[None, :]
                                tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                # Do not write tokens from invalid sequences with state_idx <= 0.
                token_valid = token_valid | (((o_T >= cum_tokens) & (o_T < cum_tokens + num_tokens_real)) & valid_state)
                cum_tokens += num_tokens_real

            tl.store(p_os, b_os.to(p_os.dtype.element_ty), mask=mask_hvs & token_valid[:, None, None])

        # Advance by the sequences processed on this core; numN is at most BLOCK_N and smaller at a V-tile boundary.
        # Cast the int64 numN to int32 to match id because loop-carried variable types must remain stable.
        id += numN.to(tl.int32)
