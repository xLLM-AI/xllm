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

# This kernel is adapted from vLLM's FLA Triton ops:
# https://github.com/vllm-project/vllm/tree/v0.18.0/vllm/model_executor/layers/fla/ops
# Upstream license: Apache License, Version 2.0.
# Modified for xLLM MLU TMO integration.

import triton
import triton.language as tl


@triton.jit
def tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    o,
    h0,
    ht,
    ssm_state_indices,
    scale,
    stride_mixed_qkv_tok: tl.constexpr,
    stride_a_tok: tl.constexpr,
    stride_b_tok: tl.constexpr,
    stride_init_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_seq: tl.constexpr,
    B,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_HV: tl.constexpr,
    BLOCK_V: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    SPLIT_HV: tl.constexpr = 0,  # 1=split HV across cores (small B); 0=serial HV in-core (large B, reuse preload)
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_v = tl.cdiv(V, BLOCK_V)
    num_hv = tl.cdiv(HV, BLOCK_HV)
    N = B
    # SPLIT_HV: small B includes the HV-block dim in flat space (high), splitting across cores to fill them;
    # large B falls back to splitting only N+V (HV iterates serially in-core, reusing scalar preload of the same n-group, faster).
    if SPLIT_HV:
        TOTAL_BLOCKS = num_v * num_hv * N
    else:
        TOTAL_BLOCKS = num_v * N

    A_log_vals = tl.load(A_log + tl.arange(0, HV)).to(tl.float32)
    dt_bias_vals = tl.load(dt_bias + tl.arange(0, HV)).to(tl.float32)

    HEADS_PER_Q: tl.constexpr = HV // H
    BH: tl.constexpr = (BLOCK_HV + HEADS_PER_Q - 1) // HEADS_PER_Q

    ones = tl.full((1, BLOCK_K), 1, tl.float32)
    neg_ones = tl.full((1, BLOCK_K), -1, tl.float32)

    # Split N into per-core segments (persistent loop); last core takes the remainder.
    num_percore = TOTAL_BLOCKS // num_jobs
    num_acturepercore = num_percore
    if pid == num_jobs - 1:
        num_acturepercore = TOTAL_BLOCKS - num_percore * (num_jobs - 1)

    o_k = tl.arange(0, BLOCK_K)
    mask_k = o_k < K

    o_N_blk = tl.arange(0, BLOCK_N)

    zeros = tl.zeros((BLOCK_V,), dtype=tl.float32).to(o.dtype.element_ty)
    # SPLIT_HV: compute this core's HV-block range [lo, hi) for the outer for i_hv_blk;
    # non-SPLIT_HV: outer i_hv_blk runs once (hv_blk_hi=1), HV driven by inner for hv_start over full range (reuse scalar preload).
    if SPLIT_HV:
        lo = num_percore * pid
        hi = lo + num_acturepercore
        hv_blk_lo = lo // (num_v * N)
        hv_blk_hi = tl.cdiv(hi, num_v * N)
    else:
        hv_blk_lo = 0
        hv_blk_hi = 1
    # This core's flat range [start, end). Variable-step while replaced by fixed-step nested for (better software pipelining).
    # N low, V-tile higher, HV-tile higher (SPLIT_HV): within one (i_hv_blk, i_v) tile, BLOCK_N consecutive work items
    # never cross a V-tile/HV-block boundary (preserves the C3 invariant).
    start = num_percore * pid
    end = start + num_acturepercore
    # Outer for i_hv_blk: SPLIT_HV iterates this core's HV-block range (each flat item handles its own 1 HV block,
    # fixing the old while-version bug "iterate all-core HV → state updated in-place multiple times across HV blocks");
    # non-SPLIT_HV runs range(0,1) once, HV driven by inner for hv_start over full range (reuse scalar preload).
    for i_hv_blk in range(hv_blk_lo, hv_blk_hi, 1):
        # This i_hv_blk's flat sub-range [seg_lo, seg_hi):
        #   non-split: the whole core range [start, end) (i_hv_blk not involved);
        #   split: intersection of core range with i_hv_blk slice [i_hv_blk*(num_v*N), (i_hv_blk+1)*(num_v*N)).
        if SPLIT_HV:
            seg_lo = tl.maximum(start, i_hv_blk * num_v * N)
            seg_hi = tl.minimum(end, (i_hv_blk + 1) * num_v * N)
        else:
            seg_lo = start
            seg_hi = end
        # V-tile sub-range (N low, i_v high).
        i_v_start = seg_lo // N
        i_v_end = (seg_hi - 1) // N
        for i_v in range(i_v_start, i_v_end + 1, 1):
            # First segment's n_lo may be != 0 (core/slice start lands mid V-tile); later segments have n_lo = 0.
            n_lo = tl.where(i_v == i_v_start, seg_lo - i_v_start * N, 0)
            # N upper bound: min of V-tile boundary N and this sub-range remainder (seg_hi - i_v*N).
            n_hi = tl.minimum(N, seg_hi - i_v * N)
            for i_n_base in range(n_lo, n_hi, BLOCK_N):
                i_n = i_n_base  # this segment's N start (body uses n_blk = i_n + o_N_blk as base address)
                numN = tl.minimum(BLOCK_N, n_hi - i_n_base)
                end_N = i_n + numN
                n_blk = i_n + o_N_blk
                mask_n_blk = (n_blk < end_N) & (n_blk < N)

                # o_v needs the LOCAL V-tile index within this HV slice: non-split i_v is local;
                # split i_v is the global flat V index (= i_hv_blk*num_v + local i_v), must strip i_hv_blk*num_v,
                # else o_v exceeds V → mask_v all False → vs/state read 0, wrong output.
                if SPLIT_HV:
                    i_v_local = i_v - i_hv_blk * num_v
                else:
                    i_v_local = i_v
                o_v = i_v_local * BLOCK_V + tl.arange(0, BLOCK_V)
                mask_v = o_v < V
                mask_h = mask_v[:, None] & mask_k[None, :]

                # state_idxs stays int32: ssm_state_indices is an int32 tensor and the values are cache-slot
                # indices that fit int32. (Upstream uses int64; int32 is the xllm convention and halves the
                # per-element load width of the [BLOCK_N, HV] scalar tables.)
                state_idxs = tl.load(ssm_state_indices + n_blk * stride_indices_seq, mask=mask_n_blk).to(tl.int32)
                a_vals = tl.load(
                    a + (n_blk * stride_a_tok)[:, None] + tl.arange(0, HV)[None, :], mask=mask_n_blk[:, None]
                )
                b_vals = tl.load(
                    b + (n_blk * stride_b_tok)[:, None] + tl.arange(0, HV)[None, :], mask=mask_n_blk[:, None]
                ).to(tl.float32)

                # Precompute scalar tables [BLOCK_N, HV]; inner loop indexes [i_n, i_hv] (aligned with fused_sigmoid_gating's b_gs/b_beta)
                xs = a_vals + dt_bias_vals[None, :]
                softplus_xs = tl.where(xs <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(xs)), xs)
                g_vals = -tl.exp(A_log_vals)[None, :] * softplus_xs  # [BLOCK_N, HV]
                beta_vals = tl.sigmoid(b_vals)  # [BLOCK_N, HV]
                exp_g_vals = tl.exp(g_vals)  # [BLOCK_N, HV], inner b_h *= exp_g_vals

                # HV blocking: load BLOCK_HV V-heads + their BH Q/K heads each iter; inner loop is scalar over local_i_hv.
                # Scalar tables (a/b/g/beta/exp_g/state_idxs) load all HV and are reused outside hv_start; inner uses gi_hv.
                # for hv_start range depends on SPLIT_HV:
                #   non-split: full [0, HV) — reuse the preload above (1 load serves num_hv blocks).
                #   split: only this i_hv_blk's 1 block [i_hv_blk*BLOCK_HV, +BLOCK_HV) — avoid cross-HV-block recounting.
                if SPLIT_HV:
                    hv_start_lo = i_hv_blk * BLOCK_HV
                    hv_start_hi = i_hv_blk * BLOCK_HV + BLOCK_HV
                else:
                    hv_start_lo = 0
                    hv_start_hi = HV
                for hv_start in range(hv_start_lo, hv_start_hi, BLOCK_HV):
                    o_hv_block = tl.arange(0, BLOCK_HV) + hv_start  # [BLOCK_HV] global V-head offset
                    mask_hvb = o_hv_block < HV
                    o_h_block = tl.arange(0, BH) + (hv_start // HEADS_PER_Q)  # [BH] global Q/K head offset
                    mask_hb = o_h_block < H
                    p_qs = (
                        mixed_qkv
                        + n_blk[:, None, None] * stride_mixed_qkv_tok
                        + o_h_block[None, :, None] * K
                        + o_k[None, None, :]
                    )
                    p_ks = (
                        mixed_qkv
                        + n_blk[:, None, None] * stride_mixed_qkv_tok
                        + (H * K)
                        + o_h_block[None, :, None] * K
                        + o_k[None, None, :]
                    )
                    mask_qks = mask_n_blk[:, None, None] & (o_k[None, None, :] < K) & mask_hb[None, :, None]
                    qs = tl.load(p_qs, mask=mask_qks, other=0).to(tl.float32)  # [BLOCK_N, BH, BK]
                    ks = tl.load(p_ks, mask=mask_qks, other=0).to(tl.float32)  # [BLOCK_N, BH, BK]
                    if USE_QK_L2NORM_IN_KERNEL:
                        # L2-normalize q/k over the K dim via a single 2D reshape+dot (aligned with fused_sigmoid_gating L193-196).
                        qs = qs.reshape((BLOCK_N * BH, BLOCK_K))
                        qs_mean = tl.rsqrt(tl.dot(ones, (qs * qs).trans(), allow_tf32=False) + 1e-6)
                        qs = qs * tl.dot(qs_mean.reshape((BLOCK_N * BH, 1)), ones, allow_tf32=False)
                        ks = ks.reshape((BLOCK_N * BH, BLOCK_K))
                        ks_mean = tl.rsqrt(tl.dot(ones, (ks * ks).trans(), allow_tf32=False) + 1e-6)
                        ks = ks * tl.dot(ks_mean.reshape((BLOCK_N * BH, 1)), ones, allow_tf32=False)
                        qs = qs.reshape((BLOCK_N, BH, BLOCK_K))
                        ks = ks.reshape((BLOCK_N, BH, BLOCK_K))
                    qs = qs * scale
                    p_vs = (
                        mixed_qkv
                        + n_blk[:, None, None] * stride_mixed_qkv_tok
                        + (2 * H * K)
                        + o_hv_block[None, :, None] * V
                        + o_v[None, None, :]
                    )
                    mask_vs_blk = mask_n_blk[:, None, None] & (o_v[None, None, :] < V) & mask_hvb[None, :, None]
                    vs = tl.load(p_vs, mask=mask_vs_blk, other=0).to(tl.float32)  # [BLOCK_N, BLOCK_HV, BLOCK_V]
                    b_os = tl.empty((BLOCK_N, BLOCK_HV, BLOCK_V), dtype=tl.float32)

                    # Inner loop processes all i_hv of the same N segment consecutively; mixed_qkv[i_n] rows stay cached.
                    for i_n_off in range(0, numN, 1):
                        state_idx = state_idxs[i_n_off]
                        for local_i_hv in range(0, BLOCK_HV, 1):
                            gi_hv = (
                                hv_start + local_i_hv
                            )  # global V-head index (indexes full-HV scalar tables / state ptr)
                            i_h = (
                                local_i_hv // HEADS_PER_Q
                            )  # Q/K head index within block, [0,BH) (matches qs/ks BH dim)

                            if gi_hv < HV:
                                # output_offsets = (i_n * HV + i_hv) * V + o_v
                                state_base_offset = gi_hv * V * K
                                state_offsets = state_base_offset + o_v[:, None] * K + o_k[None, :]

                                # Invalid state index (NULL_BLOCK_ID=0) only writes zero for this block.
                                if state_idx <= 0:
                                    b_os[i_n_off, local_i_hv, :] = zeros
                                else:
                                    p_h0 = h0 + state_idx * stride_init_state_token
                                    p_h0 = p_h0 + state_offsets
                                    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

                                    b_q = qs[i_n_off, i_h, :]
                                    b_k = ks[i_n_off, i_h, :]
                                    b_v = vs[i_n_off, local_i_hv, :]

                                    b_h *= exp_g_vals[i_n_off, gi_hv]
                                    b_v += tl.dot(neg_ones, (b_h * b_k[None, :]).trans(), allow_tf32=False)
                                    b_v *= beta_vals[i_n_off, gi_hv]
                                    b_h += tl.dot(b_v.reshape([BLOCK_V, 1]), ones, allow_tf32=False) * b_k[None, :]
                                    b_os[i_n_off, local_i_hv, :] = (
                                        tl.dot(ones, (b_h * b_q[None, :]).trans(), allow_tf32=False)
                                    ).reshape(
                                        [
                                            BLOCK_V,
                                        ]
                                    )

                                    p_ht = ht + state_idx * stride_final_state_token
                                    p_ht = p_ht + state_offsets
                                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
                            else:
                                b_os[i_n_off, local_i_hv, :] = zeros

                    # n_blk * (HV*V) + o_hv_block * V + i_v*BLOCK_V + o_v_block
                    p_os = o + n_blk[:, None, None] * (HV * V) + o_hv_block[None, :, None] * V + o_v[None, None, :]
                    tl.store(p_os, b_os.to(p_os.dtype.element_ty), mask=mask_vs_blk)
