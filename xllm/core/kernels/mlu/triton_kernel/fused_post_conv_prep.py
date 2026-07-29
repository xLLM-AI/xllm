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

import triton
import triton.language as tl


# ==========================================================
# Triton Kernel Code (from Step 3)
# ==========================================================

@triton.jit
def tmo_fused_post_conv_prep_kernel(
    # ---- inputs ----
    mixed_qkv_ptr,  # [L, qkv_dim] conv'd output (contiguous)
    a_ptr,  # [L, HV]
    b_ptr,  # [L, HV]
    # ---- params ----
    A_log_ptr,  # [HV]
    dt_bias_ptr,  # [HV]
    # ---- outputs ----
    q_ptr,  # [L, H, K] contiguous
    k_ptr,  # [L, H, K] contiguous
    v_ptr,  # [L, HV, V] contiguous
    g_ptr,  # [L, HV] float32
    beta_ptr,  # [L, HV] float32
    # ---- strides ----
    stride_x_tok,  # qkv_dim
    stride_a_tok,  # HV
    stride_b_tok,  # HV
    stride_q_tok,  # H * K
    stride_k_tok,  # H * K
    stride_v_tok,  # HV * V
    # ---- dims ----
    L,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    APPLY_L2NORM,
    L2NORM_EPS: tl.constexpr,
    # OUTPUT_G_EXP: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """Single fused kernel for post-conv1d preparation.

    Grid: (ceil(L, BLOCK_T) * (H + HV),)
      - decoded head block in [0, H):    Q/K head processing + l2norm
      - decoded head block in [H, H+HV): V head processing + gating
    """
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_t = (L + BLOCK_T - 1) // BLOCK_T
    total_blocks = ((L + BLOCK_T - 1) // BLOCK_T) * (H + HV)

    HK: tl.constexpr = H * K
    ones_col = tl.full((1, BK), 1, dtype=tl.float32)
    ones_raw = tl.full((BK, 1), 1, dtype=tl.float32)
    offs_k = tl.arange(0, BK)  # [BK]
    offs_v = tl.arange(0, BV)  # [BV]
    offs_A_dt = tl.arange(0, HV) # [HV]
    for i_tb in range(pid, num_t, num_jobs):
        # ============ gating processing ============
        offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = offs_t < L

        A_log_val = tl.load(A_log_ptr + offs_A_dt).to(tl.float32)
        dt_bias_val = tl.load(dt_bias_ptr + offs_A_dt).to(tl.float32)
        a_offsets = offs_t[:, None] * stride_a_tok + offs_A_dt[None, :]
        b_offsets = offs_t[:, None] * stride_b_tok + offs_A_dt[None, :]
        a_vals = tl.load(a_ptr + a_offsets, mask=mask_t[:,None], other=0).to(tl.float32)
        b_vals = tl.load(b_ptr + b_offsets, mask=mask_t[:,None], other=0).to(tl.float32)

        # g = -exp(A_log) * softplus(a + dt_bias)
        x = a_vals + dt_bias_val[None, :]
        sp = tl.where(
            x > 0,
            x + tl.extra.mlu.libdevice.fast_log(
                1.0 + tl.extra.mlu.libdevice.fast_expf(-x)
            ),
            tl.extra.mlu.libdevice.fast_log(
                1.0 + tl.extra.mlu.libdevice.fast_expf(x)
            ),
        )
        sp = tl.where(x <= SOFTPLUS_THRESHOLD, sp, x)
        g_vals = -tl.exp(A_log_val) * sp

        beta_vals = tl.extra.mlu.libdevice.fast_sigmoid(b_vals)
        gb_offsets = offs_t[:, None] * HV + offs_A_dt[None, :]
        tl.store(g_ptr + gb_offsets, g_vals, mask=mask_t[:,None])
        tl.store(beta_ptr + gb_offsets, beta_vals, mask=mask_t[:,None])

    for flat_pid in range(pid, total_blocks, num_jobs):
        i_tb = flat_pid % num_t
        i_head = flat_pid // num_t

        offs_t = i_tb * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
        mask_t = offs_t < L

        if i_head < H:
            # ============ Q/K head processing ============
            i_h = i_head
            mask_k = offs_k < K
            mask_2d = mask_t[:, None] & mask_k[None, :]  # [BLOCK_T, BK]

            # Load Q features: mixed_qkv[t, i_h*K + k]
            q_offsets = offs_t[:, None] * stride_x_tok + i_h * K + offs_k[None, :]
            q_f32 = tl.load(mixed_qkv_ptr + q_offsets, mask=mask_2d, other=0).to(tl.float32)

            # Load K features: mixed_qkv[t, HK + i_h*K + k]
            k_offsets = offs_t[:, None] * stride_x_tok + HK + i_h * K + offs_k[None, :]
            k_f32 = tl.load(mixed_qkv_ptr + k_offsets, mask=mask_2d, other=0).to(tl.float32)
            if APPLY_L2NORM:

                q_sq_sum = tl.dot(ones_col, (q_f32 * q_f32).trans(), allow_tf32=False)  # [BLOCK_T]
                k_sq_sum = tl.dot(ones_col, (k_f32 * k_f32).trans(), allow_tf32=False)
                q_inv = tl.extra.mlu.libdevice.fast_rcp(
                    tl.extra.mlu.libdevice.fast_sqrt(q_sq_sum + L2NORM_EPS)
                )

                q_f32 = q_f32 * tl.dot(q_inv.trans(), ones_col, allow_tf32=False)

                k_inv = tl.extra.mlu.libdevice.fast_rcp(
                    tl.extra.mlu.libdevice.fast_sqrt(k_sq_sum + L2NORM_EPS)
                )

                k_f32 = k_f32 * tl.dot(k_inv.trans(), ones_col, allow_tf32=False)

            # Store Q
            q_out = offs_t[:, None] * stride_q_tok + i_h * K + offs_k[None, :]
            tl.store(
                q_ptr + q_out,
                q_f32.to(q_ptr.dtype.element_ty),
                mask=mask_2d,
            )

            # Store K
            k_out = offs_t[:, None] * stride_k_tok + i_h * K + offs_k[None, :]
            tl.store(
                k_ptr + k_out,
                k_f32.to(k_ptr.dtype.element_ty),
                mask=mask_2d,
            )
        else:
            # ============ V head processing ============
            i_hv = i_head - H
            mask_v = offs_v < V
            mask_2d = mask_t[:, None] & mask_v[None, :]  # [BLOCK_T, BV]

            V_OFFSET: tl.constexpr = 2 * H * K

            # Load V features: mixed_qkv[t, 2*H*K + i_hv*V + v]
            v_offsets = (
                offs_t[:, None] * stride_x_tok + V_OFFSET + i_hv * V + offs_v[None, :]
            )
            v_vals = tl.load(mixed_qkv_ptr + v_offsets, mask=mask_2d, other=0)

            # Store V
            v_out = offs_t[:, None] * stride_v_tok + i_hv * V + offs_v[None, :]
            tl.store(v_ptr + v_out, v_vals, mask=mask_2d)


