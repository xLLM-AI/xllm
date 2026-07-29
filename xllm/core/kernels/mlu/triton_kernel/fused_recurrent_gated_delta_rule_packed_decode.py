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

import torch
import triton
import triton.language as tl
from triton.backends.mlu import driver


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
    BLOCK_B: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    # num_hv = tl.cdiv(HV, BLOCK_HV)
    num_v = tl.cdiv(V, BLOCK_V)
    N = B
    TOTAL_BLOCKS = num_v * N

    rangeB = tl.arange(0, BLOCK_B)
    maskB = rangeB < B
    state_idxs = tl.load(ssm_state_indices + rangeB * stride_indices_seq, mask=maskB).to(tl.int32)
    A_log_vals = tl.load(A_log + tl.arange(0,HV)).to(tl.float32)
    dt_bias_vals = tl.load(dt_bias + tl.arange(0,HV)).to(tl.float32)
    a_vals = tl.load(a + (rangeB * stride_a_tok)[:, None] + tl.arange(0,HV)[None, :], mask=maskB[:, None])
    b_vals = tl.load(b + (rangeB * stride_b_tok)[:, None] + tl.arange(0,HV)[None, :], mask=maskB[:, None]).to(tl.float32)


    xs = a_vals + dt_bias_vals[None, :]
    softplus_xs = tl.where(xs <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(xs)), xs)
    g_vals = -tl.exp(A_log_vals)[None, :] * softplus_xs  # [MAX_N, HV]
    beta_vals = tl.sigmoid(b_vals)  # [MAX_N, HV]
    exp_g_vals = tl.exp(g_vals)

    ones = tl.full((1,BLOCK_K),1,tl.float32)
    neg_ones = tl.full((1,BLOCK_K),-1,tl.float32)


    num_percore = TOTAL_BLOCKS // num_jobs
    num_acturepercore = num_percore
    if pid == num_jobs - 1:
        num_acturepercore = TOTAL_BLOCKS - num_percore * (num_jobs - 1)

    o_k = tl.arange(0, BLOCK_K)
    mask_k = o_k < K

    o_h = tl.arange(0, H)
    o_hv = tl.arange(0, HV)
    o_N_blk = tl.arange(0, BLOCK_N)

    b_os = tl.empty((BLOCK_N,HV,BLOCK_V), dtype=tl.float32)
    zeros = tl.zeros((BLOCK_V,), dtype=tl.float32).to(o.dtype.element_ty)
    for id in range(0, num_acturepercore, BLOCK_N):
        flat_pid = id + num_percore * pid
        i_v = flat_pid % num_v
        i_n = flat_pid // num_v


        if pid == num_jobs - 1 and num_acturepercore % BLOCK_N != 0 and num_acturepercore // BLOCK_N == id:
            end_N = i_n + num_acturepercore % BLOCK_N
        else:
            end_N = i_n + BLOCK_N

        o_v = i_v * BLOCK_V + tl.arange(0, BLOCK_V)
        mask_v = o_v < V
        mask_h = mask_v[:, None] & mask_k[None, :]


        n_blk = i_n + o_N_blk
        mask_n_blk = n_blk < end_N
        p_qs = mixed_qkv + n_blk[:, None, None] * stride_mixed_qkv_tok + o_h[None, :, None] * K + o_k[None, None, :]
        p_ks = mixed_qkv + n_blk[:, None, None] * stride_mixed_qkv_tok + (H * K) + o_h[None, :, None] * K + o_k[None, None, :]
        p_vs = mixed_qkv + n_blk[:, None, None] * stride_mixed_qkv_tok + (2 * H * K) + o_hv[None, :, None] * V + o_v[None, None, :]
        mask_qs = mask_n_blk[:, None, None] & (o_k[None, None, :] < K)
        mask_vs_blk = mask_n_blk[:, None, None] & (o_v[None, None, :] < V)
        qs = tl.load(p_qs, mask=mask_qs, other=0).to(tl.float32)  # [BLOCK_N, H, BK]
        ks = tl.load(p_ks, mask=mask_qs, other=0).to(tl.float32)  # [BLOCK_N, H, BK]
        vs = tl.load(p_vs, mask=mask_vs_blk, other=0).to(tl.float32)  # [BLOCK_N, HV, BV]
        if USE_QK_L2NORM_IN_KERNEL:
            qs = qs * (tl.rsqrt(tl.sum(qs * qs, axis=-1, keep_dims=True) + 1e-6))
            ks = ks * (tl.rsqrt(tl.sum(ks * ks, axis=-1, keep_dims=True) + 1e-6))
        qs = qs * scale

        i_n_start = i_n

        for i_n in range(i_n, end_N, 1):
            state_idx = state_idxs[i_n]
            i_n_off = i_n - i_n_start

            for i_hv in range(0, HV, 1):
                i_h = i_hv // (HV // H)  # h(0-1)

                output_offsets = (i_n * HV + i_hv) * V + o_v
                state_base_offset = i_hv * V * K
                state_offsets = state_base_offset + o_v[:, None] * K + o_k[None, :]

                # Invalid state index (NULL_BLOCK_ID=0) only writes zero for this block.
                if state_idx <= 0:
                    b_os[i_n_off,i_hv,:] = zeros
                    # zero = tl.zeros((BLOCK_V,), dtype=tl.float32).to(p_o.dtype.element_ty)
                    # tl.store(p_o, zero, mask=mask_v)
                else:
                    p_h0 = h0 + state_idx * stride_init_state_token
                    p_h0 = p_h0 + state_offsets
                    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

                    b_q = qs[i_n_off, i_h, :]
                    b_k = ks[i_n_off, i_h, :]
                    b_v = vs[i_n_off, i_hv, :]

                    b_h *= exp_g_vals[i_n, i_hv]
                    b_v += tl.dot(neg_ones,(b_h * b_k[None, :]).trans(), allow_tf32=False)
                    b_v *= beta_vals[i_n, i_hv]
                    b_h += tl.dot(b_v.reshape([BLOCK_V,1]),ones,allow_tf32=False) * b_k[None, :]
                    b_os[i_n_off,i_hv,:] = (tl.dot(ones,(b_h * b_q[None, :]).trans(),allow_tf32=False)).reshape([BLOCK_V,])
                    # b_o = (tl.dot(ones,(b_h * b_q[None, :]).trans(),allow_tf32=False)).reshape([BLOCK_V,])
                    # tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

                    p_ht = ht + state_idx * stride_final_state_token
                    p_ht = p_ht + state_offsets
                    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


        p_os = o + n_blk[:,None,None]*(HV*V) + o_hv[None,:,None]*V + o_v[None,None,:]
        tl.store(p_os, b_os.to(p_os.dtype.element_ty), mask=mask_vs_blk)


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if mixed_qkv.ndim != 2:
        raise ValueError(
            f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim})."
        )
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(
            f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim})."
        )
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(
            f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim})."
        )
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if (
        a.device != dev
        or b.device != dev
        or A_log.device != dev
        or dt_bias.device != dev
        or initial_state.device != dev
        or out.device != dev
        or ssm_state_indices.device != dev
    ):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            "Mismatched batch sizes: "
            f"mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(
            f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim})."
        )
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV or b.shape[1] != HV:
        raise ValueError(
            f"`a`/`b` must have shape [B, HV] with HV={HV} (got a.shape={tuple(a.shape)}, b.shape={tuple(b.shape)})."
        )
    if A_log.numel() != HV or dt_bias.numel() != HV:
        raise ValueError(
            f"`A_log` and `dt_bias` must have {HV} elements (got A_log.numel()={A_log.numel()}, dt_bias.numel()={dt_bias.numel()})."
        )
    if out.shape != (B, 1, HV, V):
        raise ValueError(
            f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)})."
        )

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(
            f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}."
        )
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(f"Invalid packed Q size {q_dim}: must be divisible by K={K}.")
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(
            f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}."
        )


    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    _devprob = driver.BangUtils().get_device_properties(torch.mlu.current_device())
    TOTAL_CORE_NUM = _devprob.get("cluster_num") * _devprob.get("core_num_per_cluster")
    grid = lambda META: (
        min(
            triton.cdiv(V, META["BLOCK_V"]) * B,
            TOTAL_CORE_NUM // META.get("num_warps", 1),
        ),
    )
    tmo_fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        o=out,
        h0=initial_state,
        ht=initial_state,
        ssm_state_indices=ssm_state_indices,
        scale=scale,
        stride_mixed_qkv_tok=stride_mixed_qkv_tok,
        stride_a_tok=stride_a_tok,
        stride_b_tok=stride_b_tok,
        stride_init_state_token=stride_init_state_token,
        stride_final_state_token=stride_final_state_token,
        stride_indices_seq=stride_indices_seq,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        BLOCK_N= 4,
        BLOCK_HV= 1,
        BLOCK_V=128,
        BLOCK_K=K,
        num_stages=2,
        num_warps=1,
    )
    return out, initial_state
