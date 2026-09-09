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
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license.
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl

from .kda_op import exp


@triton.jit(do_not_specialize=["T"])
def fused_recurrent_gated_delta_rule_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    cu_seqlens,
    scale,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,  # whether to use initial state
    STORE_FINAL_STATE: tl.constexpr,  # whether to store final state
    IS_BETA_HEADWISE: tl.constexpr,  # whether beta is headwise vector or scalar,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        all = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all = B * T
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if not IS_KDA:
        p_g = g + bos * HV + i_hv
    else:
        p_gk = g + (bos * H + i_h) * K + o_k

    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0 + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    for _ in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        # [BV, BK]
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= exp(b_g)
        else:
            b_gk = tl.load(p_gk, mask=mask_k, other=0).to(tl.float32)
            b_h *= exp(b_gk[None, :])
        # [BV]
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        # [BV, BK]
        b_h += b_v[:, None] * b_k[None, :]
        # [BV]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += H * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)

    if STORE_FINAL_STATE:
        p_ht = ht + i_nh * V * K + o_v[:, None] * K + o_k[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit
def fused_recurrent_kda_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    A_log,
    dt_bias,
    lower_bound,
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
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    USE_LOWER_BOUND: tl.constexpr,
):
    """KDA packed decode: same shape as the GDN packed decode kernel, but
    with a per-K gate (``a`` is ``[B, HV*K]`` and ``dt_bias`` is ``[HV*K]``),
    so the state decay is a per-K vector ``exp(g)`` rather than a scalar."""
    i_v, i_n, i_hv = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]

    state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq).to(tl.int64)
    p_o = o + (i_n * HV + i_hv) * V + o_v

    if state_idx < 0:
        zero = tl.zeros([BV], dtype=tl.float32).to(p_o.dtype.element_ty)
        tl.store(p_o, zero, mask=mask_v)
        return

    p_h0 = h0 + state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)

    p_mixed = mixed_qkv + i_n * stride_mixed_qkv_tok
    q_off = i_h * K + o_k
    k_off = (H * K) + i_h * K + o_k
    v_off = (2 * H * K) + i_hv * V + o_v
    b_q = tl.load(p_mixed + q_off, mask=mask_k, other=0).to(tl.float32)
    b_k = tl.load(p_mixed + k_off, mask=mask_k, other=0).to(tl.float32)
    b_v = tl.load(p_mixed + v_off, mask=mask_v, other=0).to(tl.float32)

    if USE_QK_L2NORM_IN_KERNEL:
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # KDA per-K gate: load BK values of ``a`` and ``dt_bias`` for this head.
    p_a = a + i_n * stride_a_tok + i_hv * K + o_k
    p_dt = dt_bias + i_hv * K + o_k
    b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
    b_dt = tl.load(p_dt, mask=mask_k, other=0).to(tl.float32)
    A_log_val = tl.load(A_log + i_hv).to(tl.float32)

    x = b_a + b_dt
    if USE_LOWER_BOUND:
        # KDA safe gate: lower_bound * sigmoid(exp(A_log) * (g + bias)),
        # matching the chunked prefill kernel (kda.py) and FLA reference.
        b_g = lower_bound * tl.sigmoid(tl.exp(A_log_val) * x)  # [BK]
    else:
        # Standard gate: -exp(A_log) * softplus(g + bias)
        softplus_x = tl.where(x <= SOFTPLUS_THRESHOLD, tl.log(1.0 + tl.exp(x)), x)
        b_g = -tl.exp(A_log_val) * softplus_x  # [BK]

    b_val = tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32)
    # Keep beta in fp32 (no bf16 round-trip) to match the generic decode
    # kernel `fused_sigmoid_gating_delta_rule_update`, which is the reference
    # validated against torch.
    beta_val = tl.sigmoid(b_val).to(tl.float32)

    # Per-K decay: each K-row of the [V, K] state decays by its own exp(g_k).
    b_h *= exp(b_g)[None, :]
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta_val
    b_h += b_v[:, None] * b_k[None, :]
    b_o = tl.sum(b_h * b_q[None, :], 1)
    tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)

    p_ht = ht + state_idx * stride_final_state_token
    p_ht = p_ht + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


def fused_recurrent_kda_packed_decode(
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
    lower_bound: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """KDA T=1 decode fast path. Mirrors ``fused_recurrent_gated_delta_rule_packed_decode``
    but the gate ``g`` is a per-K vector instead of a scalar.

    Args:
        mixed_qkv: ``[B, 2*H*K + HV*V]`` packed projection output after conv1d.
            Requires ``num_q_heads == num_k_heads == H`` and ``head_q_dim == head_k_dim == K``.
        a: ``[B, HV*K]`` per-K gate input (typically reshaped from ``[B, HV, K]``).
        b: ``[B, HV]`` beta input (post-sigmoid scalar per head).
        A_log: ``[HV]`` log-space decay parameter.
        dt_bias: ``[HV*K]`` per-K time-step bias.
        scale: attention scale factor (typically ``head_k_dim ** -0.5``).
        initial_state: ``[num_slots, HV, V, K]`` full state pool, updated in place.
        out: ``[B, 1, HV, V]`` contiguous output buffer.
        ssm_state_indices: ``[B]`` per-request state slot indices (-1 = skip).
        use_qk_l2norm_in_kernel: apply per-head L2 norm to Q/K inside the kernel.
        lower_bound: enable KDA safe gate when set, matching
            ``fused_sigmoid_gating_delta_rule_update``.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be a 2D tensor (got ndim={mixed_qkv.ndim}).")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError(f"`a` and `b` must be 2D tensors (got a.ndim={a.ndim}, b.ndim={b.ndim}).")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a`/`b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log`/`dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log`/`dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError(f"`ssm_state_indices` must be 1D for packed decode (got ndim={ssm_state_indices.ndim}).")
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    dev = mixed_qkv.device
    if any(t.device != dev for t in (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)):
        raise ValueError("All inputs must be on the same device.")

    B = mixed_qkv.shape[0]
    if a.shape[0] != B or b.shape[0] != B:
        raise ValueError(
            f"Mismatched batch sizes: mixed_qkv.shape[0]={B}, a.shape[0]={a.shape[0]}, b.shape[0]={b.shape[0]}."
        )
    if ssm_state_indices.shape[0] != B:
        raise ValueError(
            f"`ssm_state_indices` must have shape [B] (got {tuple(ssm_state_indices.shape)}; expected ({B},))."
        )

    if initial_state.ndim != 4:
        raise ValueError(f"`initial_state` must be a 4D tensor (got ndim={initial_state.ndim}).")
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")
    HV, V, K = initial_state.shape[-3:]
    if a.shape[1] != HV * K:
        raise ValueError(f"`a` must have shape [B, HV*K] with HV={HV}, K={K} (got a.shape={tuple(a.shape)}).")
    if b.shape[1] != HV:
        raise ValueError(f"`b` must have shape [B, HV] with HV={HV} (got b.shape={tuple(b.shape)}).")
    if A_log.numel() != HV:
        raise ValueError(f"`A_log` must have {HV} elements (got {A_log.numel()}).")
    if dt_bias.numel() != HV * K:
        raise ValueError(f"`dt_bias` must have {HV * K} elements (got {dt_bias.numel()}).")
    if out.shape != (B, 1, HV, V):
        raise ValueError(f"`out` must have shape {(B, 1, HV, V)} (got out.shape={tuple(out.shape)}).")

    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - HV * V
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid packed `mixed_qkv` last dim={qkv_dim} for HV={HV}, V={V}.")
    q_dim = qk_dim // 2
    if q_dim % K != 0:
        raise ValueError(
            f"Invalid packed Q size {q_dim}: must be divisible by K={K}. "
            "KDA packed decode requires num_q_heads == num_k_heads and "
            "head_q_dim == head_k_dim."
        )
    H = q_dim // K
    if H <= 0 or HV % H != 0:
        raise ValueError(f"Invalid head config inferred from mixed_qkv: H={H}, HV={HV}.")

    BK = triton.next_power_of_2(K)
    if triton.cdiv(K, BK) != 1:
        raise ValueError(f"Packed decode kernel only supports NK=1 (got K={K}, BK={BK}).")
    BV = min(triton.next_power_of_2(V), 32)
    num_stages = 3
    num_warps = 1

    stride_mixed_qkv_tok = mixed_qkv.stride(0)
    stride_a_tok = a.stride(0)
    stride_b_tok = b.stride(0)
    stride_init_state_token = initial_state.stride(0)
    stride_final_state_token = initial_state.stride(0)
    stride_indices_seq = ssm_state_indices.stride(0)

    NV = triton.cdiv(V, BV)
    grid = (NV, B, HV)
    fused_recurrent_kda_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
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
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out, initial_state


__all__ = [
    "fused_recurrent_gated_delta_rule_fwd_kernel",
    "fused_recurrent_kda_packed_decode",
]
