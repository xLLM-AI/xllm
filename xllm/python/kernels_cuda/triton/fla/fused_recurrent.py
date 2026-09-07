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
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_KDA: tl.constexpr,
):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
        all_tokens = T
        T = eos - bos
    else:
        bos, eos = i_n * T, i_n * T + T
        all_tokens = B * T

    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    if IS_BETA_HEADWISE:
        p_beta = beta + (bos * HV + i_hv) * V + o_v
    else:
        p_beta = beta + bos * HV + i_hv
    if IS_KDA:
        p_gk = g + (bos * H + i_h) * K + o_k
    else:
        p_g = g + bos * HV + i_hv
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v

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
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q *= scale
        if IS_KDA:
            b_h *= tl.exp(tl.load(p_gk, mask=mask_k, other=0).to(tl.float32))[None, :]
        else:
            b_h *= tl.exp(tl.load(p_g).to(tl.float32))
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_beta = (
            tl.load(p_beta, mask=mask_v, other=0).to(tl.float32) if IS_BETA_HEADWISE else tl.load(p_beta).to(tl.float32)
        )
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        tl.store(p_o, tl.sum(b_h * b_q[None, :], 1).to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_v += HV * V
        p_o += HV * V
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
        if IS_KDA:
            p_gk += H * K
        else:
            p_g += HV

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
    """Apply one KDA recurrence step to each packed decode row."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
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

    p_a = a + i_n * stride_a_tok + i_hv * K + o_k
    p_dt = dt_bias + i_hv * K + o_k
    b_a = tl.load(p_a, mask=mask_k, other=0).to(tl.float32)
    b_dt = tl.load(p_dt, mask=mask_k, other=0).to(tl.float32)
    a_log = tl.load(A_log + i_hv).to(tl.float32)

    gate_input = b_a + b_dt
    if USE_LOWER_BOUND:
        b_g = lower_bound * tl.sigmoid(tl.exp(a_log) * gate_input)
    else:
        softplus_input = tl.where(
            gate_input <= SOFTPLUS_THRESHOLD,
            tl.log(1.0 + tl.exp(gate_input)),
            gate_input,
        )
        b_g = -tl.exp(a_log) * softplus_input

    beta = tl.sigmoid(tl.load(b + i_n * stride_b_tok + i_hv).to(tl.float32))
    b_h *= tl.exp(b_g)[None, :]
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= beta
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
    """Run the SGLang-compatible packed KDA decode kernel.

    ``initial_state`` is a state pool and is updated in place at the slots
    selected by ``ssm_state_indices``.  Negative indices skip the recurrence
    and write zero to the corresponding row of the caller-owned ``out``.
    """
    if mixed_qkv.ndim != 2:
        raise ValueError(f"`mixed_qkv` must be 2D (got ndim={mixed_qkv.ndim}).")
    if mixed_qkv.stride(-1) != 1:
        raise ValueError("`mixed_qkv` must be contiguous in the last dim.")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("`a` and `b` must be 2D tensors.")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("`a` and `b` must be contiguous in the last dim.")
    if A_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("`A_log` and `dt_bias` must be 1D tensors.")
    if A_log.stride(0) != 1 or dt_bias.stride(0) != 1:
        raise ValueError("`A_log` and `dt_bias` must be contiguous.")
    if ssm_state_indices.ndim != 1:
        raise ValueError("`ssm_state_indices` must be 1D for packed decode.")
    if not out.is_contiguous():
        raise ValueError("`out` must be contiguous.")

    device = mixed_qkv.device
    tensors = (a, b, A_log, dt_bias, initial_state, out, ssm_state_indices)
    if any(tensor.device != device for tensor in tensors):
        raise ValueError("All inputs must be on the same device.")

    batch = mixed_qkv.shape[0]
    if a.shape[0] != batch or b.shape[0] != batch:
        raise ValueError("`mixed_qkv`, `a`, and `b` must have the same batch size.")
    if ssm_state_indices.shape[0] != batch:
        raise ValueError("`ssm_state_indices` must have shape [B].")
    if initial_state.ndim != 4:
        raise ValueError("`initial_state` must be a 4D state pool.")
    if initial_state.stride(-1) != 1:
        raise ValueError("`initial_state` must be contiguous in the last dim.")

    value_heads, value_dim, key_dim = initial_state.shape[-3:]
    if a.shape[1] != value_heads * key_dim:
        raise ValueError("`a` must have shape [B, HV*K].")
    if b.shape[1] != value_heads:
        raise ValueError("`b` must have shape [B, HV].")
    if A_log.numel() != value_heads or dt_bias.numel() != value_heads * key_dim:
        raise ValueError("`A_log` and `dt_bias` have incompatible dimensions.")
    if out.shape != (batch, 1, value_heads, value_dim):
        raise ValueError("`out` must have shape [B, 1, HV, V].")

    qk_dim = mixed_qkv.shape[1] - value_heads * value_dim
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError("`mixed_qkv` has an invalid packed dimension.")
    query_dim = qk_dim // 2
    if query_dim % key_dim != 0:
        raise ValueError("The packed query dimension must be divisible by K.")
    key_heads = query_dim // key_dim
    if key_heads <= 0 or value_heads % key_heads != 0:
        raise ValueError("KDA requires HV to be a multiple of H.")
    block_key = triton.next_power_of_2(key_dim)
    if triton.cdiv(key_dim, block_key) != 1:
        raise ValueError("Packed KDA decode requires one key tile.")
    block_value = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_value), batch * value_heads)
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
        stride_mixed_qkv_tok=mixed_qkv.stride(0),
        stride_a_tok=a.stride(0),
        stride_b_tok=b.stride(0),
        stride_init_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_seq=ssm_state_indices.stride(0),
        H=key_heads,
        HV=value_heads,
        K=key_dim,
        V=value_dim,
        BK=block_key,
        BV=block_value,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        USE_LOWER_BOUND=lower_bound is not None,
        num_warps=1,
        num_stages=3,
    )
    return out, initial_state


__all__ = [
    "fused_recurrent_gated_delta_rule_fwd_kernel",
    "fused_recurrent_kda_packed_decode",
]
