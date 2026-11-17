# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
# Copyright 2025 Huawei Technologies Co., Ltd.
# Copyright contributors to the vLLM project
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
"""Private Triton Ascend DSL; imported only by the NPU solution."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "q",
        "k",
        "v",
        "g",
        "beta",
        "o",
        "h0",
        "ht",
        "cu_seqlens",
        "ssm_state_indices",
        "num_accepted_tokens",
        "scale",
        "N",
        "T",
        "H",
        "HV",
        "stride_init_state_token",
        "stride_final_state_token",
    ]
)
def _fused_recurrent_gated_delta_rule_fwd_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    g: tl.tensor,
    beta: tl.tensor,
    o: tl.tensor,
    h0: tl.tensor,
    ht: tl.tensor,
    cu_seqlens: tl.tensor,
    ssm_state_indices: tl.tensor,
    num_accepted_tokens: tl.tensor,
    scale: tl.tensor,
    N: tl.int64,
    T: tl.int64,
    B: tl.constexpr,
    H: tl.int64,
    HV: tl.int64,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.int64,
    stride_final_state_token: tl.int64,
    stride_indices_seq: tl.constexpr,
    stride_indices_tok: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    INPLACE_FINAL_STATE: tl.constexpr,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_CONTINUOUS_BATCHING: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    IS_KDA: tl.constexpr,
) -> None:
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    if IS_VARLEN:
        bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
        all = T
        T = eos - bos
    else:
        bos, eos = (i_n * T, i_n * T + T)
        all = B * T
    if T == 0:
        return
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
        p_gk = g + (bos * HV + i_hv) * K + o_k
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        if IS_CONTINUOUS_BATCHING:
            if IS_SPEC_DECODING:
                i_t = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
            else:
                i_t = 0
            p_h0 = (
                h0 + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_init_state_token
            )
        else:
            p_h0 = h0 + i_n * HV * K * V
        p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        if INPLACE_FINAL_STATE:
            p_ht = (
                ht + tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t).to(tl.int64) * stride_final_state_token
            )
            p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
            tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
    if not INPLACE_FINAL_STATE:
        p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)


@triton.jit(
    do_not_specialize=[
        "q",
        "k",
        "v",
        "g",
        "beta",
        "o",
        "h0",
        "ht",
        "cu_seqlens",
        "ssm_state_indices",
        "num_accepted_tokens",
        "scale",
        "N",
        "T",
        "stride_indices_seq",
        "stride_indices_tok",
        "H",
        "HV",
        "stride_init_state_token",
        "stride_final_state_token",
    ]
)
def _fused_recurrent_gated_delta_rule_spec_fwd_kernel(
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    g: tl.tensor,
    beta: tl.tensor,
    o: tl.tensor,
    h0: tl.tensor,
    ht: tl.tensor,
    cu_seqlens: tl.tensor,
    ssm_state_indices: tl.tensor,
    num_accepted_tokens: tl.tensor,
    scale: tl.tensor,
    N: tl.int64,
    T: tl.int64,
    stride_indices_seq: tl.int64,
    stride_indices_tok: tl.int64,
    B: tl.constexpr,
    H: tl.int64,
    HV: tl.int64,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    stride_init_state_token: tl.int64,
    stride_final_state_token: tl.int64,
    IS_BETA_HEADWISE: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    IS_KDA: tl.constexpr,
) -> None:
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    bos, eos = (tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64))
    all = T
    T = eos - bos
    if T == 0:
        return
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
        p_gk = g + (bos * HV + i_hv) * K + o_k
    p_o = o + ((i_k * all + bos) * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    accepted_token_idx = tl.load(num_accepted_tokens + i_n).to(tl.int64) - 1
    initial_state_idx = tl.load(
        ssm_state_indices + i_n * stride_indices_seq + accepted_token_idx * stride_indices_tok
    ).to(tl.int64)
    p_h0 = h0 + initial_state_idx * stride_init_state_token
    p_h0 = p_h0 + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        if not IS_KDA:
            b_g = tl.load(p_g).to(tl.float32)
            b_h *= tl.exp(b_g)
        else:
            b_gk = tl.load(p_gk).to(tl.float32)
            b_h *= tl.exp(b_gk[:, None])
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        if IS_BETA_HEADWISE:
            b_beta = tl.load(p_beta, mask=mask_v, other=0).to(tl.float32)
        else:
            b_beta = tl.load(p_beta).to(tl.float32)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        final_state_idx = tl.load(ssm_state_indices + i_n * stride_indices_seq + i_t * stride_indices_tok).to(tl.int64)
        p_ht = ht + final_state_idx * stride_final_state_token
        p_ht = p_ht + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        if not IS_KDA:
            p_g += HV
        else:
            p_gk += HV * K
        p_beta += HV * (V if IS_BETA_HEADWISE else 1)
