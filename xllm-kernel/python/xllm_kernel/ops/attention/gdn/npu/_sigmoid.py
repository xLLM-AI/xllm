# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Private Triton Ascend DSL; imported only by the NPU solution."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "A_log",
        "a",
        "dt_bias",
        "softplus_beta",
        "softplus_threshold",
        "q",
        "k",
        "v",
        "b",
        "o",
        "h0_source",
        "h0_indices",
        "cu_seqlens",
        "scale",
        "B",
        "T",
        "H",
        "HV",
    ]
)
def _fused_sigmoid_gating_delta_rule_update_kernel(
    A_log: tl.tensor,
    a: tl.tensor,
    dt_bias: tl.tensor,
    softplus_beta: tl.tensor,
    softplus_threshold: tl.tensor,
    q: tl.tensor,
    k: tl.tensor,
    v: tl.tensor,
    b: tl.tensor,
    o: tl.tensor,
    h0_source: tl.tensor,
    h0_indices: tl.tensor,
    cu_seqlens: tl.tensor,
    scale: tl.tensor,
    B: tl.int64,
    T: tl.int64,
    H: tl.int64,
    HV: tl.int64,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
) -> None:
    i_k, i_v, i_nh = (tl.program_id(0), tl.program_id(1), tl.program_id(2))
    i_n, i_hv = (i_nh // HV, i_nh % HV)
    i_h = i_hv // (HV // H)
    bos = tl.load(cu_seqlens + i_n).to(tl.int64)
    eos = tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    all_tokens = B * T
    seq_len = eos - bos
    if seq_len == 0:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_b = b + bos * HV + i_hv
    p_a = a + bos * HV + i_hv
    p_o = o + ((i_k * all_tokens + bos) * HV + i_hv) * V + o_v
    p_A_log = A_log + i_hv
    p_dt_bias = dt_bias + i_hv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]
    state_idx = tl.load(h0_indices + i_n).to(tl.int64)
    safe_state_idx = tl.where(state_idx < 0, 0, state_idx)
    p_h0 = h0_source + safe_state_idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
    b_h = tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    b_h = tl.where(state_idx < 0, tl.zeros_like(b_h), b_h)
    for i_t in range(0, seq_len):
        b_q = tl.load(p_q + i_t * H * K, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k + i_t * H * K, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v + i_t * HV * V, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b + i_t * HV).to(tl.float32)
        b_a = tl.load(p_a + i_t * HV).to(tl.float32)
        x = b_a + tl.load(p_dt_bias).to(tl.float32)
        beta_x = softplus_beta * x
        softplus_x = tl.where(beta_x <= softplus_threshold, 1.0 / softplus_beta * tl.log(1.0 + tl.exp(beta_x)), x)
        b_g = -tl.exp(tl.load(p_A_log).to(tl.float32)) * softplus_x
        b_beta = 1.0 / (1.0 + tl.exp(-b_b))
        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-06)
            b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-06)
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(p_o + i_t * HV * V, b_o.to(p_o.dtype.element_ty), mask=mask_v)
    if state_idx >= 0:
        p_h0 = h0_source + state_idx * HV * K * V + i_hv * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h0, b_h.to(p_h0.dtype.element_ty), mask=mask_h)
