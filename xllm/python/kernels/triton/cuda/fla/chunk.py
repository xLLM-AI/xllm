# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import torch

from xllm.python.kernels.triton.cuda.fla.chunk_delta_h import (
    chunk_gated_delta_rule_fwd_h,
)
from xllm.python.kernels.triton.cuda.fla.chunk_o import chunk_fwd_o
from xllm.python.kernels.triton.cuda.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)
from xllm.python.kernels.triton.cuda.fla.cumsum import chunk_local_cumsum
from xllm.python.kernels.triton.cuda.fla.solve_tril import solve_tril
from xllm.python.kernels.triton.cuda.fla.utils import FLA_CHUNK_SIZE, input_guard
from xllm.python.kernels.triton.cuda.fla.wy_fast import recompute_w_u_fwd


@input_guard
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the inference-only FLA chunked gated delta rule."""
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must use the same dtype")
    if q.dtype == torch.float32:
        raise ValueError("the Triton GDN prefill kernel requires a 16-bit dtype")
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("q, k, and v must use packed [1, T, H, D] layout")
    if initial_state.shape[0] != cu_seqlens.numel() - 1:
        raise ValueError("initial_state must contain one row per sequence")

    chunk_indices = None
    cumulative_g = chunk_local_cumsum(
        g,
        chunk_size=FLA_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    a_matrix = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=cumulative_g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
    )
    a_matrix = solve_tril(
        A=a_matrix,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=a_matrix,
        g_cumsum=cumulative_g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=cumulative_g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    output = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=cumulative_g,
        scale=k.shape[-1] ** -0.5,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return output, final_state
