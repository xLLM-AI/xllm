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

"""NPU causal-convolution kernel bindings."""

from __future__ import annotations

import torch


def causal_conv1d_qkv_prefill(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
    num_qk_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused conv + split into Q/K/V for prefill.

    Returns:
        (q, k, v) with shapes [1, T, num_qk_heads, head_k_dim],
        [1, T, num_qk_heads, head_k_dim], [1, T, num_v_heads, head_v_dim].
    """
    return torch.ops.xllm_ops.causal_conv1d_qkv_prefill(
        value,
        weight,
        conv_state,
        state_indices,
        has_initial_state.to(torch.int64),
        query_start_loc,
        num_qk_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
    )


def causal_conv1d_decode(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Convolve one token per sequence and update the convolution states.

    Args:
        value: Activations of shape ``[batch_size, channels]``.
        weight: Depthwise kernel of shape ``[kernel_size, channels]``.
        conv_state: Per-sequence convolution state, updated in place.
        state_indices: State slot of every sequence.

    Returns:
        Convolved activations with the shape and dtype of ``value``.
    """
    from .tilelang.causal_conv1d_decode import (
        DIM_PER_CORE,
        _build_decode_kernel_jit,
    )

    _, dim = value.shape
    width = weight.shape[0]
    dtype_name = str(value.dtype).removeprefix("torch.")
    dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
    kernel = _build_decode_kernel_jit(
        width,
        dim_chunks,
        DIM_PER_CORE,
        dtype_name,
        True,
    )
    indices = state_indices.to(dtype=torch.int32).contiguous()
    initial_state_mode = torch.ones_like(indices, dtype=torch.int32)
    if bias is None:
        bias = torch.zeros(dim, dtype=value.dtype, device=value.device)
    return kernel(
        value.contiguous(),
        weight,
        conv_state,
        indices,
        indices,
        initial_state_mode,
        bias,
    )


__all__ = ["causal_conv1d_qkv_prefill", "causal_conv1d_decode"]
