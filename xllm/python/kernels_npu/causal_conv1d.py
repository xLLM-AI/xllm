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
from xllm_kernel.ops.attention.gdn import causal_conv1d_decode
from xllm_kernel.ops.attention.gdn import causal_conv1d_prefill as _packed_prefill
from xllm_kernel.ops.attention.gdn import causal_conv1d_qkv_prefill as _packed_qkv_prefill


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
    if weight.size(0) > weight.size(1):
        weight = weight.t().contiguous()
    return _packed_qkv_prefill(
        value,
        weight,
        conv_state,
        query_start_loc=query_start_loc.reshape(-1).tolist(),
        state_indices=state_indices.reshape(-1).tolist(),
        has_initial_state=has_initial_state.to(dtype=torch.int64).reshape(-1).tolist(),
        num_qk_heads=num_qk_heads,
        num_v_heads=num_v_heads,
        head_k_dim=head_k_dim,
        head_v_dim=head_v_dim,
    )


def causal_conv1d_prefill(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Adapt legacy tensor metadata during eager prefill."""
    if weight.size(0) > weight.size(1):
        weight = weight.t().contiguous()
    return _packed_prefill(
        value,
        weight,
        conv_state,
        query_start_loc=query_start_loc.reshape(-1).tolist(),
        state_indices=state_indices.reshape(-1).tolist(),
        has_initial_state=has_initial_state.to(dtype=torch.int64).reshape(-1).tolist(),
    )


__all__ = [
    "causal_conv1d_prefill",
    "causal_conv1d_qkv_prefill",
    "causal_conv1d_decode",
]
