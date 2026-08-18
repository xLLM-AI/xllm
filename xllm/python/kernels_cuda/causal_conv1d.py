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

"""CUDA causal-convolution kernels."""

from __future__ import annotations

import torch


@torch.library.custom_op(
    "xllm_triton::causal_conv1d_prefill",
    mutates_args=("conv_state",),
)
def causal_conv1d_prefill(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Run Triton varlen causal convolution as one graph node."""
    from .triton.causal_conv1d import (
        causal_conv1d_prefill as triton_causal_conv1d_prefill,
    )

    return triton_causal_conv1d_prefill(
        value,
        weight,
        conv_state,
        state_indices,
        has_initial_state,
        query_start_loc,
    )


@causal_conv1d_prefill.register_fake
def _causal_conv1d_prefill_fake(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    del weight, conv_state, state_indices, has_initial_state, query_start_loc
    return torch.empty_like(value)


@torch.library.custom_op(
    "xllm_triton::causal_conv1d_decode",
    mutates_args=("conv_state",),
)
def causal_conv1d_decode(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Run Triton single-token causal convolution as one graph node."""
    from .triton.causal_conv1d import (
        causal_conv1d_decode as triton_causal_conv1d_decode,
    )

    return triton_causal_conv1d_decode(
        value,
        weight,
        conv_state,
        state_indices,
    )


@causal_conv1d_decode.register_fake
def _causal_conv1d_decode_fake(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    del weight, conv_state, state_indices
    return torch.empty_like(value)


__all__ = ["causal_conv1d_prefill", "causal_conv1d_decode"]
