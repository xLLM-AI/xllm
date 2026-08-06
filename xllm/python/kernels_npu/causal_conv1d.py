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

"""NPU causal-convolution kernels.

Neither has an NPU kernel yet. The signatures are the contract an NPU
implementation has to meet; see ``kernels_cuda/causal_conv1d.py`` and the
Triton launcher it calls for the reference behaviour.
"""

from __future__ import annotations

import torch


def causal_conv1d_prefill(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Convolve a variable-length batch and update the convolution states.

    Args:
        value: Packed activations of shape ``[num_tokens, channels]``.
        weight: Depthwise kernel of shape ``[channels, kernel_size]``.
        conv_state: Per-sequence convolution state, updated in place.
        state_indices: State slot of every sequence.
        has_initial_state: Whether a sequence continues an earlier state.
        query_start_loc: Start offset of every sequence in ``value``.

    Returns:
        Convolved activations with the shape and dtype of ``value``.
    """
    del value, weight, conv_state, state_indices, has_initial_state
    del query_start_loc
    raise NotImplementedError(
        "causal_conv1d_prefill has no NPU kernel; see "
        "kernels_cuda/triton/causal_conv1d.py for the reference implementation"
    )


def causal_conv1d_decode(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Convolve one token per sequence and update the convolution states.

    Args:
        value: Activations of shape ``[batch_size, channels]``.
        weight: Depthwise kernel of shape ``[channels, kernel_size]``.
        conv_state: Per-sequence convolution state, updated in place.
        state_indices: State slot of every sequence.

    Returns:
        Convolved activations with the shape and dtype of ``value``.
    """
    del value, weight, conv_state, state_indices
    raise NotImplementedError(
        "causal_conv1d_decode has no NPU kernel; see "
        "kernels_cuda/triton/causal_conv1d.py for the reference implementation"
    )


__all__ = ["causal_conv1d_prefill", "causal_conv1d_decode"]
