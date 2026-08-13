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

"""NPU normalization kernels."""

from __future__ import annotations

import torch

rms_norm = torch.ops.xllm_ops.rms_norm
fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm


def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension of ``value`` to unit L2 norm.

    Args:
        value: Tensor whose last dimension is normalized.
        eps: Added to the squared norm before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    del value, eps
    raise NotImplementedError(
        "l2_norm has no NPU kernel; see kernels_cuda/triton/l2_norm.py for the reference implementation"
    )


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``gate``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization, same shape as ``value``.
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    del value, gate, weight, eps
    raise NotImplementedError(
        "rms_norm_gated has no NPU kernel; see kernels_cuda/triton/rms_norm.py for the reference implementation"
    )


__all__ = ["rms_norm", "fused_add_rms_norm", "l2_norm", "rms_norm_gated"]
