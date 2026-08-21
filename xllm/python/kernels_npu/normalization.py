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
import torch_npu

rms_norm = torch.ops.xllm_ops.rms_norm
fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm
_FUSED_ADD_RMS_NORM_DYNAMIC_QUANT = torch_npu.npu_add_rms_norm_dynamic_quant


def fused_add_rms_norm_dynamic_quant(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add residual, apply RMSNorm, and dynamically quantize the result."""
    outputs = _FUSED_ADD_RMS_NORM_DYNAMIC_QUANT(
        value,
        residual,
        weight,
        epsilon=eps,
        output_mask=[True, False],
    )
    return outputs[0], outputs[3], outputs[2]
rms_norm_dynamic_quant = torch.ops.xllm_ops.rms_norm_dynamic_quant


def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension of ``value`` to unit L2 norm.

    Args:
        value: Tensor whose last dimension is normalized.
        eps: Added to the squared norm before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    return torch.ops.xllm_ops.l2_norm(value, eps)


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``silu(gate)``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization (SiLU is applied internally).
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """
    return torch.ops.xllm_ops.rms_norm_gated(value, gate, weight, eps)


__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "fused_add_rms_norm_dynamic_quant",
    "rms_norm_dynamic_quant",
    "l2_norm",
    "rms_norm_gated",
]
