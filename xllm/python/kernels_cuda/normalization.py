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

"""CUDA normalization kernels."""

from __future__ import annotations

import torch

rms_norm = torch.ops.xllm_ops.rms_norm
fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm


def gemma_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Gemma RMSNorm with the checkpoint's additive gamma."""
    variance = value.pow(2).mean(dim=-1, keepdim=True)
    normalized = value * torch.rsqrt(variance + eps)
    return normalized * (weight + 1.0)


def fused_add_rms_norm_dynamic_quant(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Report that fused add-RMSNorm-dynamic-quant is NPU-only."""
    del value, residual, weight, eps
    raise NotImplementedError("fused_add_rms_norm_dynamic_quant is available only on NPU")


@torch.library.custom_op("xllm_triton::l2_norm", mutates_args=())
def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Run Triton L2 normalization as one graph node."""
    from .triton.l2_norm import l2_norm as triton_l2_norm

    return triton_l2_norm(value, eps)


@l2_norm.register_fake
def _l2_norm_fake(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    del eps
    return torch.empty_like(value)


@torch.library.custom_op("xllm_triton::rms_norm_gated", mutates_args=())
def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Run Triton gated RMSNorm as one graph node."""
    from .triton.rms_norm import (
        rms_norm_gated as triton_rms_norm_gated,
    )

    return triton_rms_norm_gated(value, gate, weight, eps)


@rms_norm_gated.register_fake
def _rms_norm_gated_fake(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    del gate, weight, eps
    return torch.empty_like(value)


__all__ = [
    "rms_norm",
    "gemma_rms_norm",
    "fused_add_rms_norm",
    "fused_add_rms_norm_dynamic_quant",
    "l2_norm",
    "rms_norm_gated",
]
