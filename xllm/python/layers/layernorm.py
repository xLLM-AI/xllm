# Copyright 2025-2026 The xLLM Authors.
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

"""RMSNorm layer (with optional fused residual-add), matching xLLM's
``apply_norm``. Depends on :mod:`python.kernels`."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels


class RMSNorm(nn.Module):
    """RMSNorm with optional fused residual-add, matching xLLM's apply_norm.

    - ``forward(x)`` -> normed x
    - ``forward(x, residual)`` -> (normed(x + residual), x + residual)
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=dtype, device=device))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return kernels.rms_norm(x, self.weight, self.eps)
        return kernels.fused_add_rms_norm(x, residual, self.weight, self.eps)


class GemmaRMSNorm(nn.Module):
    """Gemma-style RMSNorm used by Qwen3.5.

    Checkpoints store the offset from one, so the effective scale is
    ``1 + weight`` rather than ``weight``.
    """

    def __init__(
        self,
        dim: int,
        eps: float = 1e-6,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim, dtype=torch.float32, device=device))

    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # TODO: Extend the fused RMSNorm/residual-add op with Gemma's FP32
        # (weight + 1) semantics to remove eager decode kernels in a performance PR.
        original_dtype = x.dtype
        normalized = x.float()
        if residual is not None:
            normalized = normalized + residual.float()
            residual = normalized.to(original_dtype)

        variance = normalized.pow(2).mean(dim=-1, keepdim=True)
        normalized = normalized * torch.rsqrt(variance + self.eps)
        normalized = normalized * (self.weight + 1.0)
        output = normalized.to(original_dtype)
        if residual is None:
            return output
        return output, residual
