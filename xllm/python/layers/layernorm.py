# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RMSNorm layer (with optional fused residual-add), matching xLLM's
``apply_norm``. Depends on the op dispatch layer (:mod:`python.ops`)."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import ops


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
            return ops.rms_norm(x, self.weight, self.eps)
        return ops.fused_add_rms_norm(x, residual, self.weight, self.eps)
