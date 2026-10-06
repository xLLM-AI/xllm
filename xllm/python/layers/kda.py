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

"""KDA forget-gate projection and decay."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


class KdaForgetGate(nn.Module):
    """KDA forget gate consuming the merged projection latent."""

    def __init__(self, head_dim: int, num_heads: int, lower_bound: float | None) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.qkv_dim = self.head_dim * self.num_heads
        self.f_b_proj = nn.Linear(self.head_dim, self.qkv_dim, bias=False)
        self.dt_bias = nn.Parameter(torch.zeros(self.qkv_dim, dtype=torch.float32))
        self.A_log = nn.Parameter(torch.zeros(self.num_heads, dtype=torch.float32))
        self.safe_gate_lower_bound = lower_bound

    def _apply(self, fn: Callable[[torch.Tensor], torch.Tensor], recurse: bool = True) -> KdaForgetGate:
        a_log = self.A_log.detach()
        dt_bias = self.dt_bias.detach()
        super()._apply(fn, recurse)
        # Preserve FP32 constants across the model's global dtype conversion.
        self.A_log.data = a_log.to(device=self.A_log.device, dtype=torch.float32)
        self.dt_bias.data = dt_bias.to(device=self.dt_bias.device, dtype=torch.float32)
        return self

    def raw_projection(self, forget_latent: torch.Tensor) -> torch.Tensor:
        """Project the replicated latent into per-head forget logits."""
        hidden_shape = (*forget_latent.shape[:2], -1, self.head_dim)
        return self.f_b_proj(forget_latent).view(hidden_shape)

    def gate_from_raw(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply bounded sigmoid decay, or unbounded softplus when no bound is set."""
        g = raw.float() + self.dt_bias.view(1, 1, self.num_heads, self.head_dim)
        decay_rate = torch.exp(self.A_log.view(1, 1, self.num_heads, 1))
        if self.safe_gate_lower_bound is not None:
            return self.safe_gate_lower_bound * torch.sigmoid(decay_rate * g)
        return -decay_rate * F.softplus(g)
