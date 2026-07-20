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

"""NEOX-style rotary embedding cache in the layout the fused
``xllm_ops.fused_qk_norm_rope`` kernel expects. Pure table construction — no op
dispatch dependency."""

from __future__ import annotations

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """Holds the NEOX-style RoPE cos/sin cache in the exact layout the fused
    ``xllm_ops.fused_qk_norm_rope`` kernel expects.

    ``cos_sin_cache`` is a single ``[max_position, head_dim]`` tensor whose first
    ``head_dim/2`` columns are ``cos(freqs)`` and last ``head_dim/2`` are
    ``sin(freqs)`` (``freqs = outer(positions, inv_freq)``). This matches C++
    ``MRotaryEmbedding::precomputed_cos_sin_cache()`` so both paths use identical
    rotary tables. Built on ``device`` in the model dtype to match C++ exactly.
    """

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        rope_theta: float,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (
            rope_theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
                / head_dim
            )
        )
        t = torch.arange(max_position, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)  # [max_position, head_dim/2]
        cos_half = freqs.cos()  # [max_position, head_dim/2]
        sin_half = freqs.sin()
        if dtype is not None:
            cos_half = cos_half.to(dtype)
            sin_half = sin_half.to(dtype)

        device_type = torch.device(device).type if device else "cpu"
        if device_type in ("npu", "privateuseone"):
            # NPU path: pre-expand [cos_half, cos_half] so forward() is
            # index+view only. Skip cos_sin_cache (CUDA-only) to save HBM.
            self.cos_sin_cache = torch.empty(0)
            self.register_buffer(
                "_cos_expanded",
                torch.cat([cos_half, cos_half], dim=-1).contiguous(),
                persistent=False,
            )
            self.register_buffer(
                "_sin_expanded",
                torch.cat([sin_half, sin_half], dim=-1).contiguous(),
                persistent=False,
            )
        else:
            # CUDA path: [cos(freqs) | sin(freqs)] -> [max_position, head_dim].
            cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1)
            self.register_buffer(
                "cos_sin_cache", cos_sin_cache.contiguous(), persistent=False
            )

    def forward(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Index into the cache and return per-token (cos, sin).

        Returns cos/sin as ``[1, num_tokens, 1, head_dim]`` via index + view
        (zero allocation on the hot path).
        """
        num_tokens = positions.size(0)
        cos = self._cos_expanded[positions].view(1, num_tokens, 1, self.head_dim)
        sin = self._sin_expanded[positions].view(1, num_tokens, 1, self.head_dim)
        return cos, sin
