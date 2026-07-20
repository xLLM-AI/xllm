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
        # [cos(freqs) | sin(freqs)] -> [max_position, head_dim].
        cos_sin_cache = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        if dtype is not None:
            cos_sin_cache = cos_sin_cache.to(dtype)
        self.register_buffer(
            "cos_sin_cache", cos_sin_cache.contiguous(), persistent=False
        )
