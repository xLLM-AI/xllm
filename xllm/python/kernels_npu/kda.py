# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Pure-torch conv1d helpers for KDA (Kimi Delta Attention) layers.

The delta-rule runs on fla_npu fused ops (``chunk_kda_fwd`` / ``recurrent_kda``);
only conv1d stays pure-torch here. No model coupling and no fla_npu runtime
dependency — import-safe everywhere.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """FLA-style l2norm: divide by sqrt(sum(x^2)+eps) (NOT F.normalize)."""
    norm = torch.sqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return x / norm


def _causal_conv1d_fn(
    mixed_qkv: torch.Tensor, weight: torch.Tensor,
    activation: str = "silu",
) -> torch.Tensor:
    """Depthwise causal conv1d (left-pad K-1) + activation, fp32 weight."""
    # mixed_qkv: [B, conv_dim, S]; weight: [conv_dim, K] (squeezed)
    padding = weight.shape[-1] - 1
    out = F.conv1d(
        mixed_qkv.to(weight.dtype),
        weight=weight.unsqueeze(1),
        bias=None,
        padding=padding,
        groups=mixed_qkv.shape[1],
    )[:, :, : mixed_qkv.shape[-1]]
    if activation == "silu":
        out = F.silu(out)
    return out.to(mixed_qkv.dtype)


def _causal_conv1d_update(
    mixed_qkv: torch.Tensor, conv_state: torch.Tensor,
    weight: torch.Tensor, activation: str = "silu",
) -> torch.Tensor:
    """Incremental depthwise causal conv1d + activation (single-token decode).

    Prepend ``conv_state`` (last K-1 conv inputs), run a width-0-pad conv, slice
    the tail, and update ``conv_state`` in place. ``mixed_qkv`` is
    ``[B, conv_dim, S]`` (S == 1 for decode); ``conv_state`` is
    ``[B, conv_dim, K-1]``; ``weight`` is ``[conv_dim, K]`` (squeezed).
    """
    _, hidden_size, seq_len = mixed_qkv.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, mixed_qkv], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new, weight=weight.unsqueeze(1), bias=None, padding=0,
        groups=hidden_size,
    )[:, :, -seq_len:]
    if activation == "silu":
        out = F.silu(out)
    return out.to(mixed_qkv.dtype)
