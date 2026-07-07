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

"""Op dispatch layer.

Every op calls the fused xLLM kernel exposed as ``torch.ops.xllm_ops.*``, the
*same* vendor kernel the C++ decoder path uses.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

_USE_TRITON = os.environ.get("XLLM_USE_TRITON", "0").strip().lower() not in (
    "",
    "0",
    "off",
    "false",
    "none",
)


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.ops.xllm_ops.rms_norm(x, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.xllm_ops.fused_add_rms_norm(x, residual, weight, eps)


# --------------------------------------------------------------------------
# Gated SiLU MLP activation
# --------------------------------------------------------------------------
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if _USE_TRITON:
        from ..kernels import triton_ops

        return triton_ops.silu_and_mul(x)
    return torch.ops.xllm_ops.silu_and_mul(x)


# --------------------------------------------------------------------------
# Fused per-head QK-RMSNorm + RoPE (Qwen3)
# --------------------------------------------------------------------------
def fused_qk_norm_rope(
    qkv: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    interleaved: bool = False,
) -> torch.Tensor:
    pos = position_ids.to(torch.int64).contiguous()
    return torch.ops.xllm_ops.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        interleaved,
        pos,
    )


# --------------------------------------------------------------------------
# Tensor-parallel collectives. Uses PyTorch native torch.distributed API.
# No-op when tp_size==1 (no process group initialized).
# --------------------------------------------------------------------------
_tp_group = None


def set_tp_group(group) -> None:
    """Set the TP process group (called once during model init for tp>1)."""
    global _tp_group
    _tp_group = group


def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """SUM all-reduce across the TP group (RowParallel output combine)."""
    if _tp_group is None:
        return x
    out = x.clone()
    torch.distributed.all_reduce(out, group=_tp_group)
    return out


def all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """All-gather across the TP group, concatenated along ``dim``."""
    if _tp_group is None:
        return x
    chunks = [torch.empty_like(x) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, x, group=_tp_group)
    return torch.cat(chunks, dim=dim)
