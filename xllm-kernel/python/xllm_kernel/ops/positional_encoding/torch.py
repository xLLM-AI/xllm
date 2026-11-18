# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Tensor implementation of half-style partial rotary embedding."""

from __future__ import annotations

import torch

from xllm_kernel.ops.positional_encoding import PARTIAL_ROTARY_EMBEDDING_CONTRACT
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel


def _partial_rotary_embedding(value: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    half_dim = cos_half.size(-1)
    first, second, passthrough = value.split((half_dim, half_dim, value.size(-1) - 2 * half_dim), dim=-1)
    cos = cos_half.unsqueeze(1)
    sin = sin_half.unsqueeze(1)
    return torch.cat((first * cos + (-second) * sin, second * cos + first * sin, passthrough), dim=-1)


for _platform, _modes in (
    (PlatformEnum.CPU, ("eager",)),
    (PlatformEnum.CUDA, ("eager", "cudagraph")),
    (PlatformEnum.NPU, ("eager", "aclgraph")),
):
    register_kernel(
        name=f"{_platform.value}.torch.partial_rotary_embedding",
        platform=_platform,
        execution_modes=_modes,
        input_domain="[T,H,D] with [T,R] cosine and sine tables; 0 < 2*R <= D.",
        solution="torch",
        binding="torch",
        contract=PARTIAL_ROTARY_EMBEDDING_CONTRACT,
    )(_partial_rotary_embedding)
