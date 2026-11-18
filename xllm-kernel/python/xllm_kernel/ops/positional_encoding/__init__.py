# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Positional encoding contracts and startup-bound APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

PARTIAL_ROTARY_EMBEDDING_CONTRACT = OpContract("partial_rotary_embedding", 1, (), False, "torch_half_rope")
PARTIAL_ROTARY_EMBEDDING_OPERATION = Operation(
    PARTIAL_ROTARY_EMBEDDING_CONTRACT,
    tuple(
        BackendModule(
            module="xllm_kernel.ops.positional_encoding.torch",
            platform=platform,
            requires=("torch",),
            implementations=(f"{platform.value}.torch.partial_rotary_embedding",),
        )
        for platform in (PlatformEnum.CPU, PlatformEnum.CUDA, PlatformEnum.NPU)
    ),
)
OPERATIONS = (PARTIAL_ROTARY_EMBEDDING_OPERATION,)


def partial_rotary_embedding(value: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """Rotate the leading 2*R channels of [T,H,D] with half-style [T,R] tables.

    Cosine and sine are already gathered for the tokens and share the value's
    device. The suffix passes through unchanged. The result is fresh storage
    and inputs are read only. Works in eager and Python graph execution.
    """
    return PARTIAL_ROTARY_EMBEDDING_OPERATION.get_prepared().function(value, cos_half, sin_half)
