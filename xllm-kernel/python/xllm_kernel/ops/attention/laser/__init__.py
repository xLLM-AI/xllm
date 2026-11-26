# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Dense non-causal attention for BNSD inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

OPERATION = Operation(
    OpContract("laser_attention", 1, (), False, "dense_attention"),
    (
        BackendModule(
            module="xllm_kernel.ops.attention.laser.ascendc",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.ascendc.laser_attention",),
        ),
    ),
)
OPERATIONS = (OPERATION,)


def laser_attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply scaled attention in the input dtype, using Laser for long self-attention."""
    return OPERATION.get_prepared().function(query, key, value, scale)
