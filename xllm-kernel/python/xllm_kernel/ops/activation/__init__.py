# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free silu_and_mul contract and startup-bound public API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("silu_and_mul", 1, (), False, "fp32_swiglu_cast")
OPERATION = Operation(
    CONTRACT,
    (BackendModule("xllm_kernel.ops.activation.torch", "npu", ("torch_npu",), ("npu.torch.silu_and_mul.torch_npu",)),),
)

OPERATIONS = (OPERATION,)


def silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    """Execute the startup binding without selection or retry."""
    return OPERATION.get_prepared().function(value)
