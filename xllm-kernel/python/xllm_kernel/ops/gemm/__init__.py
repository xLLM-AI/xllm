# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free GEMM contracts and startup-bound public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

EINSUM_CONTRACT = OpContract("einsum", 1, (), False, "fp32_matmul_cast")
EINSUM_OUT_CONTRACT = OpContract("einsum_out", 1, ("output",), True, "fp32_matmul_cast")

EINSUM_OPERATION = Operation(
    EINSUM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.gemm.atb",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "xllm_export"),
            implementations=("npu.atb.einsum",),
        ),
    ),
)
EINSUM_OUT_OPERATION = Operation(
    EINSUM_OUT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.gemm.atb",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "xllm_export"),
            implementations=("npu.atb.einsum_out",),
        ),
    ),
)
OPERATIONS = (EINSUM_OPERATION, EINSUM_OUT_OPERATION)


def einsum(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute [T,H,D] x [H,D,O] -> [T,H,O] without mutation or aliasing."""
    return EINSUM_OPERATION.get_prepared().function(value, weight)


def einsum_out(value: torch.Tensor, weight: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    """Write einsum into a contiguous BF16 output that aliases neither input."""
    return EINSUM_OUT_OPERATION.get_prepared().function(value, weight, output)
