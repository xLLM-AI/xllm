# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free normalization contracts and startup-bound public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract, PreparedKernel

FUSED_ADD_RMS_NORM_CONTRACT = OpContract("fused_add_rms_norm", 1, (), False, "fp32_residual_norm_cast")
LAYER_NORM_CONTRACT = OpContract("layer_norm", 1, (), False, "fp32_norm_cast")
RMS_NORM_CONTRACT = OpContract("rms_norm", 1, (), False, "fp32_norm_cast")
L2_NORM_CONTRACT = OpContract("l2_norm", 1, (), False, "fp32_sum_squares_eps_rsqrt_cast")

L2_NORM_OPERATION = Operation(
    L2_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.triton",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "triton"),
            implementations=("npu.triton.l2_norm",),
        ),
    ),
)

FUSED_ADD_RMS_NORM_OPERATION = Operation(
    FUSED_ADD_RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.fused_add_rms_norm",),
        ),
    ),
)
LAYER_NORM_OPERATION = Operation(
    LAYER_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.layer_norm",),
        ),
    ),
)
RMS_NORM_OPERATION = Operation(
    RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.rms_norm",),
        ),
    ),
)

OPERATIONS = (FUSED_ADD_RMS_NORM_OPERATION, LAYER_NORM_OPERATION, RMS_NORM_OPERATION, L2_NORM_OPERATION)


def l2_norm(input: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize BF16 NPU input over the last dimension without mutation or alias.

    Accepts nonempty tensors of rank >= 1 and arbitrary strides. Returns fresh
    contiguous storage with the input shape/dtype. Reduction and epsilon are
    FP32: x * rsqrt(sum(x*x) + eps), followed by a BF16 cast. The startup-bound
    Triton implementation supports eager execution and ACLGraph capture/replay.
    """
    return L2_NORM_OPERATION.get_prepared().function(input, eps)


def prepare_rms_norm() -> PreparedKernel:
    """Return the RMSNorm binding retained for the worker and graph lifetime."""
    return RMS_NORM_OPERATION.get_prepared()


def fused_add_rms_norm(
    value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the startup binding without selection or retry."""
    return FUSED_ADD_RMS_NORM_OPERATION.get_prepared().function(value, residual, weight, eps)


def layer_norm(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Execute the startup binding without selection or retry."""
    return LAYER_NORM_OPERATION.get_prepared().function(value, weight, bias, eps)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize without mutation or aliasing using the startup binding."""
    return RMS_NORM_OPERATION.get_prepared().function(input, weight, eps)
