# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free rms_norm contract and startup-bound public API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.registry import OpContract, PreparedKernel

CONTRACT = OpContract(
    operation="rms_norm", version=1, mutates=(), aliases_inputs=False, numerical_mode="fp32_norm_cast"
)
OPERATION = Operation(
    CONTRACT,
    (
        BackendModule(
            "xllm_kernel.ops.rms_norm.aclnn_xllm",
            "npu",
            ("torch_npu", "xllm_export"),
            ("npu.aclnn.rms_norm.xllm",),
        ),
        BackendModule(
            "xllm_kernel.ops.rms_norm.aclnn_torch_npu", "npu", ("torch_npu",), ("npu.aclnn.rms_norm.torch_npu",)
        ),
        BackendModule(
            "xllm_kernel.ops.rms_norm.xlite",
            "npu",
            ("torch_npu", "xlite", "xllm_kernel._xlite"),
            ("npu.xlite.rms_norm",),
        ),
    ),
)

RMS_NORM_CONTRACT = CONTRACT


def prepare_rms_norm() -> PreparedKernel:
    """Return the binding retained for the worker and graph lifetime."""
    return OPERATION.get_prepared()


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize without mutation or aliasing using the startup binding."""
    return OPERATION.get_prepared().function(input, weight, eps)
