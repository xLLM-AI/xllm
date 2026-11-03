# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch normalization implementations registered at module import."""

from __future__ import annotations

import torch

from xllm_kernel.ops.normalization import FUSED_ADD_RMS_NORM_CONTRACT, LAYER_NORM_CONTRACT, RMS_NORM_CONTRACT
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import register_kernel

if Platform.is_npu():
    import torch_npu

    _rms_norm = torch_npu.npu_rms_norm
    _npu_add_rms_norm = torch_npu.npu_add_rms_norm

    @register_kernel(
        name="npu.torch.fused_add_rms_norm",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Functional add + RMSNorm returning (normalized,sum); neither input is mutated.",
        solution="torch",
        binding="torch_npu",
        priority=20,
        contract=FUSED_ADD_RMS_NORM_CONTRACT,
    )
    def fused_add_rms_norm(
        value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, _, summed = _npu_add_rms_norm(value, residual, weight, eps)
        return output, summed

    @register_kernel(
        name="npu.torch.layer_norm",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Contiguous NPU tensors; see operation module for the exact signature.",
        solution="torch",
        binding="torch_npu",
        priority=10,
        contract=LAYER_NORM_CONTRACT,
    )
    def layer_norm(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.nn.functional.layer_norm(value, (value.shape[-1],), weight, bias, eps)

    @register_kernel(
        name="npu.torch.rms_norm",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="torch_npu RMSNorm tensor domain",
        solution="torch",
        binding="torch_npu",
        priority=20,
        contract=RMS_NORM_CONTRACT,
    )
    def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return _rms_norm(value, weight, epsilon=eps)[0]
