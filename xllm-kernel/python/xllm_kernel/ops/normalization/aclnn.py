# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""ACLNN normalization adapters using host-provided Torch bindings."""

from __future__ import annotations

import torch

from xllm_kernel.ops.normalization import (
    FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
    RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    RMS_NORM_GATED_CONTRACT,
)
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import register_kernel

if Platform.is_npu():
    _fused_add_gemma_rms_norm = torch.ops.xllm_ops.fused_add_gemma_rms_norm
    _rms_norm_dynamic_quant = torch.ops.xllm_ops.rms_norm_dynamic_quant
    _rms_norm_gated = torch.ops.xllm_ops.rms_norm_gated

    @register_kernel(
        name="npu.aclnn.fused_add_gemma_rms_norm",
        contract=FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Matching nonempty NPU inputs; additive gamma; returns (normalized, residual sum) without input mutation.",
        solution="aclnn",
        binding="xllm_ops",
        priority=10,
    )
    def fused_add_gemma_rms_norm(
        value: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Add the residual and apply Gemma RMSNorm on NPU."""
        return _fused_add_gemma_rms_norm(value, residual, weight, eps)

    @register_kernel(
        name="npu.aclnn.rms_norm_dynamic_quant",
        contract=RMS_NORM_DYNAMIC_QUANT_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Nonempty NPU input and a matching 1D weight; returns int8 values and fp32 per-token scales.",
        solution="aclnn",
        binding="xllm_ops",
        priority=10,
    )
    def rms_norm_dynamic_quant(
        value: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply RMSNorm and return int8 values with per-token scales on NPU."""
        return _rms_norm_dynamic_quant(value, weight, eps)

    @register_kernel(
        name="npu.aclnn.rms_norm_gated",
        contract=RMS_NORM_GATED_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Matching NPU value/gate tensors; last-dimension RMSNorm followed by SiLU gating.",
        solution="aclnn",
        binding="xllm_ops",
        priority=10,
    )
    def rms_norm_gated(
        value: torch.Tensor,
        gate: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """Apply RMSNorm to ``value`` and gate the result with ``silu(gate)``.

        Args:
            value: Tensor to normalize.
            gate: Gate applied after normalization (SiLU is applied internally).
            weight: RMSNorm weight over the last dimension.
            eps: RMSNorm epsilon.

        Returns:
            A tensor with the shape and dtype of ``value``.
        """
        return _rms_norm_gated(value, gate, weight, eps)
