# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch NPU normalization adapters registered at module import."""

from __future__ import annotations

import torch

from xllm_kernel.ops.normalization import (
    FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    GEMMA_RMS_NORM_CONTRACT,
)
from xllm_kernel.platform import Platform, PlatformEnum
from xllm_kernel.registry import register_kernel

if Platform.is_npu():
    import torch_npu

    _gemma_rms_norm = torch_npu.npu_gemma_rms_norm
    _add_rms_norm_dynamic_quant = torch_npu.npu_add_rms_norm_dynamic_quant

    @register_kernel(
        name="npu.torch.gemma_rms_norm",
        contract=GEMMA_RMS_NORM_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Gemma RMSNorm with additive gamma; matching NPU input and weight dtypes.",
        solution="torch",
        binding="torch_npu",
        priority=10,
    )
    def gemma_rms_norm(
        value: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        """Apply Gemma RMSNorm with the checkpoint's additive gamma."""
        output, _ = _gemma_rms_norm(value, weight, eps)
        return output

    @register_kernel(
        name="npu.torch.fused_add_rms_norm_dynamic_quant",
        contract=FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_CONTRACT,
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="Matching NPU inputs; returns (int8 normalized, fp32 token scale, residual sum) without input mutation.",
        solution="torch",
        binding="torch_npu",
        priority=10,
    )
    def fused_add_rms_norm_dynamic_quant(
        value: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Add residual, apply RMSNorm, and dynamically quantize the result."""
        outputs = _add_rms_norm_dynamic_quant(
            value,
            residual,
            weight,
            epsilon=eps,
            output_mask=[True, False],
        )
        return outputs[0], outputs[3], outputs[2]
