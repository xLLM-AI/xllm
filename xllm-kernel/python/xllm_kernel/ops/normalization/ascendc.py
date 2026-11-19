# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the package-owned AscendC normalization implementations."""

import torch

from xllm_kernel import _ascendc_opp  # noqa: F401
from xllm_kernel.ops.normalization import (
    FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
    RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    RMS_NORM_GATED_CONTRACT,
)
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

torch.ops.xllm_kernel.prepare_gamma_add_rms_norm()
torch.ops.xllm_kernel.prepare_ascendc_normalization()
register_kernel(
    name="npu.ascendc.fused_add_gemma_rms_norm",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Nonempty NPU ND input and residual with matching shape and FP16, BF16 or FP32 dtype.",
    solution="ascendc",
    binding="aclnn",
    contract=FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
)(torch.ops.xllm_kernel.fused_add_gemma_rms_norm.default)
register_kernel(
    name="npu.ascendc.rms_norm_gated",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Nonempty NPU FP16, BF16 or FP32 input with a matching gate and 1D weight.",
    solution="ascendc",
    binding="aclnn",
    contract=RMS_NORM_GATED_CONTRACT,
)(torch.ops.xllm_kernel.rms_norm_gated.default)
register_kernel(
    name="npu.ascendc.rms_norm_dynamic_quant",
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Nonempty NPU FP16 or BF16 input with a matching 1D weight.",
    solution="ascendc",
    binding="aclnn",
    contract=RMS_NORM_DYNAMIC_QUANT_CONTRACT,
)(torch.ops.xllm_kernel.rms_norm_dynamic_quant.default)
