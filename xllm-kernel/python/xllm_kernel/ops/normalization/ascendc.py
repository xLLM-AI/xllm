# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare the package-owned AscendC normalization implementations."""

import os
from pathlib import Path

import torch

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.normalization import (
    FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
    RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    RMS_NORM_GATED_CONTRACT,
)
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

_vendor = str(Path(__file__).resolve().parents[2] / "ascendc_opp" / "xllm_kernel")
_previous_vendors = os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
os.environ["ASCEND_CUSTOM_OPP_PATH"] = os.pathsep.join(dict.fromkeys([_vendor, *filter(None, _previous_vendors)]))
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
