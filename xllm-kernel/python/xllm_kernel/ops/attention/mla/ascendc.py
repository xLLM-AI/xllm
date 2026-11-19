# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare fused MLA preprocessing on devices supported by the package."""

import torch

from xllm_kernel.ops.attention.mla import MLA_PREPROCESS_V2_CONTRACT, has_mla_preprocess_v2
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu() and has_mla_preprocess_v2():
    register_kernel(
        name="npu.ascendc.mla_preprocess_v2",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="910B/910_93 quantized MLA decode with packed weights and ND or NZ caches.",
        solution="ascendc",
        binding="aclnn",
        contract=MLA_PREPROCESS_V2_CONTRACT,
    )(torch.ops.xllm_kernel.mla_preprocess_v2.default)
