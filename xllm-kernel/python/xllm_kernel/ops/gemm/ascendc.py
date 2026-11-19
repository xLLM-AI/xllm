# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare quantized GEMM with the package-owned small-M NZ specialization."""

import torch

from xllm_kernel.ops.gemm import QUANT_MATMUL_OUT_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel import _ascendc_opp  # noqa: F401

    torch.ops.xllm_kernel.prepare_quant_matmul_out()
    register_kernel(
        name="npu.ascendc.quant_matmul_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="CANN quantized matmul with a caller-owned contiguous output.",
        solution="ascendc",
        binding="aclnn",
        contract=QUANT_MATMUL_OUT_CONTRACT,
    )(torch.ops.xllm_kernel.quant_matmul_out.default)
