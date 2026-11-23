# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Prepare quantized GEMM with the package-owned small-M NZ specialization."""

import torch

from xllm_kernel.ops.gemm import QUANT_MATMUL_CONTRACT, QUANT_MATMUL_OUT_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    import torch_npu

    from xllm_kernel import _ascendc_opp  # noqa: F401

    torch.ops.xllm_kernel.prepare_quant_matmul_out()
    _quant_matmul_out = torch.ops.xllm_kernel.quant_matmul_out.default
    register_kernel(
        name="npu.ascendc.quant_matmul_out",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="CANN quantized matmul with a caller-owned contiguous output.",
        solution="ascendc",
        binding="aclnn",
        contract=QUANT_MATMUL_OUT_CONTRACT,
    )(_quant_matmul_out)

    @register_kernel(
        name="npu.aclnn.quant_matmul",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="CANN quantized GEMM with optional physically transposed weight.",
        solution="aclnn",
        binding="aclnn",
        contract=QUANT_MATMUL_CONTRACT,
    )
    def quant_matmul(
        x1: torch.Tensor,
        x2: torch.Tensor,
        transpose2: bool,
        scale: torch.Tensor,
        offset: torch.Tensor | None,
        pertoken_scale: torch.Tensor | None,
        bias: torch.Tensor | None,
        output_dtype: torch.dtype | None,
    ) -> torch.Tensor:
        if not transpose2:
            return torch_npu.npu_quant_matmul(
                x1,
                x2,
                scale,
                offset=offset,
                pertoken_scale=pertoken_scale,
                bias=bias,
                output_dtype=output_dtype,
            )
        output = x1.new_empty((*x1.shape[:-1], x2.shape[0]), dtype=output_dtype or torch.int8)
        return _quant_matmul_out(x1, x2, transpose2, scale, offset, pertoken_scale, bias, output_dtype, output)
