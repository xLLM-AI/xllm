# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind static and dynamic CANN quantization operators."""

import torch

from xllm_kernel import _npu_binding  # noqa: F401
from xllm_kernel.ops.quantization import DYNAMIC_QUANT_OPERATION, QUANTIZE_PER_TENSOR_OPERATION
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

for operation in (QUANTIZE_PER_TENSOR_OPERATION, DYNAMIC_QUANT_OPERATION):
    contract = operation.contract
    register_kernel(
        name=f"npu.aclnn.{contract.operation}",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="NPU quantization with INT8 or INT4 packing and optional scales.",
        solution="aclnn",
        binding="aclnn",
        contract=contract,
    )(getattr(torch.ops.xllm_kernel, contract.operation).default)
