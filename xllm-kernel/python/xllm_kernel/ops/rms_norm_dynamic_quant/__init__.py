# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent rms_norm_dynamic_quant contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("rms_norm_dynamic_quant", 1, (), False, "fp32_norm_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.rms_norm_dynamic_quant.aclnn:load_xllm",
            "FP16/BF16 [T,D] plus gamma[D] -> (INT8[T,D], FP32[T]); fused FP32 norm before quantization.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.rms_norm_dynamic_quant.aclnn:load_torch_npu",
            "FP16/BF16 [T,D] plus gamma[D] -> (INT8[T,D], FP32[T]); fused FP32 norm before quantization.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
