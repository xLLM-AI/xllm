# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent dequant_swiglu_quant contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("dequant_swiglu_quant", 1, (), False, "fp32_swiglu_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.dequant_swiglu_quant.aclnn:load_xllm",
            "INT32[T,2H] with FP32 dequantization scales; explicit standard or interleaved SwiGLU; dynamic INT8 output.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.dequant_swiglu_quant.aclnn:load_torch_npu",
            "INT32[T,2H] with FP32 dequantization scales; explicit standard or interleaved SwiGLU; dynamic INT8 output.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
