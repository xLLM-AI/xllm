# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent quant_matmul_out contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("quant_matmul_out", 1, ("output",), True, "int32_accumulate_or_fp32_dequant_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.quant_matmul_out.aclnn:load_xllm",
            "Same quantized matmul, writing caller-owned BF16/FP16 output; output aliases supplied tensor.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
