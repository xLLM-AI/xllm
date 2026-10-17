# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent quant_matmul contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("quant_matmul", 1, (), False, "int32_accumulate_or_fp32_dequant_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.quant_matmul.aclnn:load_xllm",
            "INT8 value[T,K], weight[N,K], FP32 channel[N]/token[T] scales -> BF16/FP16[T,N]; no bias or offset.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.quant_matmul.aclnn:load_torch_npu",
            "INT8 value[T,K], weight[N,K], FP32 channel[N]/token[T] scales -> BF16/FP16[T,N]; no bias or offset.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
