# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent grouped_matmul contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("grouped_matmul", 1, (), False, "int32_accumulate_fp32_dequant_bf16")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.grouped_matmul.aclnn:load_torch_npu",
            "INT8 grouped W8A8 down projection; private NZ weights[E,K,N], cumulative INT64 group offsets; BF16 output.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
