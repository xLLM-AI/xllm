# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent moe_token_unpermute contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_token_unpermute", 1, (), False, "fp32_weighted_sum_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_token_unpermute.aclnn:load_torch_npu",
            "Full-load gather row indices[T*K], FP32 probabilities[T,K], routed tensor[T*K,D] -> weighted sum[T,D].",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
