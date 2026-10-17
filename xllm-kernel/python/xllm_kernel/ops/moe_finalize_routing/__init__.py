# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent moe_finalize_routing contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_finalize_routing", 1, (), False, "fp32_weighted_sum_add_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_finalize_routing.aclnn:load_torch_npu",
            "Full-load weighted routed combine plus shared-expert output; no expert bias, no dropped tokens.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
