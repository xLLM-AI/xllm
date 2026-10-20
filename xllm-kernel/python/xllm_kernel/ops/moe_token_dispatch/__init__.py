# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent moe_token_dispatch contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_token_dispatch", 1, (), False, "symmetric_int8_gather_cumulative_groups")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_token_dispatch.aclnn:load_torch_npu",
            "Full-load W8A8 dispatch; cumulative group offsets and gather row indices; native V3 metadata conversion is included.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_token_dispatch.aclnn:load_xllm",
            "Full-load W8A8 dispatch; cumulative group offsets and gather row indices; native V3 metadata conversion is included.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
