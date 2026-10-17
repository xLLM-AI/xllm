# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent einsum_out contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("einsum_out", 1, ("output",), True, "fp32_matmul_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "atb",
            "xllm_kernel.ops.einsum_out.atb:load",
            "BF16 [T,H,D] x [H,D,O] into caller-owned output[T,H,O]; stable output buffer for graph replay.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
