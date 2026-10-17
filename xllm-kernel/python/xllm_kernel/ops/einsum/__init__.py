# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent einsum contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("einsum", 1, (), False, "fp32_matmul_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "atb",
            "xllm_kernel.ops.einsum.atb:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O].",
            ("eager", "aclgraph"),
            variant="",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.einsum.torch:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O].",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.einsum.xlite:load",
            "BF16 [T,H,D] x [H,D,O] -> [T,H,O]; D/O multiples of 16.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
