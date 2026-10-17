# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent layer_norm contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("layer_norm", 1, (), False, "fp32_norm_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "torch",
            "xllm_kernel.ops.layer_norm.torch:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.layer_norm.xlite:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
