# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent silu_and_mul contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("silu_and_mul", 1, (), False, "fp32_swiglu_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.silu_and_mul.aclnn:load_xllm",
            "Contiguous NPU tensors; see operation module for the exact signature.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.silu_and_mul.aclnn:load_torch_npu",
            "Contiguous NPU tensors; see operation module for the exact signature.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.silu_and_mul.xlite:load",
            "Contiguous NPU tensors; see operation module for the exact signature.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
