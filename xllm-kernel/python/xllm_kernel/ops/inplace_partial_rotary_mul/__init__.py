# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent inplace_partial_rotary_mul contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("inplace_partial_rotary_mul", 1, ("value",), True, "fp32_interleaved_rope_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.inplace_partial_rotary_mul.aclnn:load_xllm",
            "In-place half-open [start,end) interleaved rotation on [T,H,D]; untouched tail must remain exact; CANN gather/rotate/copy timed.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.inplace_partial_rotary_mul.aclnn:load_torch_npu",
            "In-place half-open [start,end) interleaved rotation on [T,H,D]; untouched tail must remain exact; CANN gather/rotate/copy timed.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
