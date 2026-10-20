# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent scatter_nd_update contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("scatter_nd_update", 1, ("value",), True, "exact_cache_write")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.scatter_nd_update.aclnn:load_xllm",
            "Unique valid block/offset indices[T,2] write cache[B,S,1,D]; in-place with all untouched slots preserved.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.scatter_nd_update.torch:load",
            "Unique valid block/offset indices[T,2] write cache[B,S,1,D]; in-place with all untouched slots preserved.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
    ),
)
prepare = OPERATOR.prepare
