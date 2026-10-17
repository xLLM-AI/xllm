# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent reshape_paged_cache contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("reshape_paged_cache", 1, ("key_cache", "value_cache"), True, "exact_cache_write")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "atb",
            "xllm_kernel.ops.reshape_paged_cache.atb:load",
            "INT32 unique nonnegative slots[T]; key/value[T,1,D] -> paged cache[B,S,1,D]; updates both caches; no padding slots. Ascend 910 series only; the host Ascend 950 ACLNN path is excluded.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.reshape_paged_cache.torch:load",
            "INT32 unique nonnegative slots[T]; key/value[T,1,D] -> paged cache[B,S,1,D]; updates both caches; no padding slots.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
    ),
)
prepare = OPERATOR.prepare
