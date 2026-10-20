# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent sfa_dcp_remap contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("sfa_dcp_remap", 1, (), False, "exact_stable_compaction")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "tilelang",
            "xllm_kernel.ops.sfa_dcp_remap.tilelang:load",
            "DCP owner filtering, stable compaction and local token remap; INT32[T,1,2048], T<=256; TileLang AOT only, no JIT.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.sfa_dcp_remap.torch:load",
            "DCP owner filtering, stable compaction and local token remap; INT32[T,1,2048], T<=256; TileLang AOT only, no JIT.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
    ),
)
prepare = OPERATOR.prepare
