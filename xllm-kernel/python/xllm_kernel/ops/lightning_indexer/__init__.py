# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent lightning_indexer contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("lightning_indexer", 1, (), False, "relu_weighted_score_causal_topk")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xlite",
            "xllm_kernel.ops.lightning_indexer.xlite:load",
            "Single sequence, H=32,D=128,block=128,K=2048,T<=128; weight packing and workspace initialization timed",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.lightning_indexer.aclnn:load_xllm",
            "Single-sequence TND / PA_BSND, FP16/BF16 queries/cache, same-dtype head weights, cumulative INT32 lengths and paged table; causal topk.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.lightning_indexer.torch:load",
            "Single-sequence TND / PA_BSND, FP16/BF16 queries/cache, same-dtype head weights, cumulative INT32 lengths and paged table; causal topk.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
    ),
)
prepare = OPERATOR.prepare
