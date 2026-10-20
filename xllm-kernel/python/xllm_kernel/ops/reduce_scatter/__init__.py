# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent reduce_scatter contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("reduce_scatter", 1, ("output",), True, "same_dtype_sum_scatter")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "hccl",
            "xllm_kernel.ops.reduce_scatter.hccl:load_current_stream",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
            ("eager", "aclgraph"),
            variant="current_stream",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "hccl",
            "xllm_kernel.ops.reduce_scatter.hccl:load_process_group",
            "Explicit initialized default HCCL group; contiguous ND tensors; SUM/rank-order semantics; communicator acquired before graph capture.",
            ("eager", "aclgraph"),
            variant="process_group",
            binding="torch.distributed",
        ),
    ),
)
prepare = OPERATOR.prepare
