# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent sparse_flash_attention_out contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("sparse_flash_attention_out", 1, ("output",), True, "fp32_sparse_softmax_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xlite",
            "xllm_kernel.ops.sparse_flash_attention_out.xlite:load",
            "BF16 single-query decode, H<=32,D=512,R=64,K=2048,block=128; compaction, gather and workspace timed",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.sparse_flash_attention_out.aclnn:load_xllm",
            "Paged causal MLA into caller-owned output[T,H,512]; no LSE output; separate mutation contract.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
