# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent sparse_flash_attention contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("sparse_flash_attention", 1, (), False, "fp32_sparse_softmax_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.sparse_flash_attention.aclnn:load_xllm",
            "Single-sequence causal MLA with paged latent KV and interleaved RoPE; sparse token indices (-1 padding); output plus FP32 softmax max/sum.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "torch",
            "xllm_kernel.ops.sparse_flash_attention.torch:load",
            "Single-sequence causal MLA with paged latent KV and interleaved RoPE; sparse token indices (-1 padding); output plus FP32 softmax max/sum.",
            ("eager", "aclgraph"),
            variant="",
            binding="torch",
        ),
    ),
)
prepare = OPERATOR.prepare
