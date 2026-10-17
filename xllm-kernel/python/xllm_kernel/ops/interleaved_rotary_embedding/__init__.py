# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent interleaved_rotary_embedding contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("interleaved_rotary_embedding", 1, (), False, "fp32_interleaved_rope_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.interleaved_rotary_embedding.aclnn:load_torch_npu",
            "value[T,H,D], cosine/sine[T,1,1,D] with duplicated even/odd frequencies; functional interleaved RoPE; xlite table packing timed.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.interleaved_rotary_embedding.xlite:load",
            "value[T,H,D], cosine/sine[T,1,1,D] with duplicated even/odd frequencies; functional interleaved RoPE; xlite table packing timed.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
