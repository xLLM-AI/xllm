# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent moe_gate_routing contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_gate_routing", 1, (), False, "sigmoid_bias_select_normalize")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "xlite",
            "xllm_kernel.ops.moe_gate_routing.xlite:load",
            "FP32 [T,256], topk<=32, groups=1 or 8 (select 4); bitmap conversion timed",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_gate_routing.aclnn:load_torch_npu",
            "FP32 sigmoid logits[T,E] + correction[E]; group-top2 selection, renormalized top-k weights and INT32 expert IDs.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
