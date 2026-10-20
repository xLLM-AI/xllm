# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent moe_gmm1 contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("moe_gmm1", 1, (), False, "int32_accumulate_fp32_swiglu_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_gmm1.aclnn:load_torch_npu",
            "Grouped INT8 gate/up + standard SwiGLU + per-token INT8 quant; private NZ weights and FP32 scales; metadata conversion timed.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.moe_gmm1.aclnn:load_xllm",
            "Grouped INT8 gate/up + standard SwiGLU + per-token INT8 quant; private NZ weights and FP32 scales; metadata conversion timed.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
