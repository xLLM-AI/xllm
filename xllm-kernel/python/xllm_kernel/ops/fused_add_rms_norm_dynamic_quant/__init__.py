# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent fused_add_rms_norm_dynamic_quant contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("fused_add_rms_norm_dynamic_quant", 1, (), False, "fp32_add_norm_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.fused_add_rms_norm_dynamic_quant.aclnn:load_torch_npu",
            "Functional residual add + RMSNorm + dynamic INT8; returns codes, scale, residual sum; inputs unchanged.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
    ),
)
prepare = OPERATOR.prepare
