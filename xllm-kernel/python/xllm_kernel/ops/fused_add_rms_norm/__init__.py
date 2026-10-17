# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent fused_add_rms_norm contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("fused_add_rms_norm", 1, (), False, "fp32_residual_norm_cast")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.fused_add_rms_norm.aclnn:load_xllm",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.fused_add_rms_norm.aclnn:load_torch_npu",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.fused_add_rms_norm.xlite:load",
            "Functional add + RMSNorm returning (normalized,sum); neither input is mutated. xlite copies residual into its returned sum buffer.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
