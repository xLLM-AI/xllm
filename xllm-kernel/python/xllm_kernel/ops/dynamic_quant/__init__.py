# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent dynamic_quant contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("dynamic_quant", 1, (), False, "per_token_symmetric_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.dynamic_quant.aclnn:load_xllm",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
        Backend(
            "aclnn",
            "xllm_kernel.ops.dynamic_quant.aclnn:load_torch_npu",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
            ("eager", "aclgraph"),
            variant="torch_npu",
            binding="torch_npu",
        ),
        Backend(
            "xlite",
            "xllm_kernel.ops.dynamic_quant.xlite:load",
            "value[T,D] -> (int8[T,D], float32[T]); finite per-token symmetric quantization; no smoothing/grouping.",
            ("eager", "aclgraph"),
            variant="",
            binding="xllm_kernel._xlite",
        ),
    ),
)
prepare = OPERATOR.prepare
