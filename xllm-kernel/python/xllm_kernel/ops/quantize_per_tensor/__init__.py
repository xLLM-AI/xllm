# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Backend-independent quantize_per_tensor contract and explicit solution preparation."""

from xllm_kernel.operator import Backend, Operator
from xllm_kernel.registry import OpContract

CONTRACT = OpContract("quantize_per_tensor", 1, (), False, "static_saturated_int8")
OPERATOR = Operator(
    CONTRACT,
    (
        Backend(
            "aclnn",
            "xllm_kernel.ops.quantize_per_tensor.aclnn:load_xllm",
            "Static INT8 quantization with positive FP32 scalar scale and INT32 zero point; qint8 passed to native schema; saturated output.",
            ("eager", "aclgraph"),
            variant="xllm",
            binding="torch.ops.xllm_ops",
        ),
    ),
)
prepare = OPERATOR.prepare
