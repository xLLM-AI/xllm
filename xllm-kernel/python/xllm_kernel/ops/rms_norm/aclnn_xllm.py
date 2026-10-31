# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""aclnn rms_norm (xllm) registered at module import."""

from __future__ import annotations

import torch
import torch_npu

from xllm_kernel.backends.xllm import load_operator
from xllm_kernel.ops.rms_norm import CONTRACT
from xllm_kernel.registry import register_kernel

_function = load_operator("rms_norm")
if not torch._C._dispatch_has_kernel_for_dispatch_key("xllm_ops::rms_norm", "PrivateUse1"):
    raise RuntimeError("xllm_ops::rms_norm has no NPU (PrivateUse1) implementation in this host")

register_kernel(
    name="npu.aclnn.rms_norm.xllm",
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="ACLNN RMSNorm through the host xllm_ops schema; no shape specialization",
    solution="aclnn",
    variant="xllm",
    binding="torch.ops.xllm_ops",
    priority=10,
    contract=CONTRACT,
)(_function)
