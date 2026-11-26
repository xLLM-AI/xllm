# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Native sampling bindings independent of the engine library."""

import torch

from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel

if current_platform.is_npu():
    from xllm_kernel import _npu_binding  # noqa: F401
    from xllm_kernel.ops.sampling import TOP_K_TOP_P

    register_kernel(
        name="npu.aclnn.top_k_top_p",
        platform=PlatformEnum.NPU,
        execution_modes=("eager", "aclgraph"),
        input_domain="NPU logits with normalized per-row INT32 top-k and floating-point top-p thresholds.",
        solution="aclnn",
        binding="aclnn",
        contract=TOP_K_TOP_P.contract,
    )(torch.ops.xllm_kernel.top_k_top_p.default)
