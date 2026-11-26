# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Bind the packaged Laser Attention operator."""

import torch

from xllm_kernel.ops.attention.laser import OPERATION
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import register_kernel

# The embedded worker already owns the native launcher and its OPP.
# Standalone Python loads the binding shipped with the kernel wheel.
_laser = getattr(torch.ops.xllm_ops, "laser_attention", None)
if _laser is None:
    from xllm_kernel import _ascendc_opp  # noqa: F401

    torch.ops.xllm_kernel.prepare_laser_attention()
    _laser = torch.ops.xllm_kernel.laser_attention

register_kernel(
    name=OPERATION.implementations[0],
    platform=PlatformEnum.NPU,
    execution_modes=("eager", "aclgraph"),
    input_domain="Dense FP16/BF16 BNSD attention with head dimension at most 128.",
    solution="ascendc",
    binding="aclnn",
    contract=OPERATION.contract,
)(_laser.default)
