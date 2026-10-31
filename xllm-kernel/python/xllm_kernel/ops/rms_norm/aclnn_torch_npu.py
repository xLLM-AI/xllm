# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""aclnn rms_norm (torch_npu) registered at module import."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.ops.rms_norm import CONTRACT
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    import torch

import torch_npu

_rms_norm = torch_npu.npu_rms_norm


@register_kernel(
    name="npu.aclnn.rms_norm.torch_npu",
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="torch_npu RMSNorm tensor domain",
    solution="aclnn",
    variant="torch_npu",
    binding="torch_npu",
    priority=20,
    contract=CONTRACT,
)
def rms_norm(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return _rms_norm(value, weight, epsilon=eps)[0]
