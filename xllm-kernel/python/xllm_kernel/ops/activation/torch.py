# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch silu_and_mul (torch_npu) registered at module import."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.ops.activation import CONTRACT
from xllm_kernel.registry import register_kernel

if TYPE_CHECKING:
    import torch

import torch_npu

_npu_swiglu = torch_npu.npu_swiglu


@register_kernel(
    name="npu.torch.silu_and_mul.torch_npu",
    device="npu",
    execution_modes=("eager", "aclgraph"),
    input_domain="Contiguous NPU tensors; see operation module for the exact signature.",
    solution="torch",
    binding="torch_npu",
    priority=20,
    contract=CONTRACT,
)
def silu_and_mul(value: torch.Tensor) -> torch.Tensor:
    return _npu_swiglu(value, dim=-1)
