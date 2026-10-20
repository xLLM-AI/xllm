# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""ACLNN RMSNorm through the xLLM or torch_npu binding."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    import torch
    import torch_npu  # noqa: F401

    from xllm_kernel.backends.xllm import load_operator

    function = load_operator("rms_norm")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("xllm_ops::rms_norm", "PrivateUse1"):
        raise RuntimeError("xllm_ops::rms_norm has no NPU (PrivateUse1) implementation in this host")
    return function


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch_npu.npu_rms_norm(value, weight, eps)[0]

    return run
