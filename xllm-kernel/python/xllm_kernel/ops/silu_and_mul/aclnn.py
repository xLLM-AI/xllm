# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""silu_and_mul implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    return load_operator("silu_and_mul")


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_swiglu(value, dim=-1)

    return run
