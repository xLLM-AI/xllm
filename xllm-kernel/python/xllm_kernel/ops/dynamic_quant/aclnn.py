# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dynamic_quant implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("dynamic_quant")

    def run(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return op(value, None, None, None)

    return run


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_dynamic_quant(value)

    return run
