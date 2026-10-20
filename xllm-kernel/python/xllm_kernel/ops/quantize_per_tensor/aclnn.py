# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quantize_per_tensor implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    import torch

    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("quantize_per_tensor")

    def run(value: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
        return op(value, scale, zero_point, torch.qint8, -1)

    return run
