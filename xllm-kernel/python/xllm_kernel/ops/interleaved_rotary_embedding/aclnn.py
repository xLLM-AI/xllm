# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""interleaved_rotary_embedding implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
        return torch_npu.npu_interleave_rope(value.unsqueeze(2), cosine, sine).view_as(value)

    return run
