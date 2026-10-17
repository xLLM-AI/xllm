# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""inplace_partial_rotary_mul implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("inplace_partial_rotary_mul")

    def run(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int) -> None:
        op(value.unsqueeze(2), cosine, sine, "interleave", [start, end])

    return run


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int) -> None:
        value[..., start:end].copy_(
            torch_npu.npu_interleave_rope(value[..., start:end].unsqueeze(2).contiguous(), cosine, sine).squeeze(2)
        )

    return run
