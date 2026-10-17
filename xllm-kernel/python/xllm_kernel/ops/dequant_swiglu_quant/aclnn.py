# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dequant_swiglu_quant implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("dequant_swiglu_quant")

    def run(
        value: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        swiglu_mode: int = 0,
        clamp_limit: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return op(
            value, weight_scale, activation_scale, None, None, None, None, True, 1, swiglu_mode, clamp_limit, 1.0, 0.0
        )

    return run


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor,
        swiglu_mode: int = 0,
        clamp_limit: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch_npu.npu_dequant_swiglu_quant(
            value,
            weight_scale=weight_scale,
            activation_scale=activation_scale,
            activate_left=True,
            quant_mode=1,
            swiglu_mode=swiglu_mode,
            clamp_limit=clamp_limit,
            glu_alpha=1.0,
            glu_bias=0.0,
        )

    return run
