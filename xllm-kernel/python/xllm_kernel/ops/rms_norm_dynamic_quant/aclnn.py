# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""rms_norm_dynamic_quant implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    return load_operator("rms_norm_dynamic_quant")


def load_torch_npu() -> Callable[..., Any]:
    import torch
    import torch_npu

    def run(value: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = torch_npu.npu_add_rms_norm_dynamic_quant(
            value, torch.zeros_like(value), weight, epsilon=eps, output_mask=[True, False]
        )
        return outputs[0], outputs[3]

    return run
