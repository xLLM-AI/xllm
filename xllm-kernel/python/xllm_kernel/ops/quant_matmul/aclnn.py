# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quant_matmul implementations supplied by aclnn; variants name bindings."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch


def load_xllm() -> Callable[..., Any]:
    from xllm_kernel.backends.xllm import load_operator

    op = load_operator("quant_matmul")

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return op(value, weight, True, weight_scale, None, activation_scale, None, output_dtype)

    return run


def load_torch_npu() -> Callable[..., Any]:
    import torch_npu

    def run(
        value: torch.Tensor,
        weight: torch.Tensor,
        weight_scale: torch.Tensor,
        activation_scale: torch.Tensor | None,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch_npu.npu_quant_matmul(
            value, weight.T, weight_scale, pertoken_scale=activation_scale, output_dtype=output_dtype
        )

    return run
