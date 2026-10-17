# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""fused_add_rms_norm inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.fused_add_rms_norm import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    value, residual, weight = inputs.normal(*dimensions), inputs.normal(*dimensions), inputs.normal(dimensions[-1])
    summed = value.float() + residual.float()
    expected = (
        (summed * torch.rsqrt(summed.square().mean(-1, keepdim=True) + 1e-5) * weight.float()).to(value.dtype),
        summed.to(value.dtype),
    )
    x, r, w = value.npu(), residual.npu(), weight.npu()
    return Example({"value": x, "residual": r, "weight": w}, lambda function: function(x, r, w, 1e-5), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] % 64 or dimensions[-1] > 8192):
        return "xlite norm width must be divisible by 64 and <=8192"
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
