# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""add inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.add import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    inputs = Inputs(dtype, seed)
    value, residual = inputs.normal(*dimensions), inputs.normal(*dimensions)
    expected = (value.float() + residual.float()).to(value.dtype)
    x, r = value.npu(), residual.npu()
    return Example({"value": x, "residual": r}, lambda function: function(x, r), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] % 64 or dimensions[-1] > 8192):
        return "xlite requires a width divisible by 64, at most 8192"
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
