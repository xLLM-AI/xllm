# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dynamic_quant inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.quantization_reference import symmetric_int8
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.dynamic_quant import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    inputs = Inputs(dtype, seed)
    value = inputs.normal(*dimensions)
    # Include a zero row without turning the single-token benchmark into a no-op.
    if dimensions[0] > 1:
        value[0].zero_()
    expected = symmetric_int8(value)
    x = value.npu()
    return Example({"value": x}, lambda function: function(x), expected, tolerances={"/0": (0, 1), "/1": (1e-3, 1e-6)})


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dtype != "bfloat16" or dimensions[-1] % 64 or dimensions[-1] > 8192):
        return "xlite requires BF16 and width divisible by 64, <=8192"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "widths": (6144, 2048)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
