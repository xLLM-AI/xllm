# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""linear inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.linear import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    rows, width, channels = dimensions
    inputs = Inputs(dtype, seed)
    value, weight = inputs.normal(rows, width, scale=0.1), inputs.normal(channels, width, scale=0.1)
    expected = (value.float() @ weight.float().T).to(value.dtype)
    x, w = value.npu(), weight.npu()
    return Example({"value": x, "weight": w}, lambda function: function(x, w), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and any(size % 16 for size in dimensions[1:]):
        return "xlite matmul requires aligned K and N"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "inputs": (6144,), "outputs": (160, 2624, 9680)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
