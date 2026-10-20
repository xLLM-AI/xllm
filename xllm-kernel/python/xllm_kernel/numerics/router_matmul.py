# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""router_matmul inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.router_matmul import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    rows, width, experts = dimensions
    inputs = Inputs(dtype, seed)
    value, weight = inputs.normal(rows, width, scale=0.1), inputs.normal(experts, width, scale=0.1)
    expected = value.float() @ weight.float().T
    x, w = value.npu(), weight.npu()
    return Example(
        {"value": x, "weight": w},
        lambda function: function(x, w),
        expected,
        metadata={"output_dtype": "float32", "xlite": "no matching BF16-weight/FP32-output kernel"},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "inputs": (6144,), "experts": (256,)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
