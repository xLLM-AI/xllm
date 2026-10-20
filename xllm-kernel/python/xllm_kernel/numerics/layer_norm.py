# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""layer_norm inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.layer_norm import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch.nn.functional as functional

    inputs = Inputs(dtype, seed)
    value, weight, bias = inputs.normal(*dimensions), inputs.normal(dimensions[-1]), inputs.normal(dimensions[-1])
    expected = functional.layer_norm(value.float(), (dimensions[-1],), weight.float(), bias.float(), 1e-6).to(
        value.dtype
    )
    x, w, b = value.npu(), weight.npu(), bias.npu()
    return Example({"value": x, "weight": w, "bias": b}, lambda function: function(x, w, b, 1e-6), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] % 64 or dimensions[-1] > 8192):
        return "xlite norm width must be divisible by 64 and <=8192"
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (128,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
