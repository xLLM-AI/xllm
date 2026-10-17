# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""silu_and_mul inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.silu_and_mul import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch.nn.functional as functional

    inputs = Inputs(dtype, seed)
    value = inputs.normal(*dimensions)
    gate, up = value.float().chunk(2, dim=-1)
    expected = (functional.silu(gate) * up).to(value.dtype)
    x = value.npu()
    return Example({"value": x}, lambda function: function(x), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] % 128 or dimensions[-1] > 16384):
        return "xlite requires a input width divisible by 128, at most 16384"
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (4096,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
