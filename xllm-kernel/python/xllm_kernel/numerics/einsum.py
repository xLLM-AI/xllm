# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""einsum inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.einsum import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    rows, heads, width, channels = dimensions
    inputs = Inputs(dtype, seed)
    value, weight = inputs.normal(rows, heads, width, scale=0.1), inputs.normal(heads, width, channels, scale=0.1)
    expected = torch.stack([value[:, head].float() @ weight[head].float() for head in range(heads)], dim=1).to(
        value.dtype
    )
    x, w = value.npu(), weight.npu()
    return Example({"value": x, "weight": w}, lambda function: function(x, w), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and any(size % 16 for size in dimensions[-2:]):
        return "xlite einsum requires aligned reduction and output dimensions"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "heads": (4,), "inputs": (192, 512), "outputs": (256, 512)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
