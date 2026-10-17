# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quantize_per_tensor inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.quantize_per_tensor import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    value = inputs.normal(*dimensions, scale=8.0)
    scale = torch.tensor([0.1], dtype=torch.float32)
    zero = torch.tensor([3], dtype=torch.int32)
    expected = (value.float() / scale + zero).round().clamp(-128, 127).to(torch.int8)
    x, s, z = value.npu(), scale.npu(), zero.npu()
    return Example(
        {"value": x, "scale": s, "zero_point": z}, lambda function: function(x, s, z), expected, tolerances={"": (0, 1)}
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
