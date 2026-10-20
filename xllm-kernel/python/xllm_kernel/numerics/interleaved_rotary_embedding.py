# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""interleaved_rotary_embedding inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.interleaved_rotary_embedding import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, heads, width = dimensions
    value = inputs.normal(rows, heads, width)
    angle = inputs.normal(rows, width // 2, dtype=torch.float32)
    cosine = angle.cos().repeat_interleave(2, -1).to(inputs.dtype).view(rows, 1, 1, width)
    sine = angle.sin().repeat_interleave(2, -1).to(inputs.dtype).view(rows, 1, 1, width)
    pairs = value.float().reshape(rows, heads, -1, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), -1).flatten(-2)
    expected = (value.float() * cosine[:, 0].float() + rotated * sine[:, 0].float()).to(inputs.dtype)
    x, c, s = value.npu(), cosine.npu(), sine.npu()
    return Example({"value": x, "cosine": c, "sine": s}, lambda function: function(x, c, s), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] not in (64, 128) or dimensions[1] * dimensions[2] > 4096):
        return "xlite requires D=64/128 and H*D<=4096"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "heads": (4, 32), "widths": (64, 128)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
