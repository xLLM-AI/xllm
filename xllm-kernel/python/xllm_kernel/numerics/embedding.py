# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""embedding inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.embedding import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    rows, vocab, width = dimensions
    inputs = Inputs(dtype, seed)
    weight = inputs.normal(vocab, width)
    indices = inputs.integer(0, vocab, rows, dtype=torch.int32)
    expected = weight.index_select(0, indices.long())
    ids, w = indices.npu(), weight.npu()
    return Example(
        {"indices": ids, "weight": w}, lambda function: function(ids, w), expected, tolerances={"": (0.0, 0.0)}
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "xlite" and (dimensions[-1] % 64 or dimensions[-1] > 8192):
        return "xlite requires a width divisible by 64, at most 8192"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "vocab": (154880,), "widths": (384,)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
