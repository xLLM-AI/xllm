# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_finalize_routing inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.moe_finalize_routing import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, width, topk = dimensions
    value = inputs.normal(rows * topk, width)
    indices = torch.randperm(rows * topk, generator=inputs.generator).to(torch.int32)
    probs = inputs.normal(rows, topk, dtype=torch.float32).softmax(-1)
    expected = (value[indices.long()].float().view(rows, topk, width) * probs[..., None]).sum(1).to(value.dtype)
    shared = inputs.normal(rows, width)
    expected = ((value[indices.long()].float().view(rows, topk, width) * probs[..., None]).sum(1) + shared.float()).to(
        value.dtype
    )
    x, i, p, s = value.npu(), indices.npu(), probs.npu(), shared.npu()
    return Example({"value": x, "indices": i, "probs": p, "shared": s}, lambda function: function(x, i, p, s), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "widths": (6144,), "topks": (8,)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
