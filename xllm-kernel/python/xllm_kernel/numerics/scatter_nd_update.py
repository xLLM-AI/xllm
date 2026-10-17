# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""scatter_nd_update inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.scatter_nd_update import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, width = dimensions
    blocks = (rows + 15) // 16 + 1
    cache = inputs.normal(blocks, 16, 1, width)
    slots = torch.randperm(blocks * 16, generator=inputs.generator)[:rows]
    indices = torch.stack((slots // 16, slots % 16), -1).to(torch.int64)
    updates = inputs.normal(rows, 1, width)
    expected = cache.clone()
    for i, (block, offset) in enumerate(indices.tolist()):
        expected[block, offset] = updates[i]
    x, i, u = cache.npu(), indices.npu(), updates.npu()
    return Example(
        {"value": x, "indices": i, "updates": u},
        lambda function: function(x, i, u),
        expected,
        observe=lambda _: x,
        expected_mutations={"value": expected},
        tolerances={"": (0, 0)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (128, 512)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
