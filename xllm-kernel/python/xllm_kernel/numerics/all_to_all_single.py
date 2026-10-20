# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_to_all_single inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.all_to_all_single import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch
    import torch.distributed as dist

    inputs = Inputs(dtype, seed)
    rank, world = dist.get_rank(), dist.get_world_size()
    base = inputs.normal(world, *dimensions, scale=0.01)
    value = base * (rank + 1)
    expected = torch.stack([base[rank] * (source + 1) for source in range(world)])
    x = value.npu()
    out = torch.empty_like(x)
    return Example(
        {"value": x, "output": out},
        lambda function: function(x, out),
        expected,
        expected_mutations={"output": expected},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
