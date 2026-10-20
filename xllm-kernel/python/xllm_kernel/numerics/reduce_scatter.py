# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reduce_scatter inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.reduce_scatter import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch
    import torch.distributed as dist

    inputs = Inputs(dtype, seed)
    rank, world = dist.get_rank(), dist.get_world_size()
    base = inputs.normal(world, *dimensions, scale=0.01)
    reference = sum((base[rank] * (other + 1)).float() for other in range(world)).to(base.dtype)
    value = base * (rank + 1)
    x = value.npu()
    out = torch.empty_like(reference, device=x.device)
    return Example(
        {"value": x, "output": out},
        lambda function: function(x, out),
        reference,
        expected_mutations={"output": reference},
        metadata={"rank": rank, "world_size": world},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
