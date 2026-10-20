# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""all_reduce inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.all_reduce import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch.distributed as dist

    inputs = Inputs(dtype, seed)
    rank, world = dist.get_rank(), dist.get_world_size()
    base = inputs.normal(*dimensions, scale=0.01)
    # Sum the rounded rank-local tensors, not a rounded sum of unrounded values.
    reference = sum((base * (other + 1)).float() for other in range(world)).to(base.dtype)
    x = (base * (rank + 1)).npu()
    return Example(
        {"value": x},
        lambda function: function(x),
        reference,
        expected_mutations={"value": reference},
        metadata={"rank": rank, "world_size": world},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
