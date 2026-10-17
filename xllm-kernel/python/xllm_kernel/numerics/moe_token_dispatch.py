# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_token_dispatch inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.quantization_reference import symmetric_int8
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.moe_token_dispatch import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, width, experts, topk = dimensions
    value = inputs.normal(rows, width)
    ids = inputs.normal(rows, experts, dtype=torch.float32).topk(topk, -1).indices.to(torch.int32)
    order = ids.flatten().argsort(stable=True)
    inverse = order.argsort().to(torch.int32)
    q, scales = symmetric_int8(value)
    expected = (
        q[order // topk],
        inverse,
        torch.bincount(ids.flatten().long(), minlength=experts).cumsum(0),
        scales[order // topk],
    )

    def canonical(result: Any) -> Any:
        quant, index, groups, scale = (item.cpu() for item in result)
        index = index.long()
        return quant[index], index.sort().values, groups, scale[index], ids.flatten()[index.argsort()]

    x, i = value.npu(), ids.npu()
    return Example(
        {"value": x, "expert_ids": i},
        lambda function: function(x, i, experts),
        expected,
        canonicalize=canonical,
        tolerances={"/0": (0, 1), "/3": (1e-3, 1e-6)},
        metadata={"group_list": "cumulative", "row_idx_type": "gather", "native_conversion_in_timing": True},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return "topk must not exceed experts" if dimensions[-1] > dimensions[-2] else None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "widths": (6144,), "experts": (256,), "topks": (8,)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
