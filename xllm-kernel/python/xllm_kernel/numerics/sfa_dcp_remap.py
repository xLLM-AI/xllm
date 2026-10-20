# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sfa_dcp_remap inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.sfa_dcp_remap import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, topk, block, shards = dimensions
    indices = inputs.integer(-1, 65536, rows, 1, topk, dtype=torch.int32)
    indices[..., ::13] = -1
    rank = shards - 1
    expected = torch.full_like(indices, -1)
    for row in range(rows):
        kept = []
        for token in indices[row, 0].tolist():
            if token >= 0 and (token // block) % shards == rank:
                kept.append(token // (block * shards) * block + token % block)
        expected[row, 0, : len(kept)] = torch.tensor(kept, dtype=torch.int32)
    x = indices.npu()
    return Example({"indices": x}, lambda function: function(x, block, shards, rank), expected)


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    if plan.spec.solution == "tilelang" and (dimensions[0] > 256 or dimensions[1] != 2048):
        return "TileLang AOT variant requires at most 256 rows and topk=2048"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "topks": (2048,), "blocks": (128,), "shards": (2, 4)},
    _example,
    dtypes=("int32",),
    restrict=_restrict,
)
