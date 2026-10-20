# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""grouped_matmul inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.grouped_matmul import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch
    import torch_npu

    inputs = Inputs(dtype, seed)
    rows, inner, channels, experts = dimensions
    value = inputs.integer(-8, 9, rows, inner, dtype=torch.int8)
    weight = inputs.integer(-8, 9, experts, inner, channels, dtype=torch.int8)
    ws = inputs.normal(experts, channels).abs().mul_(0.01).add_(0.01).to(torch.bfloat16)
    xs = inputs.normal(rows, dtype=torch.float32).abs().mul_(0.01).add_(0.01)
    # Uneven group sizes and empty experts are intentional.
    ids = torch.arange(rows) % experts
    counts = torch.bincount(ids, minlength=experts)
    groups = counts.cumsum(0).to(torch.int64)
    reference = torch.empty(rows, channels, dtype=torch.float32)
    start = 0
    for expert, count in enumerate(counts.tolist()):
        end = start + count
        reference[start:end] = (
            (value[start:end].int() @ weight[expert].int()).float() * ws[expert].float() * xs[start:end, None]
        )
        start = end
    torch.npu.config.allow_internal_format = True
    x = value.npu()
    w = torch_npu.npu_format_cast(weight.transpose(1, 2).contiguous().npu(), 29).transpose(1, 2)
    s, t, g = ws.npu(), xs.npu(), groups.npu()
    expected = reference.to(torch.bfloat16)
    return Example(
        {"value": x, "weight": w, "weight_scale": s, "activation_scale": t, "groups": g},
        lambda function: function(x, w, s, t, g),
        expected,
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (16, 128), "inners": (2048,), "channels": (6144,), "experts": (4,)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
