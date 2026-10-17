# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""moe_gate_routing inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.moe_gate_routing import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _canonical(result: Any) -> Any:
    weights, indices = result
    order = indices.argsort(-1)
    return weights.gather(-1, order), indices.gather(-1, order)


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, experts, topk, groups = dimensions
    logits = inputs.normal(rows, experts, dtype=torch.float32)
    bias = inputs.normal(experts, dtype=torch.float32, scale=0.1)
    scores = logits.sigmoid()
    selection = scores + bias
    selected_groups = max(1, groups // 2)
    group_scores = selection.view(rows, groups, -1).topk(2, dim=-1).values.sum(-1)
    mask = torch.zeros_like(group_scores, dtype=torch.bool).scatter_(
        1, group_scores.topk(selected_groups, -1).indices, True
    )
    selection = selection.masked_fill(~mask.repeat_interleave(experts // groups, -1), float("-inf"))
    ids = selection.topk(topk, -1).indices
    weights = scores.gather(-1, ids)
    weights = weights / weights.sum(-1, keepdim=True) * 2.5
    x, b = logits.npu(), bias.npu()
    return Example(
        {"logits": x, "bias": b},
        lambda function: function(x, b, topk, groups, selected_groups, 2.5),
        (weights, ids.to(torch.int32)),
        canonicalize=_canonical,
        tolerances={"/0": (1e-4, 1e-5)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    _, experts, topk, groups = dimensions
    if plan.spec.solution == "xlite" and (experts != 256 or topk > 32 or groups not in (1, 8)):
        return "xlite pinned specialization requires E=256, topk<=32, groups=1/8"
    if experts % groups or experts // groups < 2 or topk > max(1, groups // 2) * (experts // groups):
        return "groups must partition experts and retain enough candidates for topk"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "experts": (256,), "topks": (8,), "groups": (1, 8)},
    _example,
    dtypes=("float32",),
    restrict=_restrict,
)
