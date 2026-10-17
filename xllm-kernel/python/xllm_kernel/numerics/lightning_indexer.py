# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""lightning_indexer inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.lightning_indexer import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, heads, width, length, topk = dimensions
    block = 128
    count = (length + block - 1) // block
    query = inputs.normal(rows, heads, width, scale=0.25)
    logical = inputs.normal(count * block, 1, width, scale=0.25)
    weights = inputs.normal(rows, heads, dtype=torch.float32).abs().mul_(0.1).add_(0.1).to(inputs.dtype)
    table = torch.randperm(count, generator=inputs.generator).to(torch.int32).view(1, count)
    cache = torch.empty(count, block, 1, width, dtype=inputs.dtype)
    cache[table[0].long()] = logical.view(count, block, 1, width)
    scores = torch.einsum("thd,kd->thk", query.float(), logical[:, 0].float()).relu()
    scores = (scores * weights.float()[..., None]).sum(1)
    positions = torch.arange(count * block)
    limits = length - rows + torch.arange(rows) + 1
    scores = scores.masked_fill(positions[None] >= limits[:, None], float("-inf"))
    expected = scores.topk(topk, -1).indices.to(torch.int32).unsqueeze(1)

    # Score-based selection check accepts only numerical ties at the cutoff;
    # cardinality, uniqueness, bounds and causality are checked separately.
    def canonical(result: Any) -> Any:
        ids = result.cpu().long().squeeze(1)
        if ids.shape != (rows, topk) or bool((ids < 0).any()) or bool((ids >= limits[:, None]).any()):
            raise AssertionError("indexer returned invalid or noncausal indices")
        if any(row.unique().numel() != topk for row in ids):
            raise AssertionError("indexer returned duplicate indices")
        return scores.gather(1, ids).sort(-1, descending=True).values

    q, k, w = query.npu(), cache.npu(), weights.npu()
    ql = torch.tensor([rows], dtype=torch.int32).npu()
    kl = torch.tensor([length], dtype=torch.int32).npu()
    bt = table.npu()
    return Example(
        {"query": q, "key_cache": k, "weights": w, "query_lengths": ql, "key_lengths": kl, "block_table": bt},
        lambda function: function(q, k, w, ql, kl, bt, topk),
        expected,
        canonicalize=canonical,
        tolerances={"": (0.015, 1e-3)},
        metadata={
            "selection_check": "unique causal indices and sorted CPU scores within 1.5%; not exact index equality"
        },
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    rows, heads, width, length, topk = dimensions
    if plan.spec.solution == "xlite" and (rows > 128 or heads != 32 or width != 128 or topk != 2048):
        return "xlite indexer requires T<=128,H=32,D=128,K=2048"
    if topk > length - rows + 1:
        return "this profile requires at least topk visible tokens for every query"
    if plan.spec.solution == "aclnn" and (width != 128 or topk != 2048):
        return "native GLM profile requires head_dim=128 and topk=2048"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16), "heads": (32,), "widths": (128,), "lengths": (4096,), "topks": (2048,)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
