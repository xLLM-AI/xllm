# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""sparse_flash_attention inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.sparse_flash_attention import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, heads, width, length, topk = dimensions
    block, rope = 128, 64
    count = (length + block - 1) // block
    query = inputs.normal(rows, heads, width, scale=0.25)
    qr = inputs.normal(rows, heads, rope, scale=0.25)
    logical = inputs.normal(count * block, 1, width, scale=0.25)
    logical_rope = inputs.normal(count * block, 1, rope, scale=0.25)
    table = torch.randperm(count, generator=inputs.generator).to(torch.int32).view(1, count)
    cache = torch.empty(count, block, 1, width, dtype=inputs.dtype)
    kr = torch.empty(count, block, 1, rope, dtype=inputs.dtype)
    cache[table[0].long()] = logical.view_as(cache)
    kr[table[0].long()] = logical_rope.view_as(kr)
    indices = (
        torch.stack([torch.randperm(length, generator=inputs.generator)[:topk] for _ in range(rows)])
        .to(torch.int32)
        .unsqueeze(1)
    )
    indices[..., ::11] = -1
    scale = 256**-0.5
    output = torch.empty_like(query)
    maxima = torch.empty(1, rows, heads, dtype=torch.float32)
    sums = torch.empty_like(maxima)
    for row in range(rows):
        ids = indices[row, 0].long()
        ids = ids[(ids >= 0) & (ids < length - rows + row + 1)]
        logits = (
            query[row].float() @ logical[ids, 0].float().T + qr[row].float() @ logical_rope[ids, 0].float().T
        ) * scale
        maxima[0, row] = logits.max(-1).values
        exp = (logits - maxima[0, row, :, None]).exp()
        sums[0, row] = exp.sum(-1)
        output[row] = (exp / sums[0, row, :, None]) @ logical[ids, 0].float()
    expected = (output, maxima, sums)
    q, k, i, bt, r, krn = query.npu(), cache.npu(), indices.npu(), table.npu(), qr.npu(), kr.npu()
    ql = torch.tensor([rows], dtype=torch.int32).npu()
    kl = torch.tensor([length], dtype=torch.int32).npu()

    def canonical(result: Any) -> Any:
        value, maximum, total = result
        return value, maximum + total.log()

    return Example(
        {
            "query": q,
            "cache": k,
            "indices": i,
            "query_lengths": ql,
            "key_lengths": kl,
            "block_table": bt,
            "query_rope": r,
            "key_rope": krn,
        },
        lambda function: function(q, k, i, ql, kl, bt, r, krn, scale),
        expected,
        canonicalize=canonical,
        tolerances={"/0": (2e-2, 2e-3), "/1": (2e-3, 2e-3)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    rows, heads, width, length, topk = dimensions
    if rows > length or topk > length or topk < 16:
        return "require rows<=length, 16<=topk<=length"
    if plan.spec.solution == "aclnn" and (width != 512 or topk != 2048):
        return "native GLM profile requires latent width=512 and topk=2048"
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16), "heads": (4,), "widths": (512,), "lengths": (4096,), "topks": (2048,)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
