# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""reshape_paged_cache inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.reshape_paged_cache import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, kdim, vdim = dimensions
    blocks = (rows + 15) // 16 + 1
    key_cache = inputs.normal(blocks, 16, 1, kdim)
    value_cache = inputs.normal(blocks, 16, 1, vdim)
    keys, values = inputs.normal(rows, 1, kdim), inputs.normal(rows, 1, vdim)
    slots = torch.randperm(blocks * 16, generator=inputs.generator)[:rows].to(torch.int32)
    expected_key, expected_value = key_cache.clone(), value_cache.clone()
    for row, slot in enumerate(slots.tolist()):
        expected_key[slot // 16, slot % 16] = keys[row]
        expected_value[slot // 16, slot % 16] = values[row]
    i, k, v, kc, vc = slots.npu(), keys.npu(), values.npu(), key_cache.npu(), value_cache.npu()
    return Example(
        {"slots": i, "keys": k, "values": v, "key_cache": kc, "value_cache": vc},
        lambda function: function(i, k, v, kc, vc),
        (expected_key, expected_value),
        observe=lambda _: (kc, vc),
        expected_mutations={"key_cache": expected_key, "value_cache": expected_value},
        tolerances={"/0": (0, 0), "/1": (0, 0)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "key_widths": (64,), "value_widths": (512,)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
