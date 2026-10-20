# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""quant_matmul_out inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.quant_matmul_out import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, inner, channels = dimensions
    value = inputs.integer(-16, 17, rows, inner, dtype=torch.int8)
    weight = inputs.integer(-16, 17, channels, inner, dtype=torch.int8)
    ws = inputs.normal(channels, dtype=torch.float32).abs().mul_(0.01).add_(0.01)
    xs = inputs.normal(rows, dtype=torch.float32).abs().mul_(0.01).add_(0.01)
    accumulated = value.int() @ weight.int().T
    expected = accumulated if dtype == "int32" else (accumulated.float() * ws * xs[:, None]).to(inputs.dtype)
    x, w, s, t = value.npu(), weight.npu(), ws.npu(), xs.npu()
    out = torch.empty_like(expected, device=x.device)
    return Example(
        {"value": x, "weight": w, "weight_scale": s, "activation_scale": t, "output": out},
        lambda function: function(x, w, s, None if dtype == "int32" else t, out),
        expected,
        expected_mutations={"output": expected},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "inners": (6144, 2048), "channels": (256, 512)},
    _example,
    dtypes=("bfloat16", "float16", "int32"),
    restrict=_restrict,
)
