# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""dequant_swiglu_quant inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.quantization_reference import symmetric_int8
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.dequant_swiglu_quant import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    import torch

    inputs = Inputs(dtype, seed)
    rows, width, variant = dimensions
    value = inputs.integer(-1000, 1001, rows, width, dtype=torch.int32)
    ws = inputs.normal(width, dtype=torch.float32).abs().mul_(0.02).add_(0.01)
    xs = inputs.normal(rows, dtype=torch.float32).abs().mul_(0.02).add_(0.01)
    dequant = value.float() * ws * xs[:, None]
    mode, limit = ((0, 0.0), (1, 7.0), (1, 0.0))[variant - 1]
    if mode == 0:
        gate, up = dequant.chunk(2, -1)
    else:
        gate, up = dequant[..., ::2].clamp(max=limit), dequant[..., 1::2].clamp(-limit, limit)
    expected = symmetric_int8(torch.nn.functional.silu(gate) * up)
    x, w, s = value.npu(), ws.npu(), xs.npu()
    return Example(
        {"value": x, "weight_scale": w, "activation_scale": s},
        lambda function: function(x, w, s, mode, limit),
        expected,
        tolerances={"/0": (0, 1), "/1": (2e-3, 1e-6)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "widths": (4096,), "variants": (1, 2, 3)},
    _example,
    dtypes=("bfloat16",),
    restrict=_restrict,
)
