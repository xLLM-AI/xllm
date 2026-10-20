# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""fused_add_rms_norm_dynamic_quant inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.normalization_reference import rms
from xllm_kernel.numerics.quantization_reference import symmetric_int8
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.fused_add_rms_norm_dynamic_quant import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    inputs = Inputs(dtype, seed)
    value, residual, weight = inputs.normal(*dimensions), inputs.normal(*dimensions), inputs.normal(dimensions[-1])
    summed = value.float() + residual.float()
    quant, scale = symmetric_int8(rms(summed, weight, 1e-5))
    x, r, w = value.npu(), residual.npu(), weight.npu()
    return Example(
        {"value": x, "residual": r, "weight": w},
        lambda function: function(x, r, w, 1e-5),
        (quant, scale, summed.to(value.dtype)),
        tolerances={"/0": (0, 1), "/1": (2e-3, 1e-6)},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR, {"rows": (1, 16, 128), "widths": (6144,)}, _example, dtypes=("bfloat16", "float16"), restrict=_restrict
)
