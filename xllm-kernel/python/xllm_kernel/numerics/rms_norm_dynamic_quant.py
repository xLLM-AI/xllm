# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""rms_norm_dynamic_quant inputs and independent CPU reference for accuracy and performance."""

from __future__ import annotations

from typing import Any

from xllm_kernel.numerics.factory import tensor_suite
from xllm_kernel.numerics.inputs import Inputs
from xllm_kernel.numerics.normalization_reference import rms
from xllm_kernel.numerics.quantization_reference import symmetric_int8
from xllm_kernel.numerics.tensor_case import Example
from xllm_kernel.ops.rms_norm_dynamic_quant import OPERATOR
from xllm_kernel.registry import PreparedKernel


def _example(dimensions: tuple[int, ...], dtype: str, seed: int) -> Example:
    inputs = Inputs(dtype, seed)
    value, weight = inputs.normal(*dimensions), inputs.normal(dimensions[-1])
    expected = symmetric_int8(rms(value, weight, 1e-5))
    x, w = value.npu(), weight.npu()
    return Example(
        {"value": x, "weight": w},
        lambda function: function(x, w, 1e-5),
        expected,
        tolerances={"/0": (0, 1), "/1": (2e-3, 1e-6)},
        metadata={"cann_composition": "AddRmsNormDynamicQuant with zero residual; zero creation is timed"},
    )


def _restrict(plan: PreparedKernel, dimensions: tuple[int, ...], dtype: str) -> str | None:
    return None


SUITE = tensor_suite(
    OPERATOR,
    {"rows": (1, 16, 128), "widths": (6144, 2048)},
    _example,
    dtypes=("bfloat16", "float16"),
    restrict=_restrict,
)
