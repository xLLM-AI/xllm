# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Public quantization operators and FakeTensor contracts."""

from __future__ import annotations

from collections.abc import Callable

import torch


def quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    transpose2: bool,
    scale: torch.Tensor,
    offset: torch.Tensor | None,
    pertoken_scale: torch.Tensor | None,
    bias: torch.Tensor | None,
    output_dtype: torch.dtype | None,
) -> torch.Tensor:
    return torch.ops.xllm_ops.quant_matmul(
        x1,
        x2,
        transpose2,
        scale,
        offset,
        pertoken_scale,
        bias,
        output_dtype,
    )


def quantize_per_tensor(
    value: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> torch.Tensor:
    return torch.ops.xllm_ops.quantize_per_tensor(
        value, scales, zero_points, dtype, axis
    )


def dynamic_quant(
    value: torch.Tensor,
    smooth_scales: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    dst_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return torch.ops.xllm_ops.dynamic_quant(
        value, smooth_scales, group_index, dst_type
    )


def _quant_matmul_fake(
    x1,
    x2,
    transpose2,
    scale,
    offset,
    pertoken_scale,
    bias,
    output_dtype,
):
    del scale, offset, pertoken_scale, bias
    out_last = x2.size(0) if transpose2 else x2.size(1)
    out_shape = list(x1.shape[:-1]) + [out_last]
    dtype = output_dtype if output_dtype is not None else torch.int8
    return x1.new_empty(out_shape, dtype=dtype)


def _quantize_per_tensor_fake(value, scales, zero_points, dtype, axis):
    del scales, zero_points, axis
    return value.new_empty(value.shape, dtype=dtype)


def _dynamic_quant_fake(value, smooth_scales, group_index, dst_type):
    del smooth_scales, group_index
    if dst_type == torch.quint4x2:
        if value.shape[-1] % 8:
            raise ValueError(
                "dynamic_quant int4 input's last dimension must be divisible by 8"
            )
        output_shape = (*value.shape[:-1], value.shape[-1] // 8)
        output_dtype = torch.int32
    else:
        output_shape = value.shape
        output_dtype = torch.int8
    output = value.new_empty(output_shape, dtype=output_dtype)
    scale = value.new_empty(value.shape[:-1], dtype=torch.float32)
    return output, scale


def _register_fake_if_available(name: str, fake: Callable) -> None:
    namespace, op_name = name.split("::", 1)
    if hasattr(getattr(torch.ops, namespace), op_name):
        torch.library.register_fake(name)(fake)


_register_fake_if_available("xllm_ops::quant_matmul", _quant_matmul_fake)
_register_fake_if_available(
    "xllm_ops::quantize_per_tensor", _quantize_per_tensor_fake
)
_register_fake_if_available("xllm_ops::dynamic_quant", _dynamic_quant_fake)

__all__ = ["quant_matmul", "quantize_per_tensor", "dynamic_quant"]
