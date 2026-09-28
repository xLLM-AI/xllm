# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CUDA quantization kernels.

None of these have a CUDA kernel yet. The signatures are the contract a CUDA
implementation has to meet; see ``kernels_npu/quantization.py`` for the
reference behaviour.
"""

from __future__ import annotations

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
    """Multiply two quantized matrices and dequantize the result.

    Args:
        x1: Quantized activations of shape ``[..., K]``.
        x2: Quantized weight, ``[K, N]`` or ``[N, K]`` when ``transpose2``.
        transpose2: Whether ``x2`` is stored transposed.
        scale: Per-channel dequantization scale of the product.
        offset: Per-channel dequantization offset, or ``None``.
        pertoken_scale: Per-token activation scale, or ``None``.
        bias: Bias added after dequantization, or ``None``.
        output_dtype: Result dtype; ``None`` keeps the quantized dtype.

    Returns:
        The product with shape ``[..., N]`` in ``output_dtype``.
    """
    del x1, x2, transpose2, scale, offset, pertoken_scale, bias, output_dtype
    raise NotImplementedError("quant_matmul has no CUDA kernel; quantized linear layers on CUDA are not supported yet")


def quantize_per_tensor(
    value: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    dtype: torch.dtype,
    axis: int,
) -> torch.Tensor:
    """Quantize a tensor with statically known scales.

    Args:
        value: Tensor to quantize.
        scales: Quantization scale.
        zero_points: Quantization zero point.
        dtype: Quantized dtype.
        axis: Axis the scale applies along.

    Returns:
        The quantized tensor with the shape of ``value`` and dtype ``dtype``.
    """
    del value, scales, zero_points, dtype, axis
    raise NotImplementedError(
        "quantize_per_tensor has no CUDA kernel; quantized linear layers on CUDA are not supported yet"
    )


def dynamic_quant(
    value: torch.Tensor,
    smooth_scales: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    dst_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Quantize a tensor with scales computed from the tensor itself.

    Args:
        value: Tensor to quantize.
        smooth_scales: Per-channel smoothing applied before quantization.
        group_index: Expert or group each row belongs to, for grouped smoothing.
        dst_type: Quantized dtype; ``None`` selects ``torch.int8``.

    Returns:
        The quantized tensor and its per-token scale.
    """
    del value, smooth_scales, group_index, dst_type
    raise NotImplementedError("dynamic_quant has no CUDA kernel; quantized linear layers on CUDA are not supported yet")


__all__ = [
    "quant_matmul",
    "quantize_per_tensor",
    "dynamic_quant",
]
