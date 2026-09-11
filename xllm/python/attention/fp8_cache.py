# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Software E4M3 encoding for NPU caches.

The current NPU cast operator cannot consume or produce native PyTorch FP8
tensors. Cache tensors therefore store the IEEE-like E4M3FN bit pattern in a
``torch.uint8`` tensor. These helpers encode floating-point values on cache
writes and explicitly decode them before attention computation.
"""

from __future__ import annotations

from functools import cache

import torch

_E4M3_MAX = 448.0
_E4M3_MIN_NORMAL = 2.0**-6
_E4M3_SUBNORMAL_SCALE = 2.0**9


def quantize_e4m3(value: torch.Tensor) -> torch.Tensor:
    """Encode a floating-point tensor as saturated E4M3FN bytes."""
    if not value.is_floating_point():
        raise TypeError("E4M3 cache quantization requires a floating-point tensor")

    working = value.to(torch.float32)
    absolute = torch.nan_to_num(
        working.abs(),
        nan=0.0,
        posinf=_E4M3_MAX,
        neginf=_E4M3_MAX,
    ).clamp(max=_E4M3_MAX)
    sign_bits = (working < 0).to(torch.int32) * 128

    is_normal = absolute >= _E4M3_MIN_NORMAL
    exponent = torch.floor(torch.log2(absolute.clamp_min(_E4M3_MIN_NORMAL))).to(torch.int32)
    exponent_value = torch.exp2(exponent.to(torch.float32))
    mantissa = torch.round((absolute / exponent_value - 1.0) * 8.0).to(torch.int32)
    carry = mantissa == 8
    exponent = exponent + carry.to(torch.int32)
    mantissa = torch.where(carry, torch.zeros_like(mantissa), mantissa)
    normal_bits = (exponent + 7) * 8 + mantissa

    subnormal_bits = torch.round(absolute * _E4M3_SUBNORMAL_SCALE).to(torch.int32)
    magnitude_bits = torch.where(is_normal, normal_bits, subnormal_bits).clamp(0, 126)
    return (sign_bits + magnitude_bits).to(torch.uint8)


def dequantize_e4m3(value: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode E4M3FN bytes to ``dtype`` for attention computation."""
    if value.dtype != torch.uint8:
        raise TypeError("E4M3 cache dequantization requires a torch.uint8 tensor")

    if value.device.type == "npu":
        # Avoid cache-sized arithmetic intermediates retained by ACL graphs.
        table = _get_e4m3_decode_table(value.device, dtype)
        indices = value.reshape(-1).to(torch.int32)
        return torch.index_select(table, 0, indices).reshape(value.shape)

    bits = value.to(torch.int32)
    is_negative = bits >= 128
    magnitude_bits = (bits & 127).clamp_max(126)
    exponent = magnitude_bits >> 3
    mantissa = magnitude_bits & 7

    subnormal = mantissa.to(torch.float32) * (2.0**-9)
    normal = (8 + mantissa).to(torch.float32) * torch.exp2((exponent - 10).to(torch.float32))
    decoded = torch.where(exponent == 0, subnormal, normal)
    decoded = torch.where(is_negative, -decoded, decoded)
    return decoded.to(dtype)


def create_e4m3_decode_table(device: torch.device) -> torch.Tensor:
    """Create the 256-entry FP32 lookup table consumed by fused NPU kernels."""
    encoded = torch.arange(256, dtype=torch.int32).to(torch.uint8)
    return dequantize_e4m3(encoded, torch.float32).to(device)


@cache
def _get_e4m3_decode_table(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return create_e4m3_decode_table(device).to(dtype)


__all__ = ["create_e4m3_decode_table", "dequantize_e4m3", "quantize_e4m3"]
