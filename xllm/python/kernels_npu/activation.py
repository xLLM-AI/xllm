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

"""NPU activation kernels."""

from __future__ import annotations

import torch
import torch_npu

silu_and_mul = torch.ops.xllm_ops.silu_and_mul
_DEQUANT_SWIGLU_QUANT = torch_npu.npu_dequant_swiglu_quant


def dequant_swiglu_quant(
    value: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dequantize INT32 Gate-Up output, apply SwiGLU, and requantize."""
    return _DEQUANT_SWIGLU_QUANT(
        value,
        weight_scale=weight_scale,
        activation_scale=activation_scale,
        bias=None,
        quant_scale=None,
        quant_offset=None,
        group_index=None,
        activate_left=True,
        quant_mode=1,
        swiglu_mode=0,
        clamp_limit=7.0,
    )


__all__ = [
    "silu_and_mul",
    "dequant_swiglu_quant",
]
