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
from xllm_kernel.ops.activation import silu_and_mul

_DEQUANT_SWIGLU_QUANT = torch_npu.npu_dequant_swiglu_quant


def _check_output_gate(input: torch.Tensor, gate: torch.Tensor) -> None:
    torch._check(input.ndim == 2)
    torch._check(gate.ndim in (2, 3))
    torch._check(input.shape[0] == gate.shape[0])
    width = gate.shape[-1] if gate.ndim == 2 else gate.shape[1] * gate.shape[2]
    torch._check(input.shape[1] == width)
    torch._check(width > 0)
    torch._check(input.dtype in (torch.bfloat16, torch.float16, torch.float32))
    torch._check(input.dtype == gate.dtype)
    torch._check(input.device == gate.device)
    torch._check(input.device.type in ("npu", "privateuseone"))


@torch.library.custom_op("xllm_python::fused_sigmoid_mul", mutates_args=())
def fused_sigmoid_mul(input: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
    """Apply a fused Triton gate, adapting an optional head dimension."""
    _check_output_gate(input, gate)
    from .triton.fused_sigmoid_mul import fused_sigmoid_mul as triton_fused_sigmoid_mul

    with torch.npu.device(input.device):
        # Triton consumes ND storage; contiguous() alone does not unpack NZ.
        value = input
        if torch_npu.get_npu_format(value) != torch_npu.Format.ND:
            value = torch_npu.npu_format_cast(value, torch_npu.Format.ND)
        if torch_npu.get_npu_format(gate) != torch_npu.Format.ND:
            gate = torch_npu.npu_format_cast(gate, torch_npu.Format.ND)
        output = triton_fused_sigmoid_mul(value.reshape(gate.shape), gate)
    return output.reshape(input.shape)


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
    "fused_sigmoid_mul",
    "silu_and_mul",
    "dequant_swiglu_quant",
]
