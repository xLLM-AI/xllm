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

"""CUDA activation kernels."""

from __future__ import annotations

import torch

silu_and_mul = torch.ops.xllm_ops.silu_and_mul


def dequant_swiglu_quant(
    value: torch.Tensor,
    weight_scale: torch.Tensor,
    activation_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Report that fused dequant-SwiGLU-quant is NPU-only."""
    del value, weight_scale, activation_scale
    raise NotImplementedError("dequant_swiglu_quant is available only on NPU")


@torch.library.custom_op("xllm_triton::silu_and_mul", mutates_args=())
def _silu_and_mul_triton(value: torch.Tensor) -> torch.Tensor:
    """Run the Triton gated SiLU kernel as one graph node.

    Kept alongside the C++ operator so a model graph can pick the Triton path
    explicitly through ``torch.ops.xllm_triton.silu_and_mul``.
    """
    from .triton.silu_and_mul import (
        silu_and_mul as triton_silu_and_mul,
    )

    return triton_silu_and_mul(value)


@_silu_and_mul_triton.register_fake
def _silu_and_mul_triton_fake(value: torch.Tensor) -> torch.Tensor:
    shape = list(value.shape)
    shape[-1] //= 2
    return value.new_empty(shape)


__all__ = ["silu_and_mul", "dequant_swiglu_quant"]
