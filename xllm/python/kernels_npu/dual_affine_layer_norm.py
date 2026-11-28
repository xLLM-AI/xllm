# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dual-affine LayerNorm acceptance entry using the custom ACLNN operator."""

import math
from collections.abc import Callable, Sequence

import torch


def _epsilon(attributes: dict[str, float]) -> float:
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dictionary")
    if attributes.keys() - {"epsilon"}:
        raise ValueError("unknown DualAffineLayerNorm attribute")
    epsilon = attributes.get("epsilon", 1e-5)
    if type(epsilon) not in (int, float):
        raise TypeError("epsilon must be a real number")
    if not math.isfinite(epsilon) or not 1e-7 <= epsilon <= 1e-2:
        raise ValueError("epsilon must be in [1e-7,1e-2]")
    return float(epsilon)


def dual_affine_layer_norm(
    x: torch.Tensor,
    gamma1: torch.Tensor,
    beta1: torch.Tensor,
    gamma2: torch.Tensor,
    beta2: torch.Tensor,
    epsilon: float = 1e-5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize once in FP32 and return two independent affine outputs."""
    return torch.ops.xllm_ops.dual_affine_layer_norm(x, gamma1, beta1, gamma2, beta2, epsilon)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Allocate two independent outputs and execute on the current NPU stream."""
    if len(inputs) != 5:
        raise ValueError("DualAffineLayerNorm requires five inputs")
    return torch.ops.xllm_ops.dual_affine_layer_norm(*inputs, _epsilon(attributes))


def prepare(
    inputs: Sequence[torch.Tensor], attributes: dict[str, float]
) -> Callable[[], tuple[torch.Tensor, torch.Tensor]]:
    """Validate the call without retaining any output allocation."""
    epsilon = _epsilon(attributes)
    if len(inputs) != 5:
        raise ValueError("DualAffineLayerNorm requires five inputs")
    x, *parameters = inputs
    if x.device.type != "npu" or x.ndim != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous NPU matrix")
    if x.dtype not in (torch.float16, torch.bfloat16) or not (1 <= x.shape[0] <= 4096 and 1 <= x.shape[1] <= 8192):
        raise ValueError("unsupported dtype or matrix size")
    for tensor in parameters:
        if (
            tensor.device != x.device
            or tensor.dtype != x.dtype
            or tensor.shape != (x.shape[1],)
            or not tensor.is_contiguous()
        ):
            raise ValueError("invalid affine tensor")
    return lambda: torch.ops.xllm_ops.dual_affine_layer_norm(*inputs, epsilon)


def run_direct(inputs: Sequence[torch.Tensor], attributes: dict[str, float]) -> tuple[torch.Tensor, torch.Tensor]:
    """Cross-check the shared kernel via the separately built direct library."""
    prepare(inputs, attributes)
    return torch.ops.xllm_ops.dual_affine_layer_norm_direct(*inputs, _epsilon(attributes))
