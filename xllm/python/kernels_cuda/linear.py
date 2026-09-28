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

"""CUDA weight preparation for linear layers."""

from __future__ import annotations

import torch


def prepare_row_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Lay out a row-parallel weight for the CUDA matmul kernels.

    CUDA consumes the checkpoint layout ``[N, K]`` directly, so the weight is
    returned untouched.

    Args:
        weight: Row-parallel weight of shape ``[N, K]``.

    Returns:
        The weight and whether it was transposed to ``[K, N]``.
    """
    return weight, False


def prepare_quant_weight(weight: torch.Tensor) -> torch.Tensor:
    """Return quantized weights in the ``[K, N]`` matmul layout."""
    return weight.transpose(0, 1).contiguous()


__all__ = [
    "prepare_row_parallel_weight",
    "prepare_quant_weight",
]
