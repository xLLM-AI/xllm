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

"""Tensor-parallel gated MLP shared by Python model implementations."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import kernels
from xllm.python.layers.linear import ColumnParallelLinear, RowParallelLinear


class GatedMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
        dtype: torch.dtype,
        device: torch.device,
        reduce_results: bool = True,
    ) -> None:
        super().__init__()
        if intermediate_size % tp_size:
            raise ValueError("intermediate_size must be divisible by tp_size")
        local_intermediate_size = intermediate_size // tp_size
        self.gate_up_proj = ColumnParallelLinear(
            hidden_size,
            2 * local_intermediate_size,
            tp_size,
            dtype=dtype,
            device=device,
        )
        self.down_proj = RowParallelLinear(
            local_intermediate_size,
            hidden_size,
            tp_size,
            dtype=dtype,
            device=device,
            reduce_results=reduce_results,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(kernels.silu_and_mul(self.gate_up_proj(hidden_states)))
