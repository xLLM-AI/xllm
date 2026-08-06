# Copyright 2025-2026 The xLLM Authors.
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

"""Tensor-parallel linear layers.

At ``tp_size==1`` these hold full-size weights and skip all collectives, so they
are numerically identical to plain ``nn.Linear`` and preserve the single-card
byte parity. At ``tp_size>1`` each rank holds a per-partition shard and inserts
the same all-reduce / all-gather the native C++ parallel layers use (via the op
dispatch layer :mod:`python.ops`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import ops


class ColumnParallelLinear(nn.Module):
    """Linear sharded on the output dim (dim 0): each rank owns
    ``[out_per_partition, in]`` and computes its slice of the output. No
    communication unless ``gather_output`` (then an all-gather along the last
    dim reconstructs the full output — used by lm_head). An optional bias is
    sharded on the output dim like the weight and applied per partition (before
    any gather). Mirrors native ColumnParallelLinear / QKVParallelLinear (which
    set gather_output=False so the following RowParallel all-reduce combines the
    partial outputs).
    """

    def __init__(
        self,
        in_features: int,
        out_features_per_partition: int,
        tp_size: int,
        gather_output: bool = False,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output
        self.weight = nn.Parameter(
            torch.empty(
                out_features_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features_per_partition, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.linear(x, self.weight, self.bias)
        if self.gather_output and self.tp_size > 1:
            out = ops.all_gather(out, dim=-1, world_size=self.tp_size)
        return out


class RowParallelLinear(nn.Module):
    """Linear sharded on the input dim (dim 1): each rank owns
    ``[out, in_per_partition]`` and consumes its slice of an already-partitioned
    input, producing a partial output that is SUM all-reduced across the TP
    group. An optional bias is replicated (full ``out``) and added once AFTER
    the all-reduce, so it is not summed ``tp_size`` times. Mirrors native
    RowParallelLinear (o_proj / down_proj with enable_result_reduction=true).
    """

    def __init__(
        self,
        in_features_per_partition: int,
        out_features: int,
        tp_size: int,
        bias: bool = False,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self._weight_is_transposed = False
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features_per_partition,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, dtype=dtype, device=device)
            )
        else:
            self.register_parameter("bias", None)

    def format_npu_weight_(self) -> None:
        """Store the weight as ``[K, N]`` FRACTAL_NZ for non-transposed matmul."""
        if self.weight.device.type not in ("npu", "privateuseone"):
            return
        if self._weight_is_transposed:
            return
        import torch_npu

        transposed = self.weight.data.transpose(0, 1).contiguous()
        self.weight.data = torch_npu.npu_format_cast(transposed, 29)
        self._weight_is_transposed = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_transposed:
            out = torch.matmul(x, self.weight)
        else:
            out = torch.nn.functional.linear(x, self.weight)
        if self.tp_size > 1:
            ops.all_reduce_(out)
        if self.bias is not None:
            out = out + self.bias
        return out
