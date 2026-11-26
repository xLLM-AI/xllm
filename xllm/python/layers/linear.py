# Copyright 2025-2026 The xLLM Authors.
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

"""Tensor-parallel linear layers.

At ``tp_size==1`` these hold full-size weights and skip all collectives, so they
are numerically identical to plain ``nn.Linear`` and preserve the single-card
byte parity. At ``tp_size>1`` each rank holds a per-partition shard and inserts
the same all-reduce / all-gather the native C++ parallel layers use (via
:mod:`python.distributed`).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import distributed, kernels


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
        prepare_weight_layout: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.gather_output = gather_output
        self._prepare_weight_layout = prepare_weight_layout
        self._weight_is_transposed = False
        self._prepared_weight: torch.Tensor | None = None
        self.weight = nn.Parameter(
            torch.empty(
                out_features_per_partition,
                in_features,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features_per_partition, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def process_weights_after_loading(self) -> None:
        """Prepare the backend-selected weight layout once after loading."""
        if not self._prepare_weight_layout or self._weight_is_transposed:
            return
        prepared, is_transposed = kernels.prepare_column_parallel_weight(self.weight.data)
        if (
            self._prepared_weight is not None
            and self._prepared_weight.shape == prepared.shape
            and self._prepared_weight.dtype == prepared.dtype
            and self._prepared_weight.device == prepared.device
        ):
            self._prepared_weight.copy_(prepared)
        else:
            self._prepared_weight = prepared
        self.weight.data = self._prepared_weight
        self._weight_is_transposed = is_transposed

    def prepare_weight_for_loading(self) -> None:
        """Restore checkpoint shape before a supported in-place weight reload."""
        if not self._weight_is_transposed:
            return
        self.weight.data = self.weight.new_empty(self.weight.shape[1], self.weight.shape[0])
        self._weight_is_transposed = False

    def _forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_transposed:
            out = torch.matmul(x, self.weight)
            return out if self.bias is None else out + self.bias
        return torch.nn.functional.linear(x, self.weight, self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_projection(x)
        if self.gather_output and self.tp_size > 1:
            out = distributed.tp_all_gather(out, dim=-1, world_size=self.tp_size)
        return out

    def greedy_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Select global token IDs without gathering the vocabulary logits."""
        logits = self._forward_projection(x)
        if self.tp_size == 1:
            return logits.argmax(dim=-1)
        # Pack both candidates in one collective. A float32 exactly represents
        # every local vocabulary index in this supported range, and every
        # fp16/bf16/fp32 score. Global IDs are assembled in int64 afterwards.
        partition_size = self.weight.shape[1] if self._weight_is_transposed else self.weight.shape[0]
        if partition_size > 2**24 or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError("greedy TP candidates require fp16/bf16/fp32 logits and at most 2**24 local tokens")
        scores, indices = logits.max(dim=-1)
        del logits
        candidates = torch.stack((scores.float(), indices.float()), dim=-1)
        candidates = distributed.tp_all_gather(candidates, dim=-1, world_size=self.tp_size)
        candidates = candidates.view(*indices.shape, self.tp_size, 2)
        # Rank-ordered contiguous shards make the first winning rank/token
        # exactly the first global argmax, including ties and NaN scores.
        winner = candidates[..., 0].argmax(dim=-1)
        # The winner identifies the rank whose local index is needed. Gather
        # that one value directly instead of building a rank range and doing a
        # masked sum. This keeps the candidate communication and the original
        # argmax tie/NaN behavior while removing two small decode-side ops.
        local_ids = candidates[..., 1].gather(-1, winner.unsqueeze(-1)).squeeze(-1).to(torch.long)
        return local_ids + winner * partition_size


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
        reduce_results: bool = True,
        use_checkpoint_layout: bool = False,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self._weight_is_transposed = False
        self._use_checkpoint_layout = use_checkpoint_layout
        self.reduce_results = reduce_results
        if bias and not reduce_results:
            # The bias is replicated and must be added exactly once, which is
            # only possible here when this layer owns the reduction.
            raise ValueError("a deferred reduction cannot be combined with a replicated bias")
        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features_per_partition,
                dtype=dtype,
                device=device,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def process_weights_after_loading(self) -> None:
        """Prepare the weight layout selected by the active device backend."""
        if self._use_checkpoint_layout or self._weight_is_transposed:
            return
        prepared, is_transposed = kernels.prepare_row_parallel_weight(self.weight.data)
        self.weight.data = prepared
        self._weight_is_transposed = is_transposed

    def _forward_matmul(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_is_transposed:
            return torch.matmul(x, self.weight)
        return torch.nn.functional.linear(x, self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self._forward_matmul(x)
        if self.tp_size > 1 and self.reduce_results:
            distributed.tp_all_reduce(out)
        if self.bias is not None:
            out = out + self.bias
        return out
