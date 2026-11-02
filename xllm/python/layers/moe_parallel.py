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

"""Token-parallel ownership and EPLv2 communication admission."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from xllm.python import distributed


@dataclass(frozen=True)
class Eplv2CommPolicy:
    """One immutable policy per model; no device queries in the selector."""

    mc2_capacity: int
    mode: str = "auto"

    @classmethod
    def from_geometry(
        cls,
        hidden_size: int,
        num_experts: int,
        ep_size: int,
        topk: int,
        buffer_mb: int,
        token_limit: int = 512,
        mode: str = "auto",
    ) -> Eplv2CommPolicy:
        if mode not in ("auto", "mc2", "alltoall"):
            raise ValueError("EPLv2 communication mode must be auto, mc2 or alltoall")
        if buffer_mb <= 0 or not 0 < token_limit <= 512:
            raise ValueError("EPLv2 requires positive HCCL buffer MB and a MC2 token limit in [1, 512]")
        if ep_size <= 1 or num_experts <= 0 or num_experts % ep_size or not 0 < topk <= num_experts:
            raise ValueError("Invalid EPLv2 expert geometry")
        # A3 fullmesh/default EP window bound, including the default shared
        # slot. CANN specifies the window layout using two-byte activations
        # even when dispatch quantization is enabled; traffic compression does
        # not halve this allocation requirement.
        local_experts = num_experts // ep_size
        dispatch_row = ((2 * hidden_size + 64 + 511) // 512) * 512
        combine_row = ((2 * hidden_size + 511) // 512) * 512
        row_bytes = 2 * (local_experts * ep_size * dispatch_row + (topk + 1) * combine_row)
        capacity = min(token_limit, buffer_mb * 1024 * 1024 // row_bytes)
        if not 1024 <= hidden_size <= 8192 or topk > 16:
            capacity = 0
        return cls(capacity, mode)

    def select(self, source_rows: int, *, graph: bool = False) -> str:
        if source_rows <= 0:
            raise ValueError("EPLv2 source rows must include at least one padded row")
        eligible = source_rows <= self.mc2_capacity
        if self.mode == "mc2" and not eligible:
            raise ValueError(f"EPLv2 MC2 source rows {source_rows} exceed capacity {self.mc2_capacity}")
        backend = "mc2" if self.mode != "alltoall" and eligible else "alltoall"
        if graph and backend != "mc2":
            raise ValueError("EPLv2 All-to-AllV requires eager execution; Graph rows must fit MC2 capacity")
        return backend


@dataclass(frozen=True, slots=True)
class TokenParallelLayout:
    num_tokens: int
    tp_size: int
    tp_rank: int
    routed_backend: str | None = None
    dispatch_rows: int | None = None

    def __post_init__(self) -> None:
        if self.num_tokens < 0 or self.tp_size <= 0 or not 0 <= self.tp_rank < self.tp_size:
            raise ValueError("Invalid token-parallel layout")
        if self.dispatch_rows is not None and self.dispatch_rows < self.shard_tokens:
            raise ValueError("EP dispatch capacity must cover the local SP rows")

    @classmethod
    def from_dp(
        cls,
        num_tokens: int,
        tp_size: int,
        tp_rank: int,
        dp_size: int,
        dp_rank: int,
        execution_counts: Sequence[int] | None,
    ) -> TokenParallelLayout:
        """Separate TP-local ownership from the EP-wide dispatch capacity."""
        if dp_size <= 0 or not 0 <= dp_rank < dp_size:
            raise ValueError("Invalid data-parallel layout")
        if dp_size == 1:
            return cls(num_tokens, tp_size, tp_rank)
        if isinstance(execution_counts, torch.Tensor):
            raise TypeError("DP execution counts must be host metadata, not a device tensor")
        if execution_counts is None or len(execution_counts) != dp_size:
            raise ValueError("DP EPLv2 requires execution counts for every DP rank")
        counts = tuple(execution_counts)
        if any(not isinstance(count, int) or count <= 0 for count in counts):
            raise ValueError("DP execution counts must include positive materialized dummy rows")
        if counts[dp_rank] != num_tokens:
            raise ValueError("DP execution count does not match local token rows")
        local = cls(num_tokens, tp_size, tp_rank)
        return cls(
            num_tokens, tp_size, tp_rank, dispatch_rows=max(local.shard_tokens, (max(counts) + tp_size - 1) // tp_size)
        )

    @property
    def shard_tokens(self) -> int:
        return max(1, (self.num_tokens + self.tp_size - 1) // self.tp_size)

    @property
    def valid_tokens(self) -> int:
        """Structural rows owned by this rank, before the runtime activity mask."""
        return max(0, min(self.shard_tokens, self.num_tokens - self.tp_rank * self.shard_tokens))

    @property
    def dispatch_tokens(self) -> int:
        return self.shard_tokens if self.dispatch_rows is None else self.dispatch_rows

    def pad_dispatch(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.ndim == 0 or tensor.shape[0] != self.shard_tokens:
            raise ValueError("Dispatch input must contain this rank's SP rows")
        return self._pad(tensor, self.dispatch_tokens)

    def _pad(self, tensor: torch.Tensor, rows: int) -> torch.Tensor:
        if tensor.shape[0] == rows:
            return tensor.contiguous()
        padded = tensor.new_empty((rows, *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
        padded[tensor.shape[0] :].zero_()
        return padded

    def shard(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0 or tensor.shape[0] != self.num_tokens:
            raise ValueError("Token tensor does not match the full layout")
        start = min(self.tp_rank * self.shard_tokens, self.num_tokens)
        count = min(self.shard_tokens, self.num_tokens - start)
        local = tensor.narrow(0, start, count)
        return self._pad(local, self.shard_tokens)

    def reduce_scatter(self, partial: torch.Tensor, group_name: str = "tp") -> torch.Tensor:
        """Sum full-hidden TP/EP partials and return only this rank's token rows."""
        if partial.dim() == 0 or partial.shape[0] != self.num_tokens:
            raise ValueError("Partial tensor does not match the full token layout")
        padded = self._pad(partial, self.tp_size * self.shard_tokens)
        if self.tp_size == 1:
            return padded
        return distributed.reduce_scatter(padded, self.tp_size, group_name)

    def gather(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 0 or tensor.shape[0] != self.shard_tokens:
            raise ValueError("Token tensor does not match the local layout")
        if self.tp_size == 1:
            return tensor[: self.num_tokens]
        gathered = distributed.tp_all_gather(tensor.contiguous(), 0, self.tp_size)
        return gathered[: self.num_tokens]
