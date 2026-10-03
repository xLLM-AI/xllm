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

"""Contiguous per-sequence CP queries with replicated global KV writes for DeepSeek-V4."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from xllm.python import distributed
from xllm.python.layers.rotary_embedding import _expand_half_rope_cos_sin


@dataclass
class DeepseekV4CpContext:
    """Contiguous query segments retaining global position IDs."""

    cp_size: int = 1
    cp_rank: int = 0

    # Global row indices this rank owns, ascending in global order.
    local_row_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    # Maps global row -> its position in rank-major gathered output.
    restore_indices: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    tokens_per_rank: list[int] = field(default_factory=list)
    local_positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))
    global_positions: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int64))

    local_q_seq_lens: list[int] = field(default_factory=list)
    local_kv_seq_lens: list[int] = field(default_factory=list)
    local_kv_cu_seq_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    global_q_cu_seq_lens: torch.Tensor = field(default_factory=lambda: torch.empty(0, dtype=torch.int32))
    # Full caches support lazy lookup; built request-shaped pairs take precedence.
    global_rope_caches: dict[int, torch.Tensor] = field(default_factory=dict)
    # Full-width interleaved pairs for KV rotary kernels; compressor tables remain half-width.
    global_rope_by_ratio: dict[int, tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)

    def enabled(self) -> bool:
        """Use replicated execution on all peers if any CP shard would be empty."""
        return self.cp_size > 1 and min(self.tokens_per_rank, default=0) > 0

    def set_global_rope_cache(self, ratio: int, cos_sin_cache: torch.Tensor) -> None:
        """Register a full-position RoPE cache for one compression ratio."""
        self.global_rope_caches[int(ratio)] = cos_sin_cache

    def global_rope(self, ratio: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return full-width KV RoPE rows, falling back to compression ratio 1."""
        for candidate in (int(ratio), 1):
            built = self.global_rope_by_ratio.get(candidate)
            if built is not None:
                return built
            cache = self.global_rope_caches.get(candidate)
            if cache is not None:
                break
        if cache is None or self.global_positions is None or self.global_positions.numel() == 0:
            return None, None
        cos_sin = cache.index_select(0, self.global_positions.long())
        return _expand_half_rope_cos_sin(cos_sin)

    def set_global_rope_pair(self, ratio: int, cos_sin_pair: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Expand half-width compressor tables without mutating their storage."""
        if cos_sin_pair is None or cos_sin_pair[0] is None or cos_sin_pair[1] is None:
            return
        cos, sin = cos_sin_pair
        if sin.size(-1) != cos.size(-1):
            raise RuntimeError(f"DeepSeek-V4 CP RoPE cos/sin widths differ: cos={cos.shape}, sin={sin.shape}")
        full_cos, full_sin = _expand_half_rope_cos_sin(torch.cat((cos, sin), dim=-1))
        self.global_rope_by_ratio[int(ratio)] = (full_cos, full_sin)

    def shard_rows(self, global_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled():
            return global_tensor
        indices = self.local_row_indices.to(device=global_tensor.device, dtype=torch.int64)
        return global_tensor.index_select(0, indices)

    def gather_restore(self, local_tensor: torch.Tensor) -> torch.Tensor:
        if not self.enabled():
            return local_tensor
        if self.cp_rank < 0 or self.cp_rank >= len(self.tokens_per_rank):
            raise RuntimeError("DeepSeek-V4 CP rank is outside the token-count table")
        # Uneven gathers concatenate valid rank-local rows in the order used by restore_indices.
        gathered = distributed.all_gather_variable(
            local_tensor.contiguous(),
            self.tokens_per_rank,
            self.cp_rank,
            "cp",
        )
        restore = self.restore_indices.to(gathered.device)
        return gathered.index_select(0, restore)


def _compute_cp_rows_by_rank(
    cp_size: int,
    global_q_seq_lens: Sequence[int],
) -> list[list[int]]:
    """Return global row indices owned by each CP rank (rank-major)."""
    rows_by_rank: list[list[int]] = [[] for _ in range(cp_size)]
    seq_base = 0
    for q_len in global_q_seq_lens:
        segment = (int(q_len) + cp_size - 1) // cp_size
        for rank in range(cp_size):
            start = min(rank * segment, int(q_len))
            end = min(start + segment, int(q_len))
            rows_by_rank[rank].extend(range(seq_base + start, seq_base + end))
        seq_base += int(q_len)
    return rows_by_rank


def _compute_cp_local_seq_lens(
    cp_size: int,
    cp_rank: int,
    global_q_seq_lens: Sequence[int],
    global_kv_seq_lens: Sequence[int],
) -> tuple[list[int], list[int]]:
    local_q: list[int] = []
    local_kv: list[int] = []
    for q_len, kv_len in zip(global_q_seq_lens, global_kv_seq_lens, strict=True):
        segment = (q_len + cp_size - 1) // cp_size
        start = min(cp_rank * segment, q_len)
        end = min(start + segment, q_len)
        local_q.append(end - start)
        prefix = kv_len - q_len
        local_kv.append(prefix + end)
    return local_q, local_kv


def build_deepseek_v4_cp_context(
    cp_size: int,
    cp_rank: int,
    global_q_seq_lens: Sequence[int],
    global_kv_seq_lens: Sequence[int],
    global_positions: torch.Tensor | None = None,
) -> DeepseekV4CpContext:
    """Build a contiguous DeepSeek-V4 CP plan for one prefill forward."""
    context = DeepseekV4CpContext(cp_size=cp_size, cp_rank=cp_rank)
    if cp_size <= 1:
        return context
    if cp_rank < 0 or cp_rank >= cp_size:
        raise ValueError(f"DeepSeek-V4 CP rank {cp_rank} is outside [0, {cp_size})")
    if len(global_q_seq_lens) != len(global_kv_seq_lens):
        raise ValueError(
            "DeepSeek-V4 CP expects one kv length per q length: "
            f"{len(global_q_seq_lens)} q vs {len(global_kv_seq_lens)} kv"
        )

    global_token_count = sum(global_q_seq_lens)
    rows_by_rank = _compute_cp_rows_by_rank(cp_size, global_q_seq_lens)
    context.tokens_per_rank = [len(rows) for rows in rows_by_rank]
    context.local_row_indices = torch.tensor(rows_by_rank[cp_rank], dtype=torch.int64)

    global_rows = torch.tensor([row for rows in rows_by_rank for row in rows], dtype=torch.int64)
    if global_rows.numel() != global_token_count:
        raise RuntimeError("DeepSeek-V4 CP segments must cover every global row exactly once")
    restore = torch.empty(global_token_count, dtype=torch.int64)
    if global_rows.numel() > 0:
        restore[global_rows] = torch.arange(global_rows.numel(), dtype=torch.int64)
    context.restore_indices = restore

    context.local_q_seq_lens, context.local_kv_seq_lens = _compute_cp_local_seq_lens(
        cp_size, cp_rank, global_q_seq_lens, global_kv_seq_lens
    )

    q_lens = torch.tensor(global_q_seq_lens, dtype=torch.int32)
    context.global_q_cu_seq_lens = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.cumsum(q_lens, dim=0, dtype=torch.int32))
    )
    kv_lens = torch.tensor(context.local_kv_seq_lens, dtype=torch.int32)
    context.local_kv_cu_seq_lens = torch.cat(
        (torch.zeros(1, dtype=torch.int32), torch.cumsum(kv_lens, dim=0, dtype=torch.int32))
    )

    device = global_positions.device if global_positions is not None else torch.device("cpu")
    context.local_row_indices = context.local_row_indices.to(device)

    if global_positions is not None and global_positions.numel() > 0:
        context.local_positions = global_positions.index_select(0, context.local_row_indices)
        context.global_positions = global_positions.contiguous()

    context.restore_indices = context.restore_indices.to(device)
    context.local_positions = context.local_positions.to(device)
    context.global_positions = context.global_positions.to(device)
    context.global_q_cu_seq_lens = context.global_q_cu_seq_lens.to(device)
    context.local_kv_cu_seq_lens = context.local_kv_cu_seq_lens.to(device)
    return context
