# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Logical-to-physical paged-KV mapping for DCP."""

from __future__ import annotations

import torch


class KVShardLayout:
    """Maps a logical paged-KV coordinate onto one rank's physical cache."""

    INVALID_SLOT = -1

    def __init__(
        self,
        physical_block_size: int,
        dcp_size: int,
        dcp_rank: int,
    ) -> None:
        if physical_block_size <= 0:
            raise ValueError(f"physical_block_size must be positive, got {physical_block_size}")
        if dcp_size <= 0:
            raise ValueError(f"dcp_size must be positive, got {dcp_size}")
        if dcp_rank < 0 or dcp_rank >= dcp_size:
            raise ValueError(
                f"dcp_rank must satisfy 0 <= dcp_rank < dcp_size, got dcp_rank={dcp_rank}, dcp_size={dcp_size}"
            )
        self.physical_block_size = physical_block_size
        self.dcp_size = dcp_size
        self.dcp_rank = dcp_rank

    @property
    def logical_block_size(self) -> int:
        return self.physical_block_size * self.dcp_size

    def local_seq_lens(self, seq_lens: torch.Tensor) -> torch.Tensor:
        logical = self.logical_block_size
        physical = self.physical_block_size
        full_blocks = torch.div(seq_lens, logical, rounding_mode="floor")
        remainder = torch.remainder(seq_lens, logical)
        rank_start = self.dcp_rank * physical
        owned_in_remainder = torch.clamp(remainder - rank_start, 0, physical)
        return full_blocks * physical + owned_in_remainder

    def localize_slots(self, logical_slots: torch.Tensor) -> torch.Tensor:
        valid_slots = logical_slots >= 0
        safe_slots = logical_slots.clamp_min(0)
        logical_offsets = torch.remainder(safe_slots, self.logical_block_size)
        owner_ranks = torch.div(
            logical_offsets,
            self.physical_block_size,
            rounding_mode="floor",
        )
        owned_slots = valid_slots & (owner_ranks == self.dcp_rank)
        logical_block_ids = torch.div(
            safe_slots,
            self.logical_block_size,
            rounding_mode="floor",
        )
        local_offsets = torch.remainder(logical_offsets, self.physical_block_size)
        local_slots = logical_block_ids * self.physical_block_size + local_offsets
        return torch.where(
            owned_slots,
            local_slots,
            torch.full_like(local_slots, self.INVALID_SLOT),
        )

    def expand_indexer_block_table(
        self,
        logical_block_table: torch.Tensor,
    ) -> torch.Tensor:
        if logical_block_table.dim() != 2:
            raise ValueError("indexer block table must be two-dimensional")
        shard_offsets = torch.arange(
            self.dcp_size,
            dtype=logical_block_table.dtype,
            device=logical_block_table.device,
        )
        expanded = logical_block_table.unsqueeze(-1) * self.dcp_size + shard_offsets
        expanded = torch.where(
            logical_block_table.unsqueeze(-1) >= 0,
            expanded,
            torch.full_like(expanded, -1),
        )
        return expanded.flatten(start_dim=1).contiguous()
