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

"""Window-local page addressing without changing block-draft KV ownership."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.layers.attention import Attention


def block_draft_window_context(layer: Attention) -> int | None:
    """Return the oldest possible key's distance from the block anchor."""
    if layer.fia_sparse_mode == 0 and getattr(layer, "sliding_window", 0) > 0:
        return layer.sliding_window
    if layer.fia_sparse_mode == 4 and 0 <= layer.fia_pre_tokens < 2147483647:
        return layer.fia_pre_tokens
    return None


def block_draft_model_window_context(model: nn.Module) -> int | None:
    """Require a finite left window on every ordinary attention layer."""
    contexts = [block_draft_window_context(layer) for layer in model.modules() if isinstance(layer, Attention)]
    if not contexts or any(context is None for context in contexts):
        return None
    return max(context for context in contexts if context is not None)


def windowed_block_page_count(left_context: int, query_width: int, page_size: int, source_pages: int) -> int:
    """Include the full query block and the first page's alignment overhang."""
    if left_context < 0 or min(query_width, page_size, source_pages) <= 0:
        raise ValueError("windowed block attention requires nonnegative context and positive capacities")
    tokens = left_context + query_width + page_size - 1
    return min((tokens + page_size - 1) // page_size, source_pages)


@dataclass(frozen=True, slots=True)
class WindowedBlockTable:
    """One forward's compact physical pages and their global key positions."""

    block_table: torch.Tensor
    key_positions: torch.Tensor
    capacity: int


def prepare_windowed_block_table(
    source: torch.Tensor,
    kv_lengths: torch.Tensor,
    query_width: int,
    page_size: int,
    left_context: int,
) -> WindowedBlockTable:
    """Select live window pages on Device, with fixed capture-time shapes.

    The source retains global logical page indices. Padding columns point to
    the window's first live page rather than reading unallocated tail ids;
    the existing global-position mask excludes those columns from attention.
    No KV data, rotary positions, or write slots are moved.
    """
    pages = windowed_block_page_count(left_context, query_width, page_size, source.shape[1])
    first_page = (kv_lengths - query_width - left_context).clamp_min(0) // page_size
    columns = first_page.unsqueeze(1) + torch.arange(pages, device=source.device, dtype=first_page.dtype)
    valid = (columns * page_size < kv_lengths.unsqueeze(1)) & (columns < source.shape[1])
    # Masking the result of an invalid gather would be too late. Clamp indices
    # before gathering, then replace every unused page by a known legal id.
    indices = columns.clamp(0, source.shape[1] - 1).to(torch.int64)
    selected = source.gather(1, indices)
    # Rolling SWA management can retire logical page zero. Reuse a live
    # window page for padding instead of depending on that old prefix page.
    table = torch.where(valid, selected, selected[:, :1]).contiguous()
    capacity = pages * page_size
    keys = first_page.view(-1, 1, 1, 1) * page_size
    keys = keys + torch.arange(capacity, device=source.device, dtype=kv_lengths.dtype).view(1, 1, 1, -1)
    return WindowedBlockTable(table, keys, capacity)
