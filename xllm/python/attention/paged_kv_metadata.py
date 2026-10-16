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

"""Vectorized paged-KV metadata construction for Python attention graphs."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def is_row_aligned_paged_kv(
    paged_kv_indptr: torch.Tensor | None,
    paged_kv_last_page_len: torch.Tensor | None,
    row_count: int,
) -> bool:
    """Whether the paged KV tensors already describe one row per sequence."""
    return (
        paged_kv_indptr is not None
        and paged_kv_last_page_len is not None
        and paged_kv_indptr.dim() == 1
        and paged_kv_indptr.numel() == row_count + 1
        and paged_kv_last_page_len.dim() == 1
        and paged_kv_last_page_len.numel() == row_count
    )


def build_row_aligned_paged_kv_metadata(
    block_table: torch.Tensor,
    kv_seq_lens: Sequence[int] | torch.Tensor | None,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build row-aligned page indices, indptr, and last-page lengths.

    ``block_table`` and ``kv_seq_lens`` describe the same metadata row layout.
    Length zero still owns one page with a last-page length of one, matching
    the framework builders used by decode and speculative inputs.
    """
    if kv_seq_lens is None:
        raise RuntimeError("row-aligned graph paging requires one host KV length per token row")
    # Both callers pass scheduler-owned host lengths; a device tensor here
    # would silently force a sync on this device-side capacity check.
    if isinstance(kv_seq_lens, torch.Tensor) and kv_seq_lens.device.type != "cpu":
        raise RuntimeError("row-aligned graph paging requires host-resident KV lengths")
    if page_size <= 0:
        raise RuntimeError("paged KV metadata page size must be positive")
    # Validate capacity on the host before uploading: the input is already
    # host-resident, so a device-side check would just force a D2H sync.
    host_lengths = kv_seq_lens.tolist() if isinstance(kv_seq_lens, torch.Tensor) else list(kv_seq_lens)
    if any(-(-max(length, 1) // page_size) > block_table.shape[1] for length in host_lengths):
        raise RuntimeError("paged KV metadata exceeds block table capacity")
    kv_seq_lens = torch.as_tensor(kv_seq_lens, dtype=torch.int64, device=block_table.device)
    if block_table.dim() != 2 or kv_seq_lens.dim() != 1:
        raise RuntimeError("paged KV metadata requires a rank-2 block table and rank-1 KV lengths")
    if block_table.shape[0] != kv_seq_lens.numel():
        raise RuntimeError("paged KV metadata row count does not match KV lengths")

    effective_kv_seq_lens = torch.clamp(kv_seq_lens, min=1)
    page_counts = torch.div(
        effective_kv_seq_lens + page_size - 1,
        page_size,
        rounding_mode="floor",
    ).to(torch.int32)
    paged_kv_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.int32, device=block_table.device),
            torch.cumsum(page_counts, dim=0, dtype=torch.int32),
        )
    )
    page_offsets = torch.arange(block_table.shape[1], dtype=torch.int32, device=block_table.device)
    valid_pages = page_offsets.unsqueeze(0) < page_counts.unsqueeze(1)
    paged_kv_indices = block_table.to(torch.int32).masked_select(valid_pages).contiguous()
    paged_kv_last_page_len = ((effective_kv_seq_lens - 1) % page_size + 1).to(torch.int32)
    return paged_kv_indptr.contiguous(), paged_kv_indices, paged_kv_last_page_len.contiguous()
