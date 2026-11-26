# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Prepare private sliding-window KV pages for V4.1 block draft attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _draft_swa_metadata_kernel(
    seq_lens,
    table,
    slots,
    positions,
    lengths,
    sequence_lengths,
    private_pages,
    TABLE_ROWS: tl.constexpr,
    TABLE_COLS: tl.constexpr,
    COLUMNS: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    BLOCK: tl.constexpr,
    PAGES: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
):
    owner = tl.program_id(0)
    query = tl.program_id(1)
    lane = tl.arange(0, TILE)
    span: tl.constexpr = WINDOW + WIDTH
    length = tl.load(seq_lens + owner).to(tl.int64)
    prefix = length - WIDTH
    if query == 0:
        history = prefix - WINDOW + lane
        index = tl.maximum(history, 0)
        column = (index // BLOCK) % COLUMNS
        block = tl.load(table + owner * TABLE_ROWS + column * TABLE_COLS, lane < WINDOW, 0)
        slot = tl.maximum(block.to(tl.int64), 0) * BLOCK + index % BLOCK
        tl.store(slots + owner * WINDOW + lane, tl.where(history >= 0, slot, -1), lane < WINDOW)
        tl.store(private_pages + owner * PAGES + lane, owner * PAGES + lane, lane < PAGES)
        tl.store(sequence_lengths + owner, tl.where(length > 0, span, 0))
    start = tl.minimum(tl.maximum(WINDOW - prefix, 0), span)
    pos = start + lane
    row = owner * WIDTH + query
    tl.store(positions + row * TOPK + lane, tl.where((length > 0) & (pos < span), pos, -1), lane < TOPK)
    tl.store(lengths + row, span - start)


def draft_swa_metadata(
    seq_lens: torch.Tensor, table: torch.Tensor, width: int, window: int, block: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = seq_lens.numel()
    pages = triton.cdiv(window + width, block)
    topk = triton.cdiv(window + width, 128) * 128
    slots = seq_lens.new_empty((batch, window), dtype=torch.int64)
    positions = seq_lens.new_empty((batch * width, 1, topk), dtype=torch.int32)
    lengths = positions.new_empty((batch * width, 1))
    seq_lengths = positions.new_empty((batch,))
    private_pages = positions.new_empty((batch, pages))
    _draft_swa_metadata_kernel[(batch, width)](
        seq_lens,
        table,
        slots,
        positions,
        lengths,
        seq_lengths,
        private_pages,
        *table.stride(),
        table.size(1),
        width,
        window,
        block,
        pages,
        topk,
        triton.next_power_of_2(topk),
    )
    return slots, positions, lengths, seq_lengths, private_pages


@triton.jit
def _draft_swa_pack_kernel(
    ring,
    kv,
    slots,
    cache,
    KV_BATCH: tl.constexpr,
    KV_ROW: tl.constexpr,
    KV_COL: tl.constexpr,
    WINDOW: tl.constexpr,
    WIDTH: tl.constexpr,
    CAPACITY: tl.constexpr,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    owner = tl.program_id(0)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    col = tl.arange(0, DIM)
    slot = tl.load(slots + owner * WINDOW + row, row < WINDOW, -1)
    past = tl.load(ring + tl.maximum(slot, 0)[:, None] * DIM + col[None, :], (slot >= 0)[:, None], 0)
    current = tl.load(
        kv + owner * KV_BATCH + (row[:, None] - WINDOW) * KV_ROW + col[None, :] * KV_COL,
        ((row >= WINDOW) & (row < WINDOW + WIDTH))[:, None],
        0,
    )
    value = tl.where((row < WINDOW)[:, None], past, current)
    tl.store(cache + (owner * CAPACITY + row[:, None]) * DIM + col[None, :], value, (row < WINDOW + WIDTH)[:, None])


def pack_draft_swa(ring: torch.Tensor, kv: torch.Tensor, slots: torch.Tensor, cache: torch.Tensor) -> None:
    batch, window = slots.shape
    current = kv.view(batch, -1, cache.size(-1))
    width = current.size(1)
    _draft_swa_pack_kernel[(batch, triton.cdiv(window + width, 8))](
        ring.view(-1, cache.size(-1)),
        current,
        slots,
        cache,
        *current.stride(),
        window,
        width,
        cache.size(1),
        cache.size(-1),
        8,
    )
