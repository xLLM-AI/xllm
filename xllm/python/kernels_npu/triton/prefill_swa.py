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
"""Gather ring history and append current KV rows for V4.1 paged prefill attention."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_prefill_swa_kernel(
    kv,
    ring,
    table,
    descriptors,
    output,
    KV_ROW: tl.constexpr,
    KV_COL: tl.constexpr,
    TABLE_ROW: tl.constexpr,
    TABLE_COL: tl.constexpr,
    TABLE_WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
    DIM: tl.constexpr,
    ROWS: tl.constexpr,
):
    page = tl.program_id(0)
    desc = descriptors + page * 6
    seq = tl.load(desc)
    row = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    relative = tl.load(desc + 1) + row
    history_start = tl.load(desc + 2)
    past = tl.load(desc + 3)
    input_start = tl.load(desc + 4)
    length = tl.load(desc + 5)
    history = (row < BLOCK) & (relative < past)
    current = (row < BLOCK) & (relative >= past) & (relative < length)
    position = history_start + relative
    column = position // BLOCK % TABLE_WIDTH
    physical = tl.load(table + seq * TABLE_ROW + column * TABLE_COL, history, 0)
    slot = tl.maximum(physical, 0).to(tl.int64) * BLOCK + position % BLOCK
    col = tl.arange(0, DIM)
    old = tl.load(ring + slot[:, None] * DIM + col[None, :], history[:, None], 0)
    source = (input_start + relative - past).to(tl.int64)
    new = tl.load(kv + source[:, None] * KV_ROW + col[None, :] * KV_COL, current[:, None], 0)
    target = page.to(tl.int64) * BLOCK + row
    tl.store(
        output + target[:, None] * DIM + col[None, :], tl.where(history[:, None], old, new), (row < BLOCK)[:, None]
    )


def pack_prefill_swa(
    kv: torch.Tensor, ring: torch.Tensor, table: torch.Tensor, descriptors: torch.Tensor
) -> torch.Tensor:
    pages = descriptors.size(0)
    block, dim = ring.size(1), ring.size(-1)
    packed = ring.new_empty((pages, block, 1, dim))
    rows = 8
    _pack_prefill_swa_kernel[(pages, triton.cdiv(block, rows))](
        kv,
        ring,
        table,
        descriptors,
        packed,
        kv.stride(0),
        kv.stride(-1),
        *table.stride(),
        max(1, table.size(1)),
        block,
        dim,
        rows,
    )
    return packed
