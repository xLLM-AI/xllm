# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Gather role-paired state rows into graph-owned metadata on Ascend."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=["batch_size", "total", "capacity", "row_stride"],
    do_not_specialize_on_alignment=["ids", "rows", "out"],
)
def _shared_linear_state_indices_kernel(ids, rows, out, batch_size, total, capacity, row_stride, BLOCK: tl.constexpr):
    for tile in range(tl.program_id(0), tl.cdiv(total, BLOCK), tl.num_programs(0)):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        valid = offsets < total
        safe_offsets = tl.minimum(offsets, total - 1)
        pair = safe_offsets // 2
        logical = tl.load(ids + pair % batch_size)
        row = 2 * (pair // batch_size) + safe_offsets % 2
        active = (logical > 0) & (logical < capacity)
        physical = tl.load(rows + row * row_stride + tl.where(active, logical, 0))
        tl.store(out + offsets, tl.where(active, physical, -1), mask=valid)
