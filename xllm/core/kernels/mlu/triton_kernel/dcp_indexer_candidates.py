# Copyright 2026 The xLLM Authors. All Rights Reserved.
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
# ============================================================================
"""MLU Triton postprocessing for DCP-local indexer candidates."""

import triton
import triton.language as tl


@triton.jit
def tmo_dcp_globalize_indexer_candidates_single_row_kernel(
    local_slots_ptr,
    context_lens_ptr,
    global_slots_ptr,
    rows,
    width,
    dcp_size,
    dcp_rank,
    physical_block_size,
    local_stride_row,
    local_stride_col,
    global_stride_row,
    global_stride_col,
    BLOCK_WIDTH: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0)
    column = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)
    in_bounds = (row < rows) & (column < width)
    local_slot = tl.load(local_slots_ptr + row * local_stride_row + column * local_stride_col, mask=in_bounds, other=-1)
    context_len = tl.load(context_lens_ptr + row, mask=row < rows, other=0)
    valid = in_bounds & (column < context_len) & (local_slot >= 0)
    safe_slot = tl.maximum(local_slot, 0)
    global_slot = (
        safe_slot // physical_block_size * dcp_size + dcp_rank
    ) * physical_block_size + safe_slot % physical_block_size
    tl.store(
        global_slots_ptr + row * global_stride_row + column * global_stride_col,
        tl.where(valid, global_slot, -1),
        mask=in_bounds,
    )


@triton.jit
def tmo_dcp_globalize_indexer_candidates_multi_row_kernel(
    local_slots_ptr,
    context_lens_ptr,
    global_slots_ptr,
    rows,
    width,
    dcp_size,
    dcp_rank,
    physical_block_size,
    local_stride_row,
    local_stride_col,
    global_stride_row,
    global_stride_col,
    BLOCK_WIDTH: tl.constexpr,
    ROWS_PER_PROGRAM: tl.constexpr,
):
    row = tl.program_id(0) * ROWS_PER_PROGRAM + tl.arange(0, ROWS_PER_PROGRAM)[:, None]
    column = tl.program_id(1) * BLOCK_WIDTH + tl.arange(0, BLOCK_WIDTH)[None, :]
    in_bounds = (row < rows) & (column < width)
    local_slot = tl.load(local_slots_ptr + row * local_stride_row + column * local_stride_col, mask=in_bounds, other=-1)
    context_len = tl.load(context_lens_ptr + row, mask=row < rows, other=0)
    valid = in_bounds & (column < context_len) & (local_slot >= 0)
    safe_slot = tl.maximum(local_slot, 0)
    global_slot = (
        safe_slot // physical_block_size * dcp_size + dcp_rank
    ) * physical_block_size + safe_slot % physical_block_size
    tl.store(
        global_slots_ptr + row * global_stride_row + column * global_stride_col,
        tl.where(valid, global_slot, -1),
        mask=in_bounds,
    )
