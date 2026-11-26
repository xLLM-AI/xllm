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
"""Copy the two feature ranges of a packed projection into contiguous row-major tensors."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_rows"])
def _split_packed_projection_kernel(
    packed,
    query,
    kv,
    num_rows,
    QUERY_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
):
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    cols = tl.program_id(1) * COLS + tl.arange(0, COLS)
    active = rows[:, None] < num_rows
    width: tl.constexpr = QUERY_WIDTH + KV_WIDTH
    offsets = rows[:, None].to(tl.int64)
    values = tl.load(packed + offsets * width + cols[None, :], active & (cols[None, :] < width), 0)
    tl.store(query + offsets * QUERY_WIDTH + cols[None, :], values, active & (cols[None, :] < QUERY_WIDTH))
    tl.store(
        kv + offsets * KV_WIDTH + (cols[None, :] - QUERY_WIDTH),
        values,
        active & (cols[None, :] >= QUERY_WIDTH) & (cols[None, :] < width),
    )


def split_packed_projection(packed: torch.Tensor, query_width: int) -> tuple[torch.Tensor, torch.Tensor]:
    num_rows, width = packed.shape
    kv_width = width - query_width
    if num_rows <= 1:
        return packed[:, :query_width], packed[:, query_width:]
    query = packed.new_empty((num_rows, query_width))
    kv = packed.new_empty((num_rows, kv_width))
    cols = min(triton.next_power_of_2(width), 2048)
    grid = (triton.cdiv(num_rows, 8), triton.cdiv(width, cols))
    _split_packed_projection_kernel[grid](packed, query, kv, num_rows, query_width, kv_width, 8, cols)
    return query, kv
