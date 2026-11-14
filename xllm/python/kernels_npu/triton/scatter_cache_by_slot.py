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

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _scatter_cache_by_slot_kernel(
    cache_ptr: tl.tensor,
    slots_ptr: tl.tensor,
    values_ptr: tl.tensor,
    num_slots: int,
    cache_slot_stride: int,
    cache_col_stride: int,
    slot_stride: int,
    value_slot_stride: int,
    value_col_stride: int,
    WIDTH: tl.constexpr,
    BLOCK_WIDTH: tl.constexpr,
) -> None:
    update_idx = tl.program_id(0)
    slot = tl.load(slots_ptr + update_idx * slot_stride).to(tl.int64)
    columns = tl.arange(0, BLOCK_WIDTH)
    values = tl.load(
        values_ptr + update_idx * value_slot_stride + columns * value_col_stride,
        mask=columns < WIDTH,
        other=0,
    )
    tl.store(
        cache_ptr + slot * cache_slot_stride + columns * cache_col_stride,
        values,
        mask=(columns < WIDTH) & (slot >= 0) & (slot < num_slots),
    )


def scatter_cache_by_slot(cache: torch.Tensor, slots: torch.Tensor, values: torch.Tensor) -> None:
    """Write unique cache slots while leaving negative padding slots untouched.

    Valid slots come from the cache allocator and must be unique. The mask is
    read on device, so capture and replay can use different padding patterns.
    """
    slots = slots.reshape(-1)
    if cache.ndim != 2 or values.ndim != 2:
        raise ValueError("cache and values must be matrices")
    if values.shape != (slots.numel(), cache.shape[1]):
        raise ValueError("values must provide one cache entry per slot")
    if values.dtype != cache.dtype:
        raise ValueError("cache and values must have the same dtype")
    if slots.numel() == 0 or cache.shape[1] == 0:
        return
    _scatter_cache_by_slot_kernel[(slots.numel(),)](
        cache,
        slots,
        values,
        cache.shape[0],
        cache.stride(0),
        cache.stride(1),
        slots.stride(0),
        values.stride(0),
        values.stride(1),
        WIDTH=cache.shape[1],
        BLOCK_WIDTH=triton.next_power_of_2(cache.shape[1]),
        num_warps=4,
    )
