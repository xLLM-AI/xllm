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

"""Copy FP8 cache rows while ignoring negative padding slots on device."""

from __future__ import annotations

from typing import TYPE_CHECKING

import tilelang.language as T

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

NUM_ROWS = T.symbolic("num_rows")
CACHE_ROWS = T.symbolic("cache_rows")
SUPPORTED_HEAD_DIMS = (64, 128, 512)


def build_fp8_cache_write_kernel(head_dim: int) -> PrimFunc:
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(f"unsupported FP8 cache head dimension: {head_dim}")

    @T.prim_func
    def fp8_cache_write(
        slots: T.Tensor((NUM_ROWS,), "int64"),
        values: T.Tensor((NUM_ROWS, head_dim), "uint8"),
        cache: T.Tensor((CACHE_ROWS, head_dim), "uint8"),
    ):
        with T.Kernel(24, is_npu=True) as (cid, vid):
            task_id = cid * 2 + vid
            rows_per_task = (NUM_ROWS + 47) // 48
            row_start = task_id * rows_per_task
            row_end = T.min(row_start + rows_per_task, NUM_ROWS)
            with T.Scope("V"):
                row_ub = T.alloc_ub((head_dim,), "uint8")
                for row in T.serial(row_start, row_end):
                    slot = T.Cast("int32", slots[row])
                    if slot >= 0:
                        T.copy(values[row, :], row_ub)
                        T.copy(row_ub, cache[slot, :])

    return fp8_cache_write
