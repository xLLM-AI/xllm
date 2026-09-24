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

import tilelang.language as T
from tilelang import tvm

from .utils import detect_vec_core_num

DEFAULT_TASK_COUNT = detect_vec_core_num()
VEC_NUM = 2
TILE_IDS = 64
WINDOW_IDS = TILE_IDS
GREEDY_PREFIX_VERIFY_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_greedy_prefix_verify_kernel(
    task_count: int = DEFAULT_TASK_COUNT,
    draft_bits: int = 64,
    target_bits: int = 64,
    bonus_bits: int = 64,
) -> tvm.tir.PrimFunc:
    """Build with INT32 indices; callers must validate metadata and storage spans."""
    if task_count <= 0 or task_count > (1 << 31) - 1 or task_count % VEC_NUM != 0:
        raise ValueError(f"task_count must fit positive int32 and be divisible by {VEC_NUM}, got {task_count}")
    if any(bits not in (32, 64) for bits in (draft_bits, target_bits, bonus_bits)):
        raise ValueError("draft_bits, target_bits and bonus_bits must each be 32 or 64")
    draft_dtype = f"int{draft_bits}"
    target_dtype = f"int{target_bits}"
    bonus_dtype = f"int{bonus_bits}"

    @T.prim_func
    def greedy_prefix_verify(
        draft_handle: T.handle,
        target_handle: T.handle,
        bonus_handle: T.handle,
        full_handle: T.handle,
        masked_handle: T.handle,
        batch_size: T.int32,
        draft_width: T.int32,
        draft_stride0: T.int32,
        draft_stride1: T.int32,
        target_stride0: T.int32,
        target_stride1: T.int32,
        bonus_stride0: T.int32,
        mask_enabled: T.int32,
    ):
        # Exact shape expressions use the explicit INT32 metadata, avoiding
        # inferred shape variables that the Ascend ABI declares as INT64.
        draft = T.match_buffer(
            draft_handle,
            (
                T.Select(
                    (batch_size > 0) & (draft_width > 0),
                    T.max(batch_size - 1, 0) * draft_stride0 + T.max(draft_width - 1, 0) * draft_stride1 + 1,
                    0,
                ),
            ),
            draft_dtype,
        )
        target = T.match_buffer(
            target_handle,
            (
                T.Select(
                    (batch_size > 0) & (draft_width > 0),
                    T.max(batch_size - 1, 0) * target_stride0 + T.max(draft_width - 1, 0) * target_stride1 + 1,
                    0,
                ),
            ),
            target_dtype,
        )
        bonus = T.match_buffer(
            bonus_handle,
            (T.Select(batch_size > 0, T.max(batch_size - 1, 0) * bonus_stride0 + 1, 0),),
            bonus_dtype,
        )
        full = T.match_buffer(full_handle, (batch_size * (draft_width + 1),), "int32")
        masked = T.match_buffer(masked_handle, (batch_size * (draft_width + 1),), "int32")
        with T.Kernel(task_count // VEC_NUM, is_npu=True) as (cid, vid), T.Scope("V"):
            task_id = T.alloc_var("int32")
            task_id = cid * VEC_NUM + vid
            rows_per_task = batch_size // task_count
            extra_rows = batch_size % task_count
            row_start = task_id * rows_per_task + T.min(task_id, extra_rows)
            valid_rows = rows_per_task + T.if_then_else(task_id < extra_rows, 1, 0)
            ids_per_tile = T.alloc_var("int32")
            target_capacity = (WINDOW_IDS - 1) // T.max(target_stride1, T.int32(1)) + 1
            ids_per_tile = T.min(T.int32(TILE_IDS), target_capacity)
            if mask_enabled != 0:
                draft_capacity = (WINDOW_IDS - 1) // T.max(draft_stride1, T.int32(1)) + 1
                ids_per_tile = T.min(ids_per_tile, draft_capacity)
            target_native_ub = T.alloc_ub((WINDOW_IDS,), target_dtype)
            draft_native_ub = T.alloc_ub((WINDOW_IDS,), draft_dtype)
            window_i32_ub = T.alloc_ub((WINDOW_IDS,), "int32")
            full_ub = T.alloc_ub((TILE_IDS,), "int32")
            masked_ub = T.alloc_ub((TILE_IDS,), "int32")
            draft_i32_ub = T.alloc_ub((TILE_IDS,), "int32")
            offsets_ub = T.alloc_ub((TILE_IDS,), "uint32")
            rejected_ub = T.alloc_ub((TILE_IDS,), "int32")
            full_u16_ub = T.alloc_ub((TILE_IDS * 2,), "uint16")
            rejected_u16_ub = T.alloc_ub((TILE_IDS * 2,), "uint16")
            masked_u16_ub = T.alloc_ub((TILE_IDS * 2,), "uint16")
            equal_bits_ub = T.alloc_ub((TILE_IDS // 8,), "uint8")
            bonus_native_ub = T.alloc_ub((8,), bonus_dtype)
            bonus_i32_ub = T.alloc_ub((8,), "int32")
            rejected = T.alloc_var("int32")

            for row_local in T.serial(valid_rows):
                row = row_start + row_local
                rejected = 0
                for tile in T.serial(draft_width // ids_per_tile + 1):
                    col_start = tile * ids_per_tile
                    valid_count = T.min(ids_per_tile, draft_width + 1 - col_start)
                    target_count = T.min(
                        T.int32(TILE_IDS), T.max(T.int32(0), T.min(valid_count, draft_width - col_start))
                    )
                    T.tile.fill(full_ub, 0)
                    T.tile.fill(rejected_ub, 0)
                    T.set_flag("v", "mte2", 0)
                    T.wait_flag("v", "mte2", 0)

                    if target_count > 0:
                        target_offset = row * target_stride0 + col_start * target_stride1
                        target_span = (target_count - 1) * target_stride1 + 1
                        if target_bits == 64:
                            T.copy(
                                target[target_offset : target_offset + target_span],
                                target_native_ub,
                            )
                        else:
                            T.copy(
                                target[target_offset : target_offset + target_span],
                                window_i32_ub,
                            )
                        for index in T.serial(TILE_IDS):
                            if index < target_count:
                                offsets_ub[index] = T.Cast("uint32", index * target_stride1 * 4)
                            else:
                                offsets_ub[index] = T.uint32(0)
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        T.set_flag("s", "v", 0)
                        T.wait_flag("s", "v", 0)
                        if target_bits == 64:
                            T.tile.cast(window_i32_ub, target_native_ub, "CAST_NONE", target_span)
                            T.pipe_barrier("v")
                        T.tile.gather(full_ub, window_i32_ub, offsets_ub, 0)

                        if mask_enabled != 0:
                            T.set_flag("v", "mte2", 0)
                            T.wait_flag("v", "mte2", 0)
                            T.set_flag("v", "s", 0)
                            T.wait_flag("v", "s", 0)
                            draft_offset = row * draft_stride0 + col_start * draft_stride1
                            draft_span = (target_count - 1) * draft_stride1 + 1
                            if draft_bits == 64:
                                T.copy(
                                    draft[draft_offset : draft_offset + draft_span],
                                    draft_native_ub,
                                )
                            else:
                                T.copy(
                                    draft[draft_offset : draft_offset + draft_span],
                                    window_i32_ub,
                                )
                            for index in T.serial(TILE_IDS):
                                if index < target_count:
                                    offsets_ub[index] = T.Cast("uint32", index * draft_stride1 * 4)
                                else:
                                    offsets_ub[index] = T.uint32(0)
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.set_flag("s", "v", 0)
                            T.wait_flag("s", "v", 0)
                            if draft_bits == 64:
                                T.tile.cast(window_i32_ub, draft_native_ub, "CAST_NONE", draft_span)
                                T.pipe_barrier("v")
                            T.tile.gather(draft_i32_ub, window_i32_ub, offsets_ub, 0)
                            T.pipe_barrier("v")
                            T.tile.compare(equal_bits_ub, draft_i32_ub, full_ub, "EQ")

                    if target_count < valid_count:
                        T.set_flag("v", "mte2", 0)
                        T.wait_flag("v", "mte2", 0)
                        if bonus_bits == 64:
                            T.copy(bonus[row * bonus_stride0], bonus_native_ub[0:1])
                            T.set_flag("mte2", "v", 0)
                            T.wait_flag("mte2", "v", 0)
                            T.tile.cast(bonus_i32_ub, bonus_native_ub, "CAST_NONE", 1)
                        else:
                            T.copy(bonus[row * bonus_stride0], bonus_i32_ub[0:1])
                            T.set_flag("mte2", "s", 0)
                            T.wait_flag("mte2", "s", 0)
                        T.set_flag("v", "s", 0)
                        T.wait_flag("v", "s", 0)
                        full_ub[target_count] = bonus_i32_ub[0]

                    T.set_flag("v", "s", 0)
                    T.wait_flag("v", "s", 0)
                    if mask_enabled != 0:
                        for index in T.serial(valid_count):
                            # Keep the first rejected target as replacement.
                            rejected_ub[index] = -rejected
                            if index < target_count:
                                equal_bit = (equal_bits_ub[index // 8] >> (index % 8)) & 1
                                if equal_bit == 0:
                                    rejected = 1
                        T.set_flag("s", "v", 0)
                        T.wait_flag("s", "v", 0)
                        # Or operates on 16-bit lanes; preserve all INT32 bits.
                        T.reinterpretcast(full_u16_ub, full_ub, "uint16_t")
                        T.reinterpretcast(rejected_u16_ub, rejected_ub, "uint16_t")
                        T.reinterpretcast(masked_u16_ub, masked_ub, "uint16_t")
                        T.tile.bitwise_or(masked_u16_ub, full_u16_ub, rejected_u16_ub)

                    T.set_flag("s", "mte3", 0)
                    T.wait_flag("s", "mte3", 0)
                    T.set_flag("v", "mte3", 0)
                    T.wait_flag("v", "mte3", 0)
                    output_offset = row * (draft_width + 1) + col_start
                    T.copy(full_ub, full[output_offset : output_offset + valid_count])
                    if mask_enabled != 0:
                        T.copy(masked_ub, masked[output_offset : output_offset + valid_count])
                    T.set_flag("mte3", "v", 0)
                    T.wait_flag("mte3", "v", 0)
                    T.set_flag("mte3", "s", 0)
                    T.wait_flag("mte3", "s", 0)

    return greedy_prefix_verify
