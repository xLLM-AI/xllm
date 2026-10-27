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


import triton
import triton.language as tl


@triton.jit
def tmo_dcp_localize_topk_kernel(
    global_table_ptr: tl.tensor,
    context_lens_ptr: tl.tensor,
    local_table_ptr: tl.tensor,
    local_lens_ptr: tl.tensor,
    dcp_rank: tl.tensor,
    dcp_size: tl.tensor,
    physical_block_size: tl.tensor,
    width: tl.tensor,
    global_stride_rows: tl.tensor,
    global_stride_cols: tl.tensor,
    local_stride_rows: tl.tensor,
    local_stride_cols: tl.tensor,
    BLOCK_N: tl.constexpr,
) -> None:
    LOCAL_BLOCK_N: tl.constexpr = 512 if BLOCK_N == 1024 else BLOCK_N
    ROWS: tl.constexpr = LOCAL_BLOCK_N // 128
    q = tl.num_programs(0)
    pid = tl.program_id(0)
    if (q >= 128) & (width == 2048) & (physical_block_size == 16) & ((dcp_size == 4) | (dcp_size == 8)):
        if BLOCK_N == 1024:
            if pid * ROWS < q:
                b_r = tl.arange(0, ROWS)
                b_rows = pid * ROWS + b_r
                b_valid = b_rows < q
                b_lens = tl.full((ROWS,), 0, tl.int32)
                b_full_lens = tl.full((), True, tl.int1)
                for b_len_index in tl.static_range(0, ROWS):
                    b_len_row = pid * ROWS + b_len_index
                    b_one_length = tl.load(
                        context_lens_ptr + b_len_row,
                        b_len_row < q,
                        other=0,
                        cache_modifier=".ca",
                    )
                    b_lens = tl.where(b_r == b_len_index, b_one_length, b_lens)
                    b_full_lens = b_full_lens & (b_one_length == 2048)
                b_cols = tl.arange(0, 32)[:, None] * 64 + tl.arange(0, 64)[None, :]
                b_source_ptr = (
                    global_table_ptr
                    + b_rows[:, None, None].to(tl.int64) * global_stride_rows
                    + b_cols[None, :, :].to(tl.int64) * global_stride_cols
                )
                if ((pid + 1) * ROWS <= q) & (global_stride_rows == 2048) & (global_stride_cols == 1):
                    b_flat_source = global_table_ptr + pid.to(tl.int64) * (ROWS * 2048) + tl.arange(0, ROWS * 2048)
                    b_slot = tl.load(b_flat_source).reshape((ROWS, 32, 64))
                elif (pid + 1) * ROWS <= q:
                    b_slot = tl.load(b_source_ptr)
                else:
                    b_slot = tl.load(b_source_ptr, b_valid[:, None, None], other=-1)
                b_topology_mask = ((tl.cast(dcp_size, tl.int32) - 1) << 4) | -2147483648
                b_owned = tl.extra.mlu.libdevice.eq(b_slot & b_topology_mask, tl.cast(dcp_rank, tl.int32) << 4)
                if b_full_lens:
                    b_keep = b_owned
                else:
                    b_keep = (b_cols[None, :, :] < b_lens[:, None, None]).to(tl.int8) & b_owned
                b_filter = (tl.arange(0, 64)[:, None] >= tl.arange(0, 64)[None, :]).to(tl.int8).reshape((64, 1, 1, 64))
                b_intra = tl.extra.mlu.conv2d(
                    b_keep.to(tl.int8).reshape((1, 1, ROWS * 32, 64)),
                    b_filter,
                    output_type=tl.float16,
                ).reshape((ROWS, 32, 64))
                b_totals = tl.gather(b_intra, tl.full((ROWS, 32, 1), 63, tl.int32), 2).reshape((ROWS, 32))
                b_group_input = (
                    (tl.arange(0, 32)[:, None] > tl.arange(0, 32)[None, :]).to(tl.int8).reshape((1, 1, 32, 32))
                )
                b_group_filter = b_totals.to(tl.int8).reshape((ROWS, 1, 1, 32))
                b_group_prefix = (
                    tl.extra.mlu.conv2d(b_group_input, b_group_filter, output_type=tl.float16)
                    .reshape((32, ROWS))
                    .trans(1, 0)
                )
                b_offsets = b_group_prefix
                b_prefix = (b_offsets[:, :, None] + b_intra).to(tl.int32)
                b_count = (
                    (
                        tl.gather(b_group_prefix, tl.full((ROWS, 1), 31, tl.int32), 1)
                        + tl.gather(b_totals, tl.full((ROWS, 1), 31, tl.int32), 1)
                    )
                    .reshape((ROWS,))
                    .to(tl.int32)
                )
                if (pid + 1) * ROWS <= q:
                    tl.store(local_lens_ptr + b_rows, b_count)
                else:
                    tl.store(local_lens_ptr + b_rows, b_count, b_valid)
                b_dest = b_r[:, None, None] * 2048 + b_prefix - 1
                if ROWS <= 8:
                    b_dest = b_dest.to(tl.int16)
                b_reordered = tl.extra.mlu.scatter(
                    tl.full((ROWS * 2048,), 0, tl.int32),
                    b_slot.reshape((ROWS * 2048,)),
                    b_dest.reshape((ROWS * 2048,)),
                    b_keep.to(tl.int1).reshape((ROWS * 2048,)),
                )
                b_shift = tl.where(dcp_size == 4, 2, 3)
                b_reordered = (((b_reordered >> b_shift) & -16) | (b_reordered & 15)).reshape((ROWS, 2048))
                b_target_ptr = (
                    local_table_ptr
                    + b_rows[:, None].to(tl.int64) * local_stride_rows
                    + tl.arange(0, 2048)[None, :].to(tl.int64) * local_stride_cols
                )
                if ((pid + 1) * ROWS <= q) & (local_stride_rows == 2048) & (local_stride_cols == 1):
                    b_flat_target = local_table_ptr + pid.to(tl.int64) * (ROWS * 2048) + tl.arange(0, ROWS * 2048)
                    tl.store(b_flat_target, b_reordered.reshape((ROWS * 2048,)))
                elif (pid + 1) * ROWS <= q:
                    tl.store(b_target_ptr, b_reordered)
                else:
                    tl.store(b_target_ptr, b_reordered, b_valid[:, None])
        else:
            b_tasks = tl.cdiv(q, ROWS)
            b_workers = tl.where(q >= 1024, 32, b_tasks)
            if pid < b_workers:
                b_task_span = tl.cdiv(b_tasks, b_workers)
                b_task_begin = pid * b_task_span
                b_task_end = tl.minimum(b_task_begin + b_task_span, b_tasks)
                b_cached_lengths = (q >= 1024) & (q <= 8192) & (ROWS <= 256)
                b_length_buffer = tl.full((256,), 0, tl.int32)
                for b_task in range(b_task_begin, b_task_end):
                    b_r = tl.arange(0, ROWS)
                    b_rows = b_task * ROWS + b_r
                    b_valid = b_rows < q
                    b_lens = tl.load(context_lens_ptr + b_rows, b_valid, other=0)
                    b_full_lens = tl.min(b_lens, 0) == 2048
                    b_cols = tl.arange(0, 32)[:, None] * 64 + tl.arange(0, 64)[None, :]
                    b_source_ptr = (
                        global_table_ptr
                        + b_rows[:, None, None].to(tl.int64) * global_stride_rows
                        + b_cols[None, :, :].to(tl.int64) * global_stride_cols
                    )
                    if ((b_task + 1) * ROWS <= q) & (global_stride_rows == 2048) & (global_stride_cols == 1):
                        b_flat_source = (
                            global_table_ptr + b_task.to(tl.int64) * (ROWS * 2048) + tl.arange(0, ROWS * 2048)
                        )
                        b_slot = tl.load(b_flat_source).reshape((ROWS, 32, 64))
                    elif (b_task + 1) * ROWS <= q:
                        b_slot = tl.load(b_source_ptr)
                    else:
                        b_slot = tl.load(b_source_ptr, b_valid[:, None, None], other=-1)
                    b_topology_mask = ((tl.cast(dcp_size, tl.int32) - 1) << 4) | -2147483648
                    b_owned = ((b_slot ^ (tl.cast(dcp_rank, tl.int32) << 4)) & b_topology_mask) == 0
                    if b_full_lens:
                        b_keep = b_owned
                    else:
                        b_keep = (b_cols[None, :, :] < b_lens[:, None, None]) & b_owned
                    b_filter = (
                        (tl.arange(0, 64)[:, None] >= tl.arange(0, 64)[None, :]).to(tl.int8).reshape((64, 1, 1, 64))
                    )
                    b_intra = tl.extra.mlu.conv2d(
                        b_keep.to(tl.int8).reshape((1, 1, ROWS * 32, 64)),
                        b_filter,
                        output_type=tl.float16,
                    ).reshape((ROWS, 32, 64))
                    b_totals = tl.gather(b_intra, tl.full((ROWS, 32, 1), 63, tl.int32), 2).reshape((ROWS, 32))
                    b_triangle = (tl.arange(0, 32)[:, None] < tl.arange(0, 32)[None, :]).to(tl.int8)
                    b_group_prefix = tl.dot(b_totals.to(tl.int8), b_triangle).to(tl.float16)
                    b_offsets = b_group_prefix
                    b_prefix = (b_offsets[:, :, None] + b_intra).to(tl.int32)
                    b_count = (
                        (
                            tl.gather(b_group_prefix, tl.full((ROWS, 1), 31, tl.int32), 1)
                            + tl.gather(b_totals, tl.full((ROWS, 1), 31, tl.int32), 1)
                        )
                        .reshape((ROWS,))
                        .to(tl.int32)
                    )
                    if b_cached_lengths:
                        b_length_index = (b_task - b_task_begin) * ROWS + b_r
                        b_length_buffer = tl.extra.mlu.scatter(b_length_buffer, b_count, b_length_index, b_valid)
                    else:
                        if (b_task + 1) * ROWS <= q:
                            tl.store(local_lens_ptr + b_rows, b_count)
                        else:
                            tl.store(local_lens_ptr + b_rows, b_count, b_valid)
                    b_dest = b_r[:, None, None] * 2048 + b_prefix - 1
                    if ROWS <= 8:
                        b_dest = b_dest.to(tl.int16)
                    b_reordered = tl.extra.mlu.scatter(
                        tl.full((ROWS * 2048,), 0, tl.int32),
                        b_slot.reshape((ROWS * 2048,)),
                        b_dest.reshape((ROWS * 2048,)),
                        b_keep.to(tl.int1).reshape((ROWS * 2048,)),
                    )
                    if dcp_size == 4:
                        b_reordered = ((b_reordered >> 2) & -16) | (b_reordered & 15)
                    else:
                        b_reordered = ((b_reordered >> 3) & -16) | (b_reordered & 15)
                    b_reordered = b_reordered.reshape((ROWS, 2048))
                    b_target_ptr = (
                        local_table_ptr
                        + b_rows[:, None].to(tl.int64) * local_stride_rows
                        + tl.arange(0, 2048)[None, :].to(tl.int64) * local_stride_cols
                    )
                    if ((b_task + 1) * ROWS <= q) & (local_stride_rows == 2048) & (local_stride_cols == 1):
                        b_flat_target = (
                            local_table_ptr + b_task.to(tl.int64) * (ROWS * 2048) + tl.arange(0, ROWS * 2048)
                        )
                        tl.store(b_flat_target, b_reordered.reshape((ROWS * 2048,)))
                    elif (b_task + 1) * ROWS <= q:
                        tl.store(b_target_ptr, b_reordered)
                    else:
                        tl.store(b_target_ptr, b_reordered, b_valid[:, None])
                if b_cached_lengths:
                    b_length_rows = b_task_begin * ROWS + tl.arange(0, 256)
                    tl.store(
                        local_lens_ptr + b_length_rows,
                        b_length_buffer,
                        (b_length_rows < q) & (b_length_rows < b_task_end * ROWS),
                    )
    else:
        row = tl.program_id(0)
        length = tl.load(context_lens_ptr + row)
        source = global_table_ptr + row.to(tl.int64) * global_stride_rows
        target = local_table_ptr + row.to(tl.int64) * local_stride_rows
        if (width == 2048) & (physical_block_size == 16) & ((dcp_size == 4) | (dcp_size == 8)):
            cols = tl.arange(0, 32)[:, None] * 64 + tl.arange(0, 64)[None, :]
            source_block = tl.make_block_ptr(
                source,
                (2048,),
                (tl.cast(global_stride_cols, tl.int64),),
                (0,),
                (2048,),
                (0,),
            )
            slot = tl.load(source_block).reshape((32, 64))
            topology_mask = ((tl.cast(dcp_size, tl.int32) - 1) << 4) | -2147483648
            owned = (slot & topology_mask) == (tl.cast(dcp_rank, tl.int32) << 4)
            keep = (cols < length) & owned
            flags = keep.to(tl.int32)
            triangle = (tl.arange(0, 64)[:, None] >= tl.arange(0, 64)[None, :]).to(tl.int8).reshape((64, 1, 1, 64))
            intra = (
                tl.extra.mlu.conv2d(
                    flags.to(tl.int8).reshape((1, 1, 32, 64)),
                    triangle,
                    output_type=tl.float32,
                )
                .reshape((32, 64))
                .to(tl.int32)
            )
            totals = tl.gather(intra, tl.full((32, 1), 63, tl.int32), 1).reshape((32,))
            group_id = tl.arange(0, 32)
            group_prefix = totals
            for depth in tl.static_range(0, 5):
                distance = 1 << depth
                partner = tl.maximum(group_id - distance, 0)
                increment = tl.gather(group_prefix, partner, 0)
                group_prefix = group_prefix + tl.where(group_id >= distance, increment, 0)
            offsets = group_prefix - totals
            prefix = offsets[:, None] + intra
            count = tl.sum(totals, 0)
            reordered = tl.extra.mlu.scatter(
                tl.full((2048,), 0, tl.int32),
                slot.reshape((2048,)),
                (prefix - 1).reshape((2048,)),
                keep.reshape((2048,)),
            )
            shift = tl.where(dcp_size == 4, 2, 3)
            reordered = ((reordered >> shift) & -16) | (reordered & 15)
            target_block = tl.make_block_ptr(
                target,
                (2048,),
                (tl.cast(local_stride_cols, tl.int64),),
                (0,),
                (2048,),
                (0,),
            )
            tl.store(target_block, reordered)
        else:
            count = tl.full((), 0, tl.int32)
            divisor_d = tl.minimum(tl.cast(dcp_size, tl.uint64), tl.full((), 2147483648, tl.uint64)).to(tl.int64)
            divisor_i = tl.minimum(tl.cast(physical_block_size, tl.uint64), tl.full((), 2147483648, tl.uint64)).to(
                tl.int64
            )
            owner_rank = tl.cast(dcp_rank, tl.uint64)
            for base in range(0, width, LOCAL_BLOCK_N):
                cols = base + tl.arange(0, LOCAL_BLOCK_N)
                slot = tl.load(
                    source + cols.to(tl.int64) * global_stride_cols,
                    cols < width,
                    other=-1,
                )
                safe = tl.maximum(slot, 0).to(tl.int64)
                block = safe // divisor_i
                owner = block % divisor_d
                local = (block // divisor_d) * divisor_i + safe % divisor_i
                keep = (cols < width) & (cols < length) & (slot >= 0) & (owner.to(tl.uint64) == owner_rank)
                flags = keep.to(tl.int32)
                prefix = tl.cumsum(flags, 0)
                tile_count = tl.sum(flags, 0)
                lane = tl.arange(0, LOCAL_BLOCK_N)
                dest = tl.where(keep, prefix - 1, tile_count + lane - prefix)
                payload = tl.where(keep, local.to(tl.int32), 0)
                reordered = tl.scatter(tl.full((LOCAL_BLOCK_N,), 0, tl.int32), payload, dest, axis=0)
                tl.store(
                    target + (tl.cast(count, tl.int64) + lane.to(tl.int64)) * local_stride_cols,
                    reordered,
                    lane < tile_count,
                )
                count += tile_count
            for base in range(count, width, LOCAL_BLOCK_N):
                cols = base + tl.arange(0, LOCAL_BLOCK_N)
                tl.store(target + cols.to(tl.int64) * local_stride_cols, 0, cols < width)
        tl.store(local_lens_ptr + row, count)
