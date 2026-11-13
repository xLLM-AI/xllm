#!/usr/bin/env python3

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
"""Final H3 RainFusion mixed dense/sparse attention for Ascend A3.

This is the frozen production candidate selected after the xattn/xattn2/xattn3
pipeline experiments. It retains xattn2's implicit three-slot ring, L0C slot
rotation, per-block PV staging, and Round-E output-UB fix, then adds the accepted
xattention-style two-slot K preload/refill schedule.

xattn2 = xattn candidate (per-block PV P staging + implicit ring/skew 3-flag
cross-core sync, no free flags) PLUS Edit B, an L0C slot rotation:

- QK: ``side = (lane + pipe_step) % 2`` instead of ``lane % 2``, so ``l0c[P%2]``
  is always written by QK lanes {0,2} (last writer = S2) and ``l0c[(P+1)%2]`` by
  lanes {1,3}.
- PV: the O accumulator slot becomes ``pv_l0c = (qk_lane_count + pipe_step) % 2``
  (the slot of QK lane ``n-2``, or the slot QK never touched when ``n == 1``)
  instead of ``pipe_step % 2``.

Effect: the PV MMAs no longer collide with the last QK S write on the L0C slot,
so they hide under the remaining S writes and the FIX stream keeps flowing at the
S3->O and O->S0 seams. Pure scheduling: L0C is scratch, the S column destination
(``lane*block:(lane+1)*block``), ``init=True`` per lane, FP32 accumulation and
all dtypes are unchanged, so the output is bit-identical to the xattn candidate.

PV P staging: per-pair [2,128,256] -> per-block [2,128,128] ping-pong with a
chunk4-style 5-step load(L)/consume(L-1) pipeline so each PV MMA waits for only
one 128-col P block instead of a 256-col pair. Pure scheduling; math unchanged.

QK K staging: K0/K1 are preloaded into two L1 slots; after their L1-to-L0B
reads release those slots, K2/K3 refill them while earlier MMA/FIX work remains
in flight. This is the accepted xattn3 schedule.

The selector provides ``plan_ids`` and ``plan_count`` directly as INT32. This
module performs K/V packing, mixed attention, and the optional inverse-record
epilogue in one TileLang launch.

TVMScript requires live parameter annotations, so this module intentionally
does not enable postponed annotations.
"""

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_nz_layout, make_zn_layout

BLOCK = 128
HEADS = 4
DIM = 128
NUM_CORES = 24
VEC_NUM = 2
PLAN_ALIGN = 8
STACK_BLOCKS = 4
TASK_SLOTS = 3
PRELOAD_DISTANCE = 2
ROW_TILE = 16
KV_PACK_ROWS = 256
INVERSE_ROWS = 64
NEG_INF = -3.4e38
SUMEXP_FLOOR = 1e-30
UB_LIMIT = 196352
L1_LIMIT = 512 * 1024
REDUCE_TMP_BYTES = 16 * 1024

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
}


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


@tilelang.jit(pass_configs=PASS_CONFIGS)
def h3_rainfusion_attention(
    total_rows: int,
    valid_rows: int,
    num_blocks: int,
    record_count: int,
    records_padded: int,
    dense_prefix_blocks: int,
    sparse_end_block: int,
    cap_total_rows: int = 0,
    cap_num_blocks: int = 0,
    cap_record_count: int = 0,
    cap_records_padded: int = 0,
    heads: int = HEADS,
    dim: int = DIM,
    num_cores: int = NUM_CORES,
    dtype: str = "bfloat16",
    output_order: str = "original",
    task_limit: int = 0,
):
    """Build the complete mixed RainFusion attention and inverse epilogue."""
    if heads != HEADS or dim != DIM or num_cores != NUM_CORES:
        raise ValueError("H3 RainFusion fixes heads=4, dim=128, num_cores=24")
    if dtype not in ("float16", "bfloat16"):
        raise ValueError("dtype must be float16 or bfloat16")
    if output_order not in ("original", "reordered"):
        raise ValueError("output_order must be original or reordered")
    if not 0 < valid_rows <= total_rows:
        raise ValueError("expected 0 < valid_rows <= total_rows")
    if num_blocks != (valid_rows + BLOCK - 1) // BLOCK:
        raise ValueError("num_blocks must equal ceildiv(valid_rows, 128)")
    if not 0 <= dense_prefix_blocks <= sparse_end_block <= num_blocks:
        raise ValueError("invalid dense/sparse query block partition")
    if records_padded <= 0 or record_count <= 0 or record_count > records_padded:
        raise ValueError("invalid inverse record extent")
    if records_padded % (num_cores * VEC_NUM) != 0:
        raise ValueError("records_padded must be divisible by 48 workers")
    if total_rows % KV_PACK_ROWS:
        raise ValueError("total_rows must be divisible by the 256-row K/V pack tile")

    # ---- capacity resolution: 0 means "identical to the logical value", which
    # reproduces the previous exact-key behaviour bit for bit. A capacity strictly
    # larger than the logical geometry lets ONE compiled object serve every geometry
    # that fits, with the logical values passed as runtime scalars.
    cap_total_rows = cap_total_rows or total_rows
    cap_num_blocks = cap_num_blocks or num_blocks
    cap_record_count = cap_record_count or record_count
    cap_records_padded = cap_records_padded or records_padded
    if (
        cap_total_rows < total_rows
        or cap_num_blocks < num_blocks
        or cap_record_count < record_count
        or cap_records_padded < records_padded
    ):
        raise ValueError("capacity must cover the logical geometry")
    if cap_total_rows % KV_PACK_ROWS:
        raise ValueError("cap_total_rows must be divisible by the 256-row K/V pack tile")
    if cap_records_padded % (num_cores * VEC_NUM) != 0:
        raise ValueError("cap_records_padded must be divisible by 48 workers")

    block = BLOCK
    half_m = block // VEC_NUM
    # PHYSICAL extents come from capacity; the LOGICAL values become runtime scalars.
    plan_width = _align_up(cap_num_blocks, PLAN_ALIGN)
    maximum_stacks = (cap_num_blocks + STACK_BLOCKS - 1) // STACK_BLOCKS
    num_tasks_all = cap_num_blocks * heads
    num_tasks = min(num_tasks_all, task_limit) if task_limit > 0 else num_tasks_all
    kv_tail_rows = valid_rows - (num_blocks - 1) * block
    scale = (1.0 / dim) ** 0.5
    accum_dtype = "float32"
    pack_row_tiles = (cap_total_rows + KV_PACK_ROWS - 1) // KV_PACK_ROWS
    pack_tasks = 2 * heads * pack_row_tiles
    pack_workers = num_cores * VEC_NUM
    records_per_worker = cap_records_padded // (num_cores * VEC_NUM)

    # L1: Q + two-slot K refill + stack-4 V + two-slot P.
    l1_q_addr = 0
    l1_k_addr = l1_q_addr + block * dim * 2
    l1_v_addr = l1_k_addr + 2 * block * dim * 2
    l1_p_addr = l1_v_addr + STACK_BLOCKS * block * dim * 2
    assert l1_p_addr + 2 * block * block * 2 < L1_LIMIT

    # UB: the FP32 score relay is consumed in 16x512 row tiles. Buffers used
    # only by packing/inverse reorder deliberately alias this plan (Round-E:
    # output_ub no longer aliases it - see ub_output_addr below).
    ub_work_addr = 0
    ub_p_addr = ub_work_addr + 2 * ROW_TILE * STACK_BLOCKS * block * 4
    ub_running_o_addr = ub_p_addr + ROW_TILE * STACK_BLOCKS * block * 2
    ub_partial_o_addr = ub_running_o_addr + half_m * dim * 4
    ub_brcb_addr = ub_partial_o_addr + half_m * dim * 4
    ub_col_pos_addr = ub_brcb_addr + ROW_TILE * 8 * 4
    ub_mask_addr = ub_col_pos_addr + STACK_BLOCKS * block * 4
    ub_running_max_addr = ub_mask_addr + ROW_TILE * (STACK_BLOCKS * block // 8)
    ub_running_sum_addr = ub_running_max_addr + half_m * 4
    ub_rescale_slots_addr = ub_running_sum_addr + half_m * 4
    ub_tile_max_addr = ub_rescale_slots_addr + TASK_SLOTS * half_m * 4
    ub_new_max_addr = ub_tile_max_addr + ROW_TILE * 4
    ub_rescale_addr = ub_new_max_addr + ROW_TILE * 4
    ub_stack_sum_addr = ub_rescale_addr + ROW_TILE * 4
    ub_reduce_stage1_addr = ub_stack_sum_addr + ROW_TILE * 4
    ub_reduce_stage2_addr = ub_reduce_stage1_addr + ROW_TILE * (STACK_BLOCKS * block // 8) * 4
    ub_tmp_addr = _align_up(
        ub_reduce_stage2_addr + ROW_TILE * (STACK_BLOCKS * block // 64) * 4,
        32,
    )
    assert ub_tmp_addr + REDUCE_TMP_BYTES < UB_LIMIT
    # Round-E fix: output_ub must NOT alias score_work_ub_ping (UB[0, 16KB)).
    # At task boundaries with small active_stacks the AIV reaches the next
    # task's s_ready wait with the credit already set (the AIC's one-lane QK
    # finishes early), so there is no blocking gap: the epilogue cast (V) and
    # output write (MTE3) on output_ub race with the next task's first S read
    # (MTE2 into ping) and the in-place softmax writes (V). Poison + CPU
    # resimulation proved both corruptions (task0 reordered_output = S-tile
    # bytes; task1 rows 0-7 = cast bytes). Give output_ub its own region in
    # the free UB tail; zero added instructions, zero hot-path cost.
    ub_output_addr = _align_up(ub_tmp_addr + REDUCE_TMP_BYTES, 32)
    assert ub_output_addr + half_m * dim * 2 <= UB_LIMIT
    assert 2 * KV_PACK_ROWS * dim * 2 <= UB_LIMIT
    assert 2 * INVERSE_ROWS * heads * dim * 2 <= UB_LIMIT
    ub_inverse_stage_addr = 0
    ub_inverse_records_addr = 2 * INVERSE_ROWS * heads * dim * 2
    assert ub_inverse_records_addr + records_per_worker * 4 * 4 < UB_LIMIT

    # Cross-core ownership IDs.
    s_ready = 0
    p_ready = 2
    o_ready = 4

    @T.prim_func
    def main(
        reordered_q: T.Tensor([cap_total_rows, heads * dim], dtype),  # type: ignore
        reordered_k: T.Tensor([cap_total_rows, heads * dim], dtype),  # type: ignore
        reordered_v: T.Tensor([cap_total_rows, heads * dim], dtype),  # type: ignore
        plan_ids: T.Tensor([cap_num_blocks, heads, plan_width], "int32"),  # type: ignore
        plan_count: T.Tensor([cap_num_blocks, PLAN_ALIGN], "int32"),  # type: ignore
        reorder_records_flat: T.Tensor([cap_records_padded * 4], "int32"),  # type: ignore
        output: T.Tensor([cap_total_rows, heads * dim], dtype),  # type: ignore
        workspace_s: T.Tensor([TASK_SLOTS, num_cores, block, STACK_BLOCKS * block], accum_dtype),  # type: ignore
        workspace_p: T.Tensor([TASK_SLOTS, num_cores, block, STACK_BLOCKS * block], dtype),  # type: ignore
        workspace_o: T.Tensor([TASK_SLOTS, num_cores, block, dim], accum_dtype),  # type: ignore
        packed_k: T.Tensor([heads, cap_total_rows, dim], dtype),  # type: ignore
        packed_v: T.Tensor([heads, cap_total_rows, dim], dtype),  # type: ignore
        reordered_output: T.Tensor([cap_total_rows, heads * dim], dtype),  # type: ignore
        # ---- RUNTIME logical geometry (capacity ABI) ----
        # Capacity sizes every physical buffer, stride and GM extent above; these are the
        # ACTIVE extents the kernel walks. Passing them as arguments is what lets one
        # dispatched object serve many logical geometries.
        rt_valid_rows: T.int32,
        rt_num_blocks: T.int32,
        rt_kv_tail_rows: T.int32,
        rt_dense_prefix_blocks: T.int32,
        rt_sparse_end_block: T.int32,
        rt_record_count: T.int32,
        rt_records_per_worker: T.int32,
        rt_pack_row_tiles: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1([block, dim], dtype)
            k_stack_l1 = T.alloc_L1([2, block, dim], dtype)
            v_stack_l1 = T.alloc_L1([STACK_BLOCKS, block, dim], dtype)
            p_l1 = T.alloc_L1([2, block, block], dtype)
            T.annotate_layout(
                {
                    q_l1: make_zn_layout(q_l1),
                    k_stack_l1: make_nz_layout(k_stack_l1),
                    v_stack_l1: make_zn_layout(v_stack_l1),
                    p_l1: make_zn_layout(p_l1),
                }
            )

            l0a = T.alloc_L0A([2, block, dim], dtype)
            l0b = T.alloc_L0B([2, dim, block], dtype)
            l0c = T.alloc_L0C([2, block, block], accum_dtype)

            score_work_ub_ping = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block], accum_dtype)
            score_work_ub_pong = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block], accum_dtype)
            probability_ub = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block], dtype)
            running_o_ub = T.alloc_ub([half_m, dim], accum_dtype)
            partial_o_ub = T.alloc_ub([half_m, dim], accum_dtype)
            mask_source_ub = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block], accum_dtype)
            brcb_ub = T.alloc_ub([ROW_TILE, 8], accum_dtype)
            col_pos_ub = T.alloc_ub([1, STACK_BLOCKS * block], accum_dtype)
            mask_ub = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block // 8], "uint8")
            output_ub = T.alloc_ub([half_m, dim], dtype)
            running_max = T.alloc_ub([half_m, 1], accum_dtype)
            running_sum = T.alloc_ub([half_m, 1], accum_dtype)
            rescale_slots = T.alloc_ub([TASK_SLOTS, half_m, 1], accum_dtype)
            tile_max = T.alloc_ub([ROW_TILE, 1], accum_dtype)
            new_max = T.alloc_ub([ROW_TILE, 1], accum_dtype)
            rescale = T.alloc_ub([ROW_TILE, 1], accum_dtype)
            stack_sum = T.alloc_ub([ROW_TILE, 1], accum_dtype)
            reduce_stage1 = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block // 8], accum_dtype)
            reduce_stage2 = T.alloc_ub([ROW_TILE, STACK_BLOCKS * block // 64], accum_dtype)
            tmp_ub = T.alloc_ub([REDUCE_TMP_BYTES], "uint8")

            pack_ub = T.alloc_ub([2, KV_PACK_ROWS, dim], dtype)
            records_ub = T.alloc_ub([records_per_worker * 4], "int32")
            inverse_stage_ub = T.alloc_ub([2, INVERSE_ROWS, heads * dim], dtype)

            T.annotate_address(
                {
                    q_l1: l1_q_addr,
                    k_stack_l1: l1_k_addr,
                    v_stack_l1: l1_v_addr,
                    p_l1: l1_p_addr,
                    l0a: 0,
                    l0b: 0,
                    l0c: 0,
                    score_work_ub_ping: ub_work_addr,
                    score_work_ub_pong: ub_work_addr + ROW_TILE * STACK_BLOCKS * block * 4,
                    probability_ub: ub_p_addr,
                    running_o_ub: ub_running_o_addr,
                    partial_o_ub: ub_partial_o_addr,
                    mask_source_ub: ub_partial_o_addr,
                    brcb_ub: ub_brcb_addr,
                    col_pos_ub: ub_col_pos_addr,
                    mask_ub: ub_mask_addr,
                    output_ub: ub_output_addr,
                    running_max: ub_running_max_addr,
                    running_sum: ub_running_sum_addr,
                    rescale_slots: ub_rescale_slots_addr,
                    tile_max: ub_tile_max_addr,
                    new_max: ub_new_max_addr,
                    rescale: ub_rescale_addr,
                    stack_sum: ub_stack_sum_addr,
                    reduce_stage1: ub_reduce_stage1_addr,
                    reduce_stage2: ub_reduce_stage2_addr,
                    tmp_ub: ub_tmp_addr,
                    pack_ub: ub_work_addr,
                    records_ub: ub_inverse_records_addr,
                    inverse_stage_ub: ub_inverse_stage_addr,
                }
            )

            # The C/V query task loop must be bounded by the LOGICAL block count. With
            # the capacity bound (cap_num_blocks * heads = 1932) a geometry needing only
            # 470 * 4 = 1880 tasks would run 52 extra tasks whose qb exceeds the real
            # block count, reading rows past valid_rows.
            rt_num_tasks = rt_num_blocks * heads
            my_iters = T.if_then_else(
                cid < rt_num_tasks,
                T.ceildiv(rt_num_tasks - cid, num_cores),
                0,
            )

            # Phase 1: TND valid prefix -> head-major caller workspace.
            with T.Scope("V"):
                T.set_flag("MTE3", "MTE2", 0)
                T.set_flag("MTE3", "MTE2", 1)
                pack_worker = cid * VEC_NUM + vid
                # The pack loop's COUNT and its tensor/head/tile decomposition must all
                # use the LOGICAL row-tile count. With the capacity value (242 tiles,
                # 1936 tasks) a geometry needing only ceildiv(60132,256) = 235 tiles would
                # run tile 235 at pack_row 60160 and compute rt_valid_rows - pack_row =
                # -28 rows. Binding count, head and tile together keeps the decomposition
                # self-consistent with the number of tasks actually enumerated.
                rt_pack_tasks = 2 * heads * rt_pack_row_tiles
                pack_iters = T.if_then_else(
                    pack_worker < rt_pack_tasks,
                    T.ceildiv(rt_pack_tasks - pack_worker, pack_workers),
                    0,
                )
                for pack_iter in T.serial(pack_iters):
                    pack_task = pack_worker + pack_iter * pack_workers
                    pack_tensor = pack_task // (heads * rt_pack_row_tiles)
                    pack_local = pack_task % (heads * rt_pack_row_tiles)
                    pack_head = pack_local // rt_pack_row_tiles
                    pack_row_tile = pack_local % rt_pack_row_tiles
                    pack_row = pack_row_tile * KV_PACK_ROWS
                    pack_valid = T.if_then_else(
                        pack_row + KV_PACK_ROWS <= rt_valid_rows,
                        KV_PACK_ROWS,
                        rt_valid_rows - pack_row,
                    )
                    pack_side = pack_iter % 2
                    T.wait_flag("MTE3", "MTE2", pack_side)
                    if pack_tensor == 0:
                        T.copy(
                            reordered_k[
                                pack_row : pack_row + pack_valid,
                                pack_head * dim : (pack_head + 1) * dim,
                            ],
                            pack_ub[pack_side, :, :],
                            pad_value=T.cast(0, dtype),
                        )
                    else:
                        T.copy(
                            reordered_v[
                                pack_row : pack_row + pack_valid,
                                pack_head * dim : (pack_head + 1) * dim,
                            ],
                            pack_ub[pack_side, :, :],
                            pad_value=T.cast(0, dtype),
                        )
                    T.set_flag("MTE2", "MTE3", pack_side)
                    T.wait_flag("MTE2", "MTE3", pack_side)
                    if pack_tensor == 0:
                        T.copy(
                            pack_ub[pack_side, :, :],
                            packed_k[pack_head, pack_row : pack_row + KV_PACK_ROWS, :],
                        )
                    else:
                        T.copy(
                            pack_ub[pack_side, :, :],
                            packed_v[pack_head, pack_row : pack_row + KV_PACK_ROWS, :],
                        )
                    T.set_flag("MTE3", "MTE2", pack_side)
                T.wait_flag("MTE3", "MTE2", 0)
                T.wait_flag("MTE3", "MTE2", 1)

            T.sync_all()

            # Phase 2C: QK and PV on AIC with stack-4, three-slot ownership.
            with T.Scope("C"):
                sig_k_l1 = 0
                sig_v_l1 = 4
                sig_p_l1 = 2
                sig_q_l1 = 6
                sig_l0ab = 0
                sig_l0c = 0
                for event in T.serial(5):
                    T.set_flag("MTE1", "MTE2", event)
                T.set_flag("MTE1", "MTE2", sig_q_l1)
                T.set_flag("M", "MTE1", sig_l0ab)
                T.set_flag("M", "MTE1", sig_l0ab + 1)
                T.set_flag("FIX", "M", sig_l0c)
                T.set_flag("FIX", "M", sig_l0c + 1)

                for core_index in T.serial(my_iters):
                    task = cid + core_index * num_cores
                    qb = task // heads
                    head = task % heads
                    q_row = qb * block
                    q_valid = T.if_then_else(
                        q_row + block <= rt_valid_rows,
                        block,
                        rt_valid_rows - q_row,
                    )
                    sparse_task = qb >= rt_dense_prefix_blocks and qb < rt_sparse_end_block
                    active_count = T.if_then_else(
                        sparse_task,
                        plan_count[qb, head],
                        rt_num_blocks,
                    )
                    active_stacks = T.ceildiv(active_count, STACK_BLOCKS)

                    T.wait_flag("MTE1", "MTE2", sig_q_l1)
                    T.copy(
                        reordered_q[
                            q_row : q_row + q_valid,
                            head * dim : (head + 1) * dim,
                        ],
                        q_l1[0:q_valid, :],
                    )
                    T.set_flag("MTE2", "MTE1", sig_q_l1)
                    T.wait_flag("MTE2", "MTE1", sig_q_l1)

                    for pipe_step in T.serial(maximum_stacks + PRELOAD_DISTANCE):
                        qk_n = T.if_then_else(
                            pipe_step < active_stacks,
                            T.min(STACK_BLOCKS, active_count - pipe_step * STACK_BLOCKS),
                            0,
                        )
                        pv_l0c = (qk_n + pipe_step) % 2
                        if pipe_step < active_stacks:
                            qk_stack = pipe_step
                            qk_slot = qk_stack % TASK_SLOTS
                            qk_lane_count = qk_n
                            side0 = pipe_step % 2
                            side1 = (pipe_step + 1) % 2

                            if qk_lane_count > 0:
                                selected_pos0 = qk_stack * STACK_BLOCKS
                                block_id0 = T.if_then_else(
                                    sparse_task,
                                    plan_ids[qb, head, selected_pos0],
                                    selected_pos0,
                                )
                                T.wait_flag("MTE1", "MTE2", sig_k_l1)
                                T.copy(
                                    packed_k[
                                        head,
                                        block_id0 * block : (block_id0 + 1) * block,
                                        :,
                                    ],
                                    k_stack_l1[0, :, :],
                                )
                                T.set_flag("MTE2", "MTE1", sig_k_l1)

                            if qk_lane_count > 1:
                                selected_pos1 = qk_stack * STACK_BLOCKS + 1
                                block_id1 = T.if_then_else(
                                    sparse_task,
                                    plan_ids[qb, head, selected_pos1],
                                    selected_pos1,
                                )
                                T.wait_flag("MTE1", "MTE2", sig_k_l1 + 1)
                                T.copy(
                                    packed_k[
                                        head,
                                        block_id1 * block : (block_id1 + 1) * block,
                                        :,
                                    ],
                                    k_stack_l1[1, :, :],
                                )
                                T.set_flag("MTE2", "MTE1", sig_k_l1 + 1)

                            if qk_lane_count > 0:
                                T.wait_flag("M", "MTE1", sig_l0ab + side0)
                                T.copy(q_l1, l0a[side0, :, :])
                                T.wait_flag("MTE2", "MTE1", sig_k_l1)
                                T.copy(
                                    k_stack_l1[0, :, :],
                                    l0b[side0, :, :],
                                    transpose=True,
                                )
                                T.set_flag("MTE1", "MTE2", sig_k_l1)
                                T.set_flag("MTE1", "M", sig_l0ab + side0)
                                T.wait_flag("MTE1", "M", sig_l0ab + side0)
                                T.wait_flag("FIX", "M", sig_l0c + side0)
                                T.mma(
                                    l0a[side0, :, :],
                                    l0b[side0, :, :],
                                    l0c[side0, :, :],
                                    init=True,
                                )
                                T.set_flag("M", "MTE1", sig_l0ab + side0)
                                T.set_flag("M", "FIX", sig_l0c + side0)

                            if qk_lane_count > 2:
                                selected_pos2 = qk_stack * STACK_BLOCKS + 2
                                block_id2 = T.if_then_else(
                                    sparse_task,
                                    plan_ids[qb, head, selected_pos2],
                                    selected_pos2,
                                )
                                T.wait_flag("MTE1", "MTE2", sig_k_l1)
                                T.copy(
                                    packed_k[
                                        head,
                                        block_id2 * block : (block_id2 + 1) * block,
                                        :,
                                    ],
                                    k_stack_l1[0, :, :],
                                )
                                T.set_flag("MTE2", "MTE1", sig_k_l1)

                            if qk_lane_count > 1:
                                T.wait_flag("M", "MTE1", sig_l0ab + side1)
                                T.copy(q_l1, l0a[side1, :, :])
                                T.wait_flag("MTE2", "MTE1", sig_k_l1 + 1)
                                T.copy(
                                    k_stack_l1[1, :, :],
                                    l0b[side1, :, :],
                                    transpose=True,
                                )
                                T.set_flag("MTE1", "MTE2", sig_k_l1 + 1)
                                T.set_flag("MTE1", "M", sig_l0ab + side1)
                                T.wait_flag("MTE1", "M", sig_l0ab + side1)
                                T.wait_flag("FIX", "M", sig_l0c + side1)
                                T.mma(
                                    l0a[side1, :, :],
                                    l0b[side1, :, :],
                                    l0c[side1, :, :],
                                    init=True,
                                )
                                T.set_flag("M", "MTE1", sig_l0ab + side1)
                                T.set_flag("M", "FIX", sig_l0c + side1)

                            if qk_lane_count > 3:
                                selected_pos3 = qk_stack * STACK_BLOCKS + 3
                                block_id3 = T.if_then_else(
                                    sparse_task,
                                    plan_ids[qb, head, selected_pos3],
                                    selected_pos3,
                                )
                                T.wait_flag("MTE1", "MTE2", sig_k_l1 + 1)
                                T.copy(
                                    packed_k[
                                        head,
                                        block_id3 * block : (block_id3 + 1) * block,
                                        :,
                                    ],
                                    k_stack_l1[1, :, :],
                                )
                                T.set_flag("MTE2", "MTE1", sig_k_l1 + 1)

                            if qk_lane_count > 0:
                                T.wait_flag("M", "FIX", sig_l0c + side0)
                                T.copy(
                                    l0c[side0, :, :],
                                    workspace_s[qk_slot, cid, :, 0:block],
                                )
                                T.set_flag("FIX", "M", sig_l0c + side0)

                            if qk_lane_count > 2:
                                T.wait_flag("M", "MTE1", sig_l0ab + side0)
                                T.wait_flag("MTE2", "MTE1", sig_k_l1)
                                T.copy(
                                    k_stack_l1[0, :, :],
                                    l0b[side0, :, :],
                                    transpose=True,
                                )
                                T.set_flag("MTE1", "MTE2", sig_k_l1)
                                T.set_flag("MTE1", "M", sig_l0ab + side0)
                                T.wait_flag("MTE1", "M", sig_l0ab + side0)
                                T.wait_flag("FIX", "M", sig_l0c + side0)
                                T.mma(
                                    l0a[side0, :, :],
                                    l0b[side0, :, :],
                                    l0c[side0, :, :],
                                    init=True,
                                )
                                T.set_flag("M", "MTE1", sig_l0ab + side0)
                                T.set_flag("M", "FIX", sig_l0c + side0)

                            if qk_lane_count > 1:
                                T.wait_flag("M", "FIX", sig_l0c + side1)
                                T.copy(
                                    l0c[side1, :, :],
                                    workspace_s[qk_slot, cid, :, block : 2 * block],
                                )
                                T.set_flag("FIX", "M", sig_l0c + side1)

                            if qk_lane_count > 3:
                                T.wait_flag("M", "MTE1", sig_l0ab + side1)
                                T.wait_flag("MTE2", "MTE1", sig_k_l1 + 1)
                                T.copy(
                                    k_stack_l1[1, :, :],
                                    l0b[side1, :, :],
                                    transpose=True,
                                )
                                T.set_flag("MTE1", "MTE2", sig_k_l1 + 1)
                                T.set_flag("MTE1", "M", sig_l0ab + side1)
                                T.wait_flag("MTE1", "M", sig_l0ab + side1)
                                T.wait_flag("FIX", "M", sig_l0c + side1)
                                T.mma(
                                    l0a[side1, :, :],
                                    l0b[side1, :, :],
                                    l0c[side1, :, :],
                                    init=True,
                                )
                                T.set_flag("M", "MTE1", sig_l0ab + side1)
                                T.set_flag("M", "FIX", sig_l0c + side1)

                            if qk_lane_count > 2:
                                T.wait_flag("M", "FIX", sig_l0c + side0)
                                T.copy(
                                    l0c[side0, :, :],
                                    workspace_s[qk_slot, cid, :, 2 * block : 3 * block],
                                )
                                T.set_flag("FIX", "M", sig_l0c + side0)

                            if qk_lane_count > 3:
                                T.wait_flag("M", "FIX", sig_l0c + side1)
                                T.copy(
                                    l0c[side1, :, :],
                                    workspace_s[qk_slot, cid, :, 3 * block : 4 * block],
                                )
                                T.set_flag("FIX", "M", sig_l0c + side1)
                            T.set_cross_flag("FIX", s_ready, 2)

                        if pipe_step >= PRELOAD_DISTANCE:
                            pv_stack = pipe_step - PRELOAD_DISTANCE
                            if pv_stack < active_stacks:
                                pv_slot = pv_stack % TASK_SLOTS
                                pv_lane_count = T.min(
                                    STACK_BLOCKS,
                                    active_count - pv_stack * STACK_BLOCKS,
                                )
                                T.wait_flag("MTE1", "MTE2", sig_v_l1)
                                for lane in T.serial(STACK_BLOCKS):
                                    if lane < pv_lane_count:
                                        selected_pos = pv_stack * STACK_BLOCKS + lane
                                        block_id = T.if_then_else(
                                            sparse_task,
                                            plan_ids[qb, head, selected_pos],
                                            selected_pos,
                                        )
                                        T.copy(
                                            packed_v[
                                                head,
                                                block_id * block : (block_id + 1) * block,
                                                :,
                                            ],
                                            v_stack_l1[lane, :, :],
                                        )
                                T.set_flag("MTE2", "MTE1", sig_v_l1)
                                T.wait_cross_flag(p_ready)
                                T.wait_flag("MTE2", "MTE1", sig_v_l1)
                                T.wait_flag("FIX", "M", sig_l0c + pv_l0c)
                                for pv_step in T.serial(STACK_BLOCKS + 1):
                                    if pv_step < pv_lane_count:
                                        load_side = pv_step % 2
                                        T.wait_flag("MTE1", "MTE2", sig_p_l1 + load_side)
                                        T.copy(
                                            workspace_p[
                                                pv_slot,
                                                cid,
                                                :,
                                                pv_step * block : (pv_step + 1) * block,
                                            ],
                                            p_l1[load_side, :, :],
                                        )
                                        T.set_flag("MTE2", "MTE1", sig_p_l1 + load_side)
                                    if pv_step >= 1:
                                        lane = pv_step - 1
                                        if lane < pv_lane_count:
                                            side = lane % 2
                                            T.wait_flag("M", "MTE1", sig_l0ab + side)
                                            T.copy(v_stack_l1[lane, :, :], l0b[side, :, :])
                                            T.wait_flag("MTE2", "MTE1", sig_p_l1 + side)
                                            T.copy(p_l1[side, :, :], l0a[side, :, :])
                                            T.set_flag("MTE1", "MTE2", sig_p_l1 + side)
                                            T.set_flag("MTE1", "M", sig_l0ab + side)
                                            T.wait_flag("MTE1", "M", sig_l0ab + side)
                                            T.mma(
                                                l0a[side, :, :],
                                                l0b[side, :, :],
                                                l0c[pv_l0c, :, :],
                                                init=lane == 0,
                                            )
                                            T.set_flag("M", "MTE1", sig_l0ab + side)
                                T.set_flag("M", "FIX", sig_l0c + pv_l0c)
                                T.wait_flag("M", "FIX", sig_l0c + pv_l0c)
                                T.copy(
                                    l0c[pv_l0c, :, :],
                                    workspace_o[pv_slot, cid, :, :],
                                )
                                T.set_flag("FIX", "M", sig_l0c + pv_l0c)
                                T.set_flag("MTE1", "MTE2", sig_v_l1)
                                T.set_cross_flag("FIX", o_ready, 2)

                    T.set_flag("MTE1", "MTE2", sig_q_l1)

                for event in T.serial(5):
                    T.wait_flag("MTE1", "MTE2", event)
                T.wait_flag("MTE1", "MTE2", sig_q_l1)
                T.wait_flag("M", "MTE1", sig_l0ab)
                T.wait_flag("M", "MTE1", sig_l0ab + 1)
                T.wait_flag("FIX", "M", sig_l0c)
                T.wait_flag("FIX", "M", sig_l0c + 1)

            # Phase 2V: one-pass FP32 stack softmax, dtype P, FP32 online state.
            with T.Scope("V"):
                score_io_event = 0
                output_io_event = 2
                p_store_event = 0
                output_store_event = 1
                T.set_flag("V", "MTE2", score_io_event)
                T.set_flag("V", "MTE2", score_io_event + 1)
                T.set_flag("V", "MTE2", output_io_event)
                T.set_flag("MTE3", "V", p_store_event)
                T.set_flag("MTE3", "V", output_store_event)
                T.tile.arith_progression(
                    col_pos_ub,
                    T.float32(0.0),
                    T.float32(1.0),
                    STACK_BLOCKS * block,
                )

                for core_index in T.serial(my_iters):
                    task = cid + core_index * num_cores
                    qb = task // heads
                    head = task % heads
                    q_row = qb * block
                    q_valid = T.if_then_else(
                        q_row + block <= rt_valid_rows,
                        block,
                        rt_valid_rows - q_row,
                    )
                    sparse_task = qb >= rt_dense_prefix_blocks and qb < rt_sparse_end_block
                    active_count = T.if_then_else(
                        sparse_task,
                        plan_count[qb, head],
                        rt_num_blocks,
                    )
                    active_stacks = T.ceildiv(active_count, STACK_BLOCKS)
                    row_offset = vid * half_m

                    T.tile.fill(running_o_ub, 0.0)
                    T.tile.fill(running_max, NEG_INF)
                    T.tile.fill(running_sum, 0.0)

                    for pipe_step in T.serial(maximum_stacks + PRELOAD_DISTANCE):
                        if pipe_step < active_stacks:
                            sm_stack = pipe_step
                            sm_slot = sm_stack % TASK_SLOTS
                            T.wait_cross_flag(s_ready)
                            if sm_stack == active_stacks - 1:
                                valid_lanes = T.min(
                                    STACK_BLOCKS,
                                    active_count - sm_stack * STACK_BLOCKS,
                                )
                                valid_cols = (valid_lanes - 1) * block + rt_kv_tail_rows
                                T.tile.broadcast(
                                    mask_source_ub,
                                    col_pos_ub,
                                    axis=0,
                                    tmp=tmp_ub,
                                )
                                T.tile.compare(
                                    mask_ub,
                                    mask_source_ub,
                                    T.cast(valid_cols, "float32"),
                                    "LT",
                                )

                            for row_loop in T.serial(half_m // ROW_TILE + 1):
                                if row_loop < half_m // ROW_TILE:
                                    load_row_start = row_loop * ROW_TILE
                                    if row_loop % 2 == 0:
                                        T.wait_flag("V", "MTE2", score_io_event)
                                        T.copy(
                                            workspace_s[
                                                sm_slot,
                                                cid,
                                                row_offset + load_row_start : row_offset + load_row_start + ROW_TILE,
                                                :,
                                            ],
                                            score_work_ub_ping,
                                        )
                                        T.set_flag("MTE2", "V", score_io_event)
                                    else:
                                        T.wait_flag("V", "MTE2", score_io_event + 1)
                                        T.copy(
                                            workspace_s[
                                                sm_slot,
                                                cid,
                                                row_offset + load_row_start : row_offset + load_row_start + ROW_TILE,
                                                :,
                                            ],
                                            score_work_ub_pong,
                                        )
                                        T.set_flag("MTE2", "V", score_io_event + 1)
                                if row_loop >= 1:
                                    delayed_row_loop = row_loop - 1
                                    row_start = delayed_row_loop * ROW_TILE
                                    if delayed_row_loop % 2 == 0:
                                        T.wait_flag("MTE2", "V", score_io_event)
                                        T.tile.mul(score_work_ub_ping, score_work_ub_ping, scale)
                                        if sm_stack == active_stacks - 1:
                                            T.tile.select(
                                                score_work_ub_ping,
                                                mask_ub,
                                                score_work_ub_ping,
                                                T.float32(NEG_INF),
                                                "VSEL_TENSOR_SCALAR_MODE",
                                            )
                                        T.tile.block_reduce_max(
                                            reduce_stage1,
                                            score_work_ub_ping,
                                            ROW_TILE * STACK_BLOCKS * block // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_max(
                                            reduce_stage2,
                                            reduce_stage1,
                                            ROW_TILE * STACK_BLOCKS * block // 8 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_max(
                                            tile_max,
                                            reduce_stage2,
                                            ROW_TILE * STACK_BLOCKS * block // 64 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        if sm_stack == 0:
                                            T.tile.mul(new_max, tile_max, 1.0)
                                            T.tile.fill(rescale, 0.0)
                                        else:
                                            T.tile.max(
                                                new_max,
                                                running_max[row_start : row_start + ROW_TILE, :],
                                                tile_max,
                                            )
                                            T.tile.sub(
                                                rescale,
                                                running_max[row_start : row_start + ROW_TILE, :],
                                                new_max,
                                            )
                                            T.tile.exp(rescale, rescale)
                                        T.tile.brcb_experiment(
                                            brcb_ub,
                                            new_max,
                                            ROW_TILE // 8,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        for score_chunk in T.serial(STACK_BLOCKS * block // 64):
                                            score_col = score_chunk * 64
                                            T.tile.row_expand_sub_experiment(
                                                score_work_ub_ping[:, score_col : score_col + 64],
                                                score_work_ub_ping[:, score_col : score_col + 64],
                                                brcb_ub,
                                            )
                                        T.pipe_barrier("v")
                                        T.tile.exp(score_work_ub_ping, score_work_ub_ping)
                                        if sm_stack == active_stacks - 1:
                                            T.tile.select(
                                                score_work_ub_ping,
                                                mask_ub,
                                                score_work_ub_ping,
                                                T.float32(0.0),
                                                "VSEL_TENSOR_SCALAR_MODE",
                                            )
                                        T.tile.block_reduce_sum(
                                            reduce_stage1,
                                            score_work_ub_ping,
                                            ROW_TILE * STACK_BLOCKS * block // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_sum(
                                            reduce_stage2,
                                            reduce_stage1,
                                            ROW_TILE * STACK_BLOCKS * block // 8 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_sum(
                                            stack_sum,
                                            reduce_stage2,
                                            ROW_TILE * STACK_BLOCKS * block // 64 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")

                                        T.wait_flag("MTE3", "V", p_store_event)
                                        T.copy(score_work_ub_ping, probability_ub)
                                        T.set_flag("V", "MTE3", p_store_event)
                                        T.wait_flag("V", "MTE3", p_store_event)
                                        T.copy(
                                            probability_ub,
                                            workspace_p[
                                                sm_slot,
                                                cid,
                                                row_offset + row_start : row_offset + row_start + ROW_TILE,
                                                :,
                                            ],
                                        )
                                        T.set_flag("MTE3", "V", p_store_event)

                                        T.tile.mul(
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            rescale,
                                        )
                                        T.tile.add(
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            stack_sum,
                                        )
                                        T.tile.mul(
                                            rescale_slots[
                                                sm_slot,
                                                row_start : row_start + ROW_TILE,
                                                :,
                                            ],
                                            rescale,
                                            1.0,
                                        )
                                        T.tile.mul(
                                            running_max[row_start : row_start + ROW_TILE, :],
                                            new_max,
                                            1.0,
                                        )

                                        T.set_flag("V", "MTE2", score_io_event)
                                    else:
                                        T.wait_flag("MTE2", "V", score_io_event + 1)
                                        T.tile.mul(score_work_ub_pong, score_work_ub_pong, scale)
                                        if sm_stack == active_stacks - 1:
                                            T.tile.select(
                                                score_work_ub_pong,
                                                mask_ub,
                                                score_work_ub_pong,
                                                T.float32(NEG_INF),
                                                "VSEL_TENSOR_SCALAR_MODE",
                                            )
                                        T.tile.block_reduce_max(
                                            reduce_stage1,
                                            score_work_ub_pong,
                                            ROW_TILE * STACK_BLOCKS * block // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_max(
                                            reduce_stage2,
                                            reduce_stage1,
                                            ROW_TILE * STACK_BLOCKS * block // 8 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_max(
                                            tile_max,
                                            reduce_stage2,
                                            ROW_TILE * STACK_BLOCKS * block // 64 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        if sm_stack == 0:
                                            T.tile.mul(new_max, tile_max, 1.0)
                                            T.tile.fill(rescale, 0.0)
                                        else:
                                            T.tile.max(
                                                new_max,
                                                running_max[row_start : row_start + ROW_TILE, :],
                                                tile_max,
                                            )
                                            T.tile.sub(
                                                rescale,
                                                running_max[row_start : row_start + ROW_TILE, :],
                                                new_max,
                                            )
                                            T.tile.exp(rescale, rescale)
                                        T.tile.brcb_experiment(
                                            brcb_ub,
                                            new_max,
                                            ROW_TILE // 8,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        for score_chunk in T.serial(STACK_BLOCKS * block // 64):
                                            score_col = score_chunk * 64
                                            T.tile.row_expand_sub_experiment(
                                                score_work_ub_pong[:, score_col : score_col + 64],
                                                score_work_ub_pong[:, score_col : score_col + 64],
                                                brcb_ub,
                                            )
                                        T.pipe_barrier("v")
                                        T.tile.exp(score_work_ub_pong, score_work_ub_pong)
                                        if sm_stack == active_stacks - 1:
                                            T.tile.select(
                                                score_work_ub_pong,
                                                mask_ub,
                                                score_work_ub_pong,
                                                T.float32(0.0),
                                                "VSEL_TENSOR_SCALAR_MODE",
                                            )
                                        T.tile.block_reduce_sum(
                                            reduce_stage1,
                                            score_work_ub_pong,
                                            ROW_TILE * STACK_BLOCKS * block // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_sum(
                                            reduce_stage2,
                                            reduce_stage1,
                                            ROW_TILE * STACK_BLOCKS * block // 8 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")
                                        T.tile.block_reduce_sum(
                                            stack_sum,
                                            reduce_stage2,
                                            ROW_TILE * STACK_BLOCKS * block // 64 // 64,
                                            64,
                                            1,
                                            1,
                                            8,
                                        )
                                        T.pipe_barrier("v")

                                        T.wait_flag("MTE3", "V", p_store_event)
                                        T.copy(score_work_ub_pong, probability_ub)
                                        T.set_flag("V", "MTE3", p_store_event)
                                        T.wait_flag("V", "MTE3", p_store_event)
                                        T.copy(
                                            probability_ub,
                                            workspace_p[
                                                sm_slot,
                                                cid,
                                                row_offset + row_start : row_offset + row_start + ROW_TILE,
                                                :,
                                            ],
                                        )
                                        T.set_flag("MTE3", "V", p_store_event)

                                        T.tile.mul(
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            rescale,
                                        )
                                        T.tile.add(
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            running_sum[row_start : row_start + ROW_TILE, :],
                                            stack_sum,
                                        )
                                        T.tile.mul(
                                            rescale_slots[
                                                sm_slot,
                                                row_start : row_start + ROW_TILE,
                                                :,
                                            ],
                                            rescale,
                                            1.0,
                                        )
                                        T.tile.mul(
                                            running_max[row_start : row_start + ROW_TILE, :],
                                            new_max,
                                            1.0,
                                        )

                                        T.set_flag("V", "MTE2", score_io_event + 1)
                            T.set_cross_flag("MTE3", p_ready, 2)

                        if pipe_step >= PRELOAD_DISTANCE:
                            out_stack = pipe_step - PRELOAD_DISTANCE
                            if out_stack < active_stacks:
                                out_slot = out_stack % TASK_SLOTS
                                for row_tile in T.serial(half_m // ROW_TILE):
                                    row_start = row_tile * ROW_TILE
                                    T.tile.brcb_experiment(
                                        brcb_ub,
                                        rescale_slots[
                                            out_slot,
                                            row_start : row_start + ROW_TILE,
                                            :,
                                        ],
                                        ROW_TILE // 8,
                                        1,
                                        8,
                                    )
                                    T.pipe_barrier("v")
                                    for output_chunk in T.serial(dim // 64):
                                        output_col = output_chunk * 64
                                        T.tile.row_expand_mul_experiment(
                                            running_o_ub[
                                                row_start : row_start + ROW_TILE,
                                                output_col : output_col + 64,
                                            ],
                                            running_o_ub[
                                                row_start : row_start + ROW_TILE,
                                                output_col : output_col + 64,
                                            ],
                                            brcb_ub,
                                        )
                                    T.pipe_barrier("v")
                                T.wait_cross_flag(o_ready)
                                T.wait_flag("V", "MTE2", output_io_event)
                                T.copy(
                                    workspace_o[
                                        out_slot,
                                        cid,
                                        row_offset : row_offset + half_m,
                                        :,
                                    ],
                                    partial_o_ub,
                                )
                                T.set_flag("MTE2", "V", output_io_event)
                                T.wait_flag("MTE2", "V", output_io_event)
                                T.tile.add(
                                    running_o_ub,
                                    running_o_ub,
                                    partial_o_ub,
                                )
                                T.set_flag("V", "MTE2", output_io_event)

                    T.tile.max(running_sum, running_sum, SUMEXP_FLOOR)
                    for row_tile in T.serial(half_m // ROW_TILE):
                        row_start = row_tile * ROW_TILE
                        T.tile.brcb_experiment(
                            brcb_ub,
                            running_sum[row_start : row_start + ROW_TILE, :],
                            ROW_TILE // 8,
                            1,
                            8,
                        )
                        T.pipe_barrier("v")
                        for output_chunk in T.serial(dim // 64):
                            output_col = output_chunk * 64
                            T.tile.row_expand_div_experiment(
                                running_o_ub[
                                    row_start : row_start + ROW_TILE,
                                    output_col : output_col + 64,
                                ],
                                running_o_ub[
                                    row_start : row_start + ROW_TILE,
                                    output_col : output_col + 64,
                                ],
                                brcb_ub,
                            )
                        T.pipe_barrier("v")
                    T.wait_flag("MTE3", "V", output_store_event)
                    T.copy(running_o_ub, output_ub)
                    T.set_flag("V", "MTE3", output_store_event)
                    T.wait_flag("V", "MTE3", output_store_event)
                    valid_for_aiv = T.if_then_else(
                        q_valid >= row_offset + half_m,
                        half_m,
                        T.if_then_else(
                            q_valid > row_offset,
                            q_valid - row_offset,
                            0,
                        ),
                    )
                    if valid_for_aiv > 0:
                        if output_order == "original":
                            T.copy(
                                output_ub[0:valid_for_aiv, :],
                                reordered_output[
                                    q_row + row_offset : q_row + row_offset + valid_for_aiv,
                                    head * dim : (head + 1) * dim,
                                ],
                            )
                        else:
                            T.copy(
                                output_ub[0:valid_for_aiv, :],
                                output[
                                    q_row + row_offset : q_row + row_offset + valid_for_aiv,
                                    head * dim : (head + 1) * dim,
                                ],
                            )
                    T.set_flag("MTE3", "V", output_store_event)

                T.wait_flag("V", "MTE2", score_io_event)
                T.wait_flag("V", "MTE2", score_io_event + 1)
                T.wait_flag("V", "MTE2", output_io_event)
                T.wait_flag("MTE3", "V", p_store_event)
                T.wait_flag("MTE3", "V", output_store_event)

            # Phase 3: reverse worker-major records inside the same launch.
            if output_order == "original":
                T.sync_all()
                with T.Scope("V"):
                    T.set_flag("MTE3", "MTE2", 0)
                    T.set_flag("MTE3", "MTE2", 1)
                    inverse_worker = cid * VEC_NUM + vid
                    inverse_iters = T.if_then_else(
                        inverse_worker < rt_record_count,
                        T.ceildiv(
                            rt_record_count - inverse_worker,
                            num_cores * VEC_NUM,
                        ),
                        0,
                    )
                    if inverse_iters > 0:
                        T.copy(
                            reorder_records_flat[
                                inverse_worker * rt_records_per_worker * 4 : inverse_worker * rt_records_per_worker * 4
                                + inverse_iters * 4
                            ],
                            records_ub[0 : inverse_iters * 4],
                        )
                        T.set_flag("MTE2", "S", 0)
                        T.wait_flag("MTE2", "S", 0)
                    for record_iter in T.serial(inverse_iters):
                        src_row = records_ub[record_iter * 4]
                        dst_row = records_ub[record_iter * 4 + 1]
                        row_count = records_ub[record_iter * 4 + 2]
                        T.wait_flag("MTE3", "MTE2", record_iter % 2)
                        T.copy(
                            reordered_output[dst_row : dst_row + row_count, :],
                            inverse_stage_ub[record_iter % 2, 0:row_count, :],
                        )
                        T.set_flag("MTE2", "MTE3", record_iter % 2)
                        T.wait_flag("MTE2", "MTE3", record_iter % 2)
                        T.copy(
                            inverse_stage_ub[record_iter % 2, 0:row_count, :],
                            output[src_row : src_row + row_count, :],
                        )
                        T.set_flag("MTE3", "MTE2", record_iter % 2)
                    T.wait_flag("MTE3", "MTE2", 0)
                    T.wait_flag("MTE3", "MTE2", 1)

    return main


__all__ = [
    "PASS_CONFIGS",
    "h3_rainfusion_attention",
]
