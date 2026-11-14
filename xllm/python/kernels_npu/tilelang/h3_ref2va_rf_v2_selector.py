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
"""MiniMax-H3 Ref2VA fused rf_v2 block selector — RainFusion ABI variant (A3, Expert mode).

Candidate kernel for the TileLang RainFusion plan Stage 1
(/data/workspace/lwd/learn/rainfussion/h3_ref2va_tilelang_rainfusion_plan.md §5).
Derived from the frozen ``h3_ref2va_rf_v2_selector.py`` (evidence snapshot in
``stage0_evidence/``); the original file is NOT modified.

Differences vs the frozen compact-ABI selector:

  * Phase A additionally reorders V (``reordered_v`` output), so the kernel
    emits the full reordered TND Q/K/V triple consumed by the attention stage.
  * Outputs use the TileLang attention plan ABI directly:
      plan_ids    INT32 [total_blocks, heads, align8(total_blocks)] (470,4,472)
      plan_count  INT32 [total_blocks, 8]                           (470,8)
    Rows ``[0, ff_blocks)`` and ``[vfb, total_blocks)`` are dense rows:
    count = total_blocks and ids = [0..total_blocks-1] (static template
    written by the Vector cores). Rows ``[ff_blocks, vfb)`` are sparse rows:
    Top-R over ALL total_blocks logical KV blocks (no dense-context filter)
    UNION first-frame blocks UNION dense-context blocks, ascending, -1 padded.
  * Internal compaction is INT32 in UB and stores directly to the padded plan;
    there is no INT64 conversion or metadata-normalize launch.

The production geometry, per-frame remainder reorder builder, validation, host
launcher, and device kernel are all implemented in this file.

Phase structure, flag protocol and parser/landing rules are inherited from
the frozen file — see its module docstring for the full reading guide. New
Phase D flags (per pipe pair, added to the inherited map):

  Phase D (V):  S->MTE3 0  per-row: INT32 plan ready to store
                S->MTE3 1  one-time: dense INT32 template ready to store

Selector semantics are ported from MindIE-SD
``mindiesd/layers/flash_attn/sparse_flash_attn_rf_v2.py`` (Copyright (c)
Huawei Technologies Co., Ltd., licensed under Mulan PSL v2,
http://license.coscl.org.cn/MulanPSL2). The full-ABI row semantics match
``get_multi_span_blockwise_mask`` + ``get_mask_index`` exactly: retained
(dense) query rows select every KV block; sparse rows keep the ranked union;
ids ascend; padding is -1; dtype is INT32.

Local semantic differences (same policy as the frozen selector):

- production TopK keeps a deterministic fixed ``topk_r`` set instead of the
  ``score >= kth`` threshold mask,
- pooling/scores run on BF16 pools with FP32 accumulation, the reference
  softmax is skipped because it is strictly monotonic per row.

TVMScript requires live parameter annotations, so this file must not import
postponed annotations.

Ported verbatim from the frozen TileLang delivery
``examples/h3_ref2va_rf_v2_selector/h3_ref2va_rf_v2_selector_rfa.py``
(SHA256 4289f558a750c88d3c6eb62c6014b6b5f786c8195a3d652782c82db715c3935a).
The kernel body below is byte-identical to the frozen source; only the
torch-dependent host helpers (record-table builder, workspace allocator,
output validators, host launcher) are omitted here and reimplemented in C++
by ``h3_ref2va_rf_v2_selector_wrapper.cpp``.
"""

import tilelang
import tilelang.language as T
from tilelang.intrinsics import make_nz_layout, make_zn_layout

BLOCK, HEADS, DIM, CORES, VEC_NUM = 128, 4, 128, 24, 2
M_TILE, N_TILE, CHUNK_ROWS = 16, 128, 64

FLAG_FIRST_FRAME, FLAG_DENSE, FLAG_TAIL = 1, 2, 4


def _ceildiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _align_up(a: int, b: int) -> int:
    return _ceildiv(a, b) * b


def _derive(valid_rows: int, grid_t: int, grid_h: int, grid_w: int, block: int = BLOCK) -> tuple[int, ...]:
    video_rows = grid_t * grid_h * grid_w
    total_blocks = _ceildiv(valid_rows, block)
    tail_rows = valid_rows - (total_blocks - 1) * block
    ff_rows = grid_h * grid_w
    ff_blocks = _ceildiv(ff_rows, block)
    video_full_blocks = video_rows // block
    num_sq = video_full_blocks - ff_blocks
    return (
        video_rows,
        total_blocks,
        tail_rows,
        ff_rows,
        ff_blocks,
        video_full_blocks,
        num_sq,
        _align_up(num_sq, M_TILE),
        _align_up(total_blocks, 16),
    )


def _check_shape(total_rows: int, valid_rows: int, video_start: int, grid_t: int, grid_h: int, grid_w: int) -> None:
    video_rows, _tb, _tail, _ff, ff_blocks, vfb, num_sq, _qp, _kp = _derive(valid_rows, grid_t, grid_h, grid_w)
    if total_rows % BLOCK or not 0 < valid_rows <= total_rows:
        raise ValueError("total_rows must be block aligned and cover valid_rows")
    if min(video_start, grid_t, grid_h, grid_w) < 0 or grid_t * grid_h * grid_w == 0:
        raise ValueError("video start and grid dimensions must be valid")
    if video_start + video_rows > valid_rows or ff_blocks > vfb or num_sq < 1:
        raise ValueError("invalid sparse-video geometry")


RING_SLOTS = 2  # score-ring slots per core (C/D ping-pong)
SCORE_READY = 0  # cross flag: C -> V, ring slot written
SCORE_FREE = 1  # cross flag: V -> C, ring slot consumed

UB_LIMIT = 196352
L1_LIMIT = 512 * 1024
REDUCE_TMP_BYTES = 32 * 1024  # ReduceSum<float,[128,128],dim=0> workspace

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: False,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: False,
    # Matches the accepted examples/h3_video_sparse_attention/final kernel:
    # auto V/S pipe sync on top of the manual flag discipline below.
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC_VS: True,
}


# ====================================================================
# Kernel
# ====================================================================
@tilelang.jit(pass_configs=PASS_CONFIGS)
def h3_ref2va_rf_v2_selector_rfa(
    total_rows: int,
    valid_rows: int,
    video_start: int,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    record_count: int,
    topk_r: int,
    heads: int = HEADS,
    dim: int = DIM,
    block: int = BLOCK,
    num_cores: int = CORES,
    cap_total_blocks: int = 0,
    cap_record_count: int = 0,
    cap_topk_r: int = 0,
    cap_total_rows: int = 0,
):
    """Build one compile-time H3 shape specialization.

    CAPACITY versus LOGICAL geometry. The capacity axes size every PHYSICAL buffer,
    stride and pool extent; the logical axes are the ACTIVE lengths the kernel actually
    walks. Defaulting a capacity to 0 means "same as logical", which reproduces the
    previous exact-key behaviour bit for bit, so the frozen production key and the snow
    key are unaffected. Setting a capacity strictly larger than the logical value lets
    ONE compiled object serve every geometry that fits, instead of one exact AOT key per
    prompt length. TopK/Sort only ever see the ACTUAL candidate count, never the
    capacity, so the selection result is a function of the logical geometry alone.
    """
    if heads != HEADS or dim != DIM or block != BLOCK or num_cores != CORES:
        raise ValueError("selector fixes heads=4, dim=128, block=128, cores=24")
    _check_shape(total_rows, valid_rows, video_start, grid_t, grid_h, grid_w)
    dtype = "bfloat16"
    accum_dtype = "float32"
    workers = num_cores * VEC_NUM
    sm_scale = (1.0 / dim) ** 0.5

    # ---- record table (worker-major; see build_reorder_records) ----
    records_per_worker = _ceildiv(record_count, workers)
    records_padded = workers * records_per_worker

    _vr, total_blocks, tail_rows, _fr, ff_blocks, vfb, num_sq, q_pool_pad, k_pool_pad = _derive(
        valid_rows, grid_t, grid_h, grid_w, block
    )
    if not 1 <= topk_r <= total_blocks or record_count < 1:
        raise ValueError("invalid topk_r or record_count")
    ctx_blocks = total_blocks - vfb  # dense-context KV blocks [vfb, total_blocks)
    dense_qb_count = ff_blocks + ctx_blocks  # dense query rows of the full ABI
    dense_row_tasks = dense_qb_count * heads  # (qb, head) dense output rows

    # ---- capacity resolution: 0 means "identical to the logical value" ----
    cap_total_blocks = cap_total_blocks or total_blocks
    cap_record_count = cap_record_count or record_count
    cap_topk_r = cap_topk_r or topk_r
    cap_total_rows = cap_total_rows or total_rows
    if (
        cap_total_blocks < total_blocks
        or cap_record_count < record_count
        or cap_topk_r < topk_r
        or cap_total_rows < total_rows
    ):
        raise ValueError("capacity must cover the logical geometry")
    # The tail block's GM->UB copy reads a FULL 128-row block, so the physical storage
    # must cover ceil(valid_rows/block)*block. Both qualified geometries satisfy this
    # exactly (N13: 470*128 = 60160 = total_rows; 1440x736: 476*128 = 60928 = total_rows),
    # and total_rows <= cap_total_rows keeps the read inside the capacity buffer.
    if _ceildiv(valid_rows, block) * block > total_rows:
        raise ValueError("tail block requires total_rows to cover ceil(valid_rows/block)*block")
    cap_records_per_worker = _ceildiv(cap_record_count, workers)
    cap_records_padded = workers * cap_records_per_worker
    cap_q_pool_pad = _align_up(cap_total_blocks - _ceildiv(grid_h * grid_w, block), M_TILE)
    cap_k_pool_pad = _align_up(cap_total_blocks, 16)

    # Full-ABI row capacity: every PHYSICAL row is cap_total_blocks wide, while the
    # ACTIVE plan length per row stays the logical count. Keeping the stride and the
    # active length separate is what lets one object serve many logical geometries.
    capacity = cap_total_blocks
    capacity_pad = _align_up(capacity, 8)
    m_tiles, n_tiles = q_pool_pad // M_TILE, _ceildiv(k_pool_pad, N_TILE)
    last_n_cols = k_pool_pad - (n_tiles - 1) * N_TILE

    # Sort works on a 32-element repeat granularity: aligned = ceil(actual_num/32)*32,
    # dst must hold 2*aligned interleaved (value,index) pairs, and src must hold aligned
    # elements because positions beyond actual_num are padded with -inf before sorting.
    # Sizing on an 8-element granularity agrees only while ceil(K/32)*32 == align_up(K,8),
    # i.e. up to K=96. At K=97 the 32-granularity jumps to 128 while align_up(97,8) gives
    # 104 and align_up(194,8) gives 200, which overrun. The corrected formulas size from
    # the CAPACITY topk_r on the 32-granularity, so one object covers every K up to the
    # capacity; at K=94 and K=96 they give exactly the previous 96/192.
    sort_aligned = _align_up(cap_topk_r, 32)
    score_row_pad = _align_up(capacity, 32)
    pairs_pad = 2 * sort_aligned
    idx_pad = sort_aligned
    score_tasks = heads * m_tiles
    pool_tasks = heads * (k_pool_pad + q_pool_pad)
    # Compile-time CAPACITY companions. The zero-pad region is bounded by the capacity
    # pools, so its worst-case task count stays a compile-time constant while the actual
    # split is computed at runtime inside the kernel.
    cap_pool_tasks = heads * (cap_k_pool_pad + cap_q_pool_pad)
    cap_per_head_zero = cap_k_pool_pad + cap_q_pool_pad
    cap_n_tiles = _ceildiv(cap_k_pool_pad, N_TILE)
    blocks_per_head = total_blocks + num_sq  # pool tasks per head (K then Q)
    k_zero = k_pool_pad - total_blocks  # k_pool zero-pad rows per head
    per_head_zero = k_zero + (q_pool_pad - num_sq)  # zero-pad tasks per head
    rows_per_vid = M_TILE // VEC_NUM  # ring rows per AIV per score tile

    # ---- UB address plan (phases A / B / D never overlap in time) ----
    # Phase A: stage 0..128K | Phase B: pool_in 0..64K, pool_f32 64K..128K,
    # reduce_tmp 128K..160K, pool_out 160K.., pool_bf16 ..<=165376 |
    # records_ub reuses the pool_out address (A vs B disjoint) |
    # Phase D: compact INT32 selection buffers from address 0.
    stage_bytes = RING_SLOTS * CHUNK_ROWS * heads * dim * 2
    ub_stage_addr = 0
    ub_pool_in_addr = 0
    ub_pool_f32_addr = ub_pool_in_addr + 2 * block * dim * 2
    ub_reduce_tmp_addr = ub_pool_f32_addr + block * dim * 4
    ub_pool_out_addr = ub_reduce_tmp_addr + REDUCE_TMP_BYTES
    ub_pool_bf16_addr = ub_pool_out_addr + 2 * dim * 4
    assert stage_bytes <= UB_LIMIT
    assert ub_pool_bf16_addr + 2 * dim * 2 <= UB_LIMIT
    assert ub_pool_out_addr + cap_records_per_worker * 16 <= UB_LIMIT

    ub_score_row_addr = 0
    ub_pairs_addr = _align_up(ub_score_row_addr + score_row_pad * 4, 32)
    ub_topk_tmp_addr = _align_up(ub_pairs_addr + pairs_pad * 4, 32)
    topk_tmp_elems = 4 * score_row_pad
    ub_idx_f32_addr = _align_up(ub_topk_tmp_addr + topk_tmp_elems * 4, 32)
    ub_idx_i32_addr = _align_up(ub_idx_f32_addr + idx_pad * 4, 32)
    ub_sel_addr = _align_up(ub_idx_i32_addr + idx_pad * 4, 64)
    ub_count_addr = _align_up(ub_sel_addr + capacity_pad * 4, 32)
    ub_dense_i32_addr = _align_up(ub_count_addr + 32, 64)
    ub_dense_count_i32_addr = _align_up(ub_dense_i32_addr + capacity_pad * 4, 32)
    ub_score_rows_addr = _align_up(ub_dense_count_i32_addr + 32, 64)
    assert ub_score_rows_addr + rows_per_vid * score_row_pad * 4 <= UB_LIMIT

    l1_q_addr = 0
    l1_k_addr = _align_up(l1_q_addr + M_TILE * dim * 2, 512)
    assert l1_k_addr + RING_SLOTS * N_TILE * dim * 2 <= L1_LIMIT
    assert M_TILE * dim * 2 <= 64 * 1024
    assert RING_SLOTS * M_TILE * N_TILE * 4 <= 128 * 1024

    @T.prim_func
    def main(
        q_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        k_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        v_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        records_gm: T.Tensor([cap_records_padded * 4], "int32"),
        plan_ids_gm: T.Tensor([cap_total_blocks, heads, capacity_pad], "int32"),
        plan_count_gm: T.Tensor([cap_total_blocks, 8], "int32"),
        rq_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        rk_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        rv_gm: T.Tensor([cap_total_rows, heads * dim], dtype),
        q_pool_gm: T.Tensor([heads, cap_q_pool_pad, dim], dtype),
        k_pool_gm: T.Tensor([heads, cap_k_pool_pad, dim], dtype),
        ring_gm: T.Tensor([num_cores, RING_SLOTS, M_TILE, cap_k_pool_pad], accum_dtype),
        # ---- RUNTIME logical geometry (V3b) ----
        # Capacity sizes every physical buffer, stride and GM extent above; these
        # scalars are the ACTIVE lengths the kernel actually walks. Passing them as
        # arguments is what lets ONE dispatched object serve many logical geometries
        # instead of one exact AOT key per prompt length. TopK's src keeps a static
        # (capacity) shape because tilelang's topk requires a compile-time src shape;
        # only K and actual_num are symbolic, which is exactly this split.
        rt_total_blocks: T.int32,
        rt_topk_r: T.int32,
        rt_record_count: T.int32,
        rt_records_per_worker: T.int32,
        # The real tail extent and the video dense/sparse split, previously baked from the
        # compile-time grid. Physical strides and pool extents stay at CAPACITY; only the
        # iteration counts and the tail reduction extent become runtime.
        rt_tail_rows: T.int32,
        rt_ff_blocks: T.int32,
        rt_vfb: T.int32,
        rt_num_sq: T.int32,
    ):
        with T.Kernel(num_cores, is_npu=True) as (cid, vid):
            count_var = T.alloc_var("int32", init=0)
            # ---- on-chip buffers (order fixed; addresses from the plan above) ----
            stage_ub = T.alloc_ub([RING_SLOTS, CHUNK_ROWS, heads * dim], dtype)  # A: record staging
            pool_in_ub = T.alloc_ub([2, block, dim], dtype)  # B: block input (double)
            pool_f32_ub = T.alloc_ub([block, dim], accum_dtype)  # B: fp32 work (V-only, single)
            reduce_tmp_ub = T.alloc_ub([REDUCE_TMP_BYTES], "uint8")  # B: ReduceSum workspace
            pool_out_ub = T.alloc_ub([2, 1, dim], accum_dtype)  # B: fp32 pool row (double)
            pool_bf16_ub = T.alloc_ub([2, 1, dim], dtype)  # B: bf16 pool row (double)
            score_row_ub = T.alloc_ub([score_row_pad], accum_dtype)  # D: one score row
            pairs_ub = T.alloc_ub([pairs_pad], accum_dtype)  # D: topk (val,idx) pairs
            topk_tmp_ub = T.alloc_ub([4 * score_row_pad], accum_dtype)  # D: TopK sort workspace
            idx_f32_ub = T.alloc_ub([idx_pad], accum_dtype)  # D: gathered indices (fp32)
            idx_i32_ub = T.alloc_ub([idx_pad], "int32")  # D: gathered indices (int32)
            sel_ub = T.alloc_ub([capacity_pad], "int32")  # D: compacted row (int32)
            count_ub = T.alloc_ub([8], "int32")  # D: sparse count staging (int32)
            q_l1 = T.alloc_L1([M_TILE, dim], dtype)  # C: q_pool tile
            k_l1 = T.alloc_L1([RING_SLOTS, N_TILE, dim], dtype)  # C: k_pool tiles (double)
            l0a = T.alloc_L0A([M_TILE, dim], dtype)
            l0b = T.alloc_L0B([RING_SLOTS, dim, N_TILE], dtype)
            l0c = T.alloc_L0C([RING_SLOTS, M_TILE, N_TILE], accum_dtype)
            records_ub = T.alloc_ub([cap_records_per_worker * 4], "int32")  # A: worker record slice (capacity)
            dense_i32_ub = T.alloc_ub([capacity_pad], "int32")  # D: dense template (int32)
            dense_count_i32_ub = T.alloc_ub([8], "int32")  # D: dense count 470 (int32)
            score_rows_ub = T.alloc_ub([rows_per_vid, score_row_pad], accum_dtype)
            T.annotate_layout({q_l1: make_zn_layout(q_l1), k_l1: make_nz_layout(k_l1)})
            T.annotate_address(
                {
                    stage_ub: ub_stage_addr,
                    pool_in_ub: ub_pool_in_addr,
                    pool_f32_ub: ub_pool_f32_addr,
                    reduce_tmp_ub: ub_reduce_tmp_addr,
                    pool_out_ub: ub_pool_out_addr,
                    pool_bf16_ub: ub_pool_bf16_addr,
                    score_row_ub: ub_score_row_addr,
                    pairs_ub: ub_pairs_addr,
                    topk_tmp_ub: ub_topk_tmp_addr,
                    idx_f32_ub: ub_idx_f32_addr,
                    idx_i32_ub: ub_idx_i32_addr,
                    sel_ub: ub_sel_addr,
                    count_ub: ub_count_addr,
                    records_ub: ub_pool_out_addr,
                    dense_i32_ub: ub_dense_i32_addr,
                    dense_count_i32_ub: ub_dense_count_i32_addr,
                    score_rows_ub: ub_score_rows_addr,
                    q_l1: l1_q_addr,
                    k_l1: l1_k_addr,
                    l0a: 0,
                    l0b: 0,
                    l0c: 0,
                }
            )

            # ================================================================
            # Phase A — record-aware reorder pack of Q/K/V (AIV, MTE2/MTE3 only)
            # ================================================================
            # Tasks: the worker-major record table; worker w owns the contiguous
            # slice [w*records_per_worker, ...). One preload DMA brings the slice
            # into records_ub; each record then moves its row span Q, K and V
            # through the 64 KiB ping-pong stage (slot = it % 2):
            #   wait slot free -> GM->UB -> wait loaded -> UB->GM -> slot free.
            with T.Scope("V"):
                T.set_flag("MTE3", "MTE2", 0)  # initial: both stage slots free
                T.set_flag("MTE3", "MTE2", 1)
                worker = cid * VEC_NUM + vid
                rec_iters = T.if_then_else(worker < rt_record_count, T.ceildiv(rt_record_count - worker, workers), 0)
                if rec_iters > 0:
                    # One flat DMA preloads this worker's contiguous worker-major
                    # record slice; per-record metadata is then read from UB
                    # instead of three blocking GM scalar loads per record. The
                    # copy must stay 1D: a [rows, 4] int32 region lowers to a
                    # 16-byte DataCopyPad block with isPad=false, which faults
                    # (MTE write out of range); the flat byte-length block takes
                    # the legal aligned/isPad paths instead.
                    T.copy(
                        records_gm[
                            worker * rt_records_per_worker * 4 : worker * rt_records_per_worker * 4 + rec_iters * 4
                        ],
                        records_ub[0 : rec_iters * 4],
                    )
                    T.set_flag("MTE2", "S", 0)
                    T.wait_flag("MTE2", "S", 0)
                for it in T.serial(rec_iters):
                    src0 = records_ub[it * 4]
                    dst0 = records_ub[it * 4 + 1]
                    rows = records_ub[it * 4 + 2]
                    # Q span: GM -> stage -> reordered GM.
                    T.wait_flag("MTE3", "MTE2", it % 2)
                    T.copy(q_gm[src0 : src0 + rows, :], stage_ub[it % 2, 0:rows, :])
                    T.set_flag("MTE2", "MTE3", it % 2)
                    T.wait_flag("MTE2", "MTE3", it % 2)
                    T.copy(stage_ub[it % 2, 0:rows, :], rq_gm[dst0 : dst0 + rows, :])
                    T.set_flag("MTE3", "MTE2", it % 2)
                    # K span: same chain over the same stage slot.
                    T.wait_flag("MTE3", "MTE2", it % 2)
                    T.copy(k_gm[src0 : src0 + rows, :], stage_ub[it % 2, 0:rows, :])
                    T.set_flag("MTE2", "MTE3", it % 2)
                    T.wait_flag("MTE2", "MTE3", it % 2)
                    T.copy(stage_ub[it % 2, 0:rows, :], rk_gm[dst0 : dst0 + rows, :])
                    T.set_flag("MTE3", "MTE2", it % 2)
                    # V span: same chain over the same stage slot (RFA addition).
                    T.wait_flag("MTE3", "MTE2", it % 2)
                    T.copy(v_gm[src0 : src0 + rows, :], stage_ub[it % 2, 0:rows, :])
                    T.set_flag("MTE2", "MTE3", it % 2)
                    T.wait_flag("MTE2", "MTE3", it % 2)
                    T.copy(stage_ub[it % 2, 0:rows, :], rv_gm[dst0 : dst0 + rows, :])
                    T.set_flag("MTE3", "MTE2", it % 2)
                T.wait_flag("MTE3", "MTE2", 0)  # drain
                T.wait_flag("MTE3", "MTE2", 1)

            T.sync_all()  # reordered Q/K/V visible to every core

            # ================================================================
            # Phase B — contiguous block pooling (AIV, 2-slot double buffer)
            # ================================================================
            # Identical to the frozen selector (pools only consume Q/K).
            with T.Scope("V"):
                T.set_flag("V", "MTE2", 0)  # initial: both pool_in slots free
                T.set_flag("V", "MTE2", 1)
                T.set_flag("MTE3", "V", 0)  # initial: both out slots free
                T.set_flag("MTE3", "V", 1)
                worker = cid * VEC_NUM + vid
                rt_blocks_per_head = rt_total_blocks + rt_num_sq
                pool_task_count = heads * rt_blocks_per_head
                pool_iters = T.if_then_else(worker < pool_task_count, T.ceildiv(pool_task_count - worker, workers), 0)
                for it in T.serial(pool_iters):
                    task = worker + it * workers
                    # Every ping-pong use inlines the `it % 2` expression
                    # instead of a let binding: the earlier let-bound variant
                    # lost the modulo in production codegen (event ids >= 2
                    # and pool_in offsets beyond the 2-slot buffer => AIV
                    # timeout deadlock). Inline expressions verify correct
                    # on the production specialization.
                    head_c = task // rt_blocks_per_head
                    r_c = task % rt_blocks_per_head
                    # -- load stage (MTE2) --
                    T.wait_flag("V", "MTE2", it % 2)
                    head_l = task // rt_blocks_per_head
                    r_l = task % rt_blocks_per_head
                    if r_l < rt_total_blocks:
                        if r_l == rt_total_blocks - 1:
                            T.copy(
                                # SINGLE-FACTOR FIX: the tail GM->UB copy takes the FULL static
                                # block extent, exactly as every non-tail block does. A runtime
                                # copy extent lowers to copy_gm_to_ub<BF16,128,128>(..., rt_tail_rows, ...)
                                # whose maskShapeM < dstM drives the helper's padGap Duplicate path
                                # (tl_templates/ascend/common.h:254-263), and that nested path uses a
                                # HARDCODED event 0 which collides with the outer Phase B double-buffer
                                # events 0/1. Reading the full 128 rows keeps the compile-time extent
                                # (100 for N13, 128 here) so padGap is never entered. The rows beyond
                                # valid_rows are storage padding and are discarded downstream: only the
                                # zero-fill + cast consumes rt_tail_rows.
                                rk_gm[r_l * block : (r_l + 1) * block, head_l * dim : (head_l + 1) * dim],
                                pool_in_ub[it % 2, :, :],
                            )
                        else:
                            T.copy(
                                rk_gm[r_l * block : r_l * block + block, head_l * dim : (head_l + 1) * dim],
                                pool_in_ub[it % 2, :, :],
                            )
                    else:
                        iq_l = r_l - rt_total_blocks
                        T.copy(
                            rq_gm[
                                (rt_ff_blocks + iq_l) * block : (rt_ff_blocks + iq_l + 1) * block,
                                head_l * dim : (head_l + 1) * dim,
                            ],
                            pool_in_ub[it % 2, :, :],
                        )
                    T.set_flag("MTE2", "V", it % 2)
                    # -- compute stage (V) --
                    T.wait_flag("MTE2", "V", it % 2)
                    if r_c == rt_total_blocks - 1:
                        # Tail block: fewer than `block` rows are valid. reduce_sum requires a
                        # STATIC reduce extent ("Expected static reduce extents" if it is a
                        # runtime scalar), so the reduction runs over the full static `block`
                        # extent and the rows beyond the real tail are EXPLICITLY ISOLATED by
                        # zeroing the whole fp32 work buffer first and casting only the real
                        # rt_tail_rows into it. Zero rows contribute nothing to the sum, and the
                        # mean then divides by the runtime rt_tail_rows, so rows outside
                        # valid_rows can never influence the pooled value.
                        T.tile.fill(pool_f32_ub, T.float32(0.0))
                        T.tile.cast(
                            pool_f32_ub[0:rt_tail_rows, :],
                            pool_in_ub[it % 2, 0:rt_tail_rows, :],
                            "CAST_NONE",
                            rt_tail_rows * dim,
                        )
                        T.set_flag("V", "MTE2", it % 2)  # pool_in slot free
                        T.wait_flag("MTE3", "V", it % 2)  # out slot free
                        T.reduce_sum(pool_f32_ub, pool_out_ub[it % 2, :, :], dim=0, tmp=reduce_tmp_ub)
                        T.tile.div(
                            pool_out_ub[it % 2, :, :], pool_out_ub[it % 2, :, :], T.Cast(accum_dtype, rt_tail_rows)
                        )
                    else:
                        T.tile.cast(pool_f32_ub, pool_in_ub[it % 2, :, :], "CAST_NONE", block * dim)
                        T.set_flag("V", "MTE2", it % 2)
                        T.wait_flag("MTE3", "V", it % 2)
                        T.reduce_sum(pool_f32_ub, pool_out_ub[it % 2, :, :], dim=0, tmp=reduce_tmp_ub)
                        T.tile.div(pool_out_ub[it % 2, :, :], pool_out_ub[it % 2, :, :], T.float32(float(block)))
                    T.tile.cast(pool_bf16_ub[it % 2, :, :], pool_out_ub[it % 2, :, :], "CAST_RINT", dim)
                    # -- store stage (MTE3) --
                    T.set_flag("V", "MTE3", 0)
                    T.wait_flag("V", "MTE3", 0)
                    if r_c < rt_total_blocks:
                        T.copy(pool_bf16_ub[it % 2, :, :], k_pool_gm[head_c, r_c : r_c + 1, :])
                    else:
                        iq_c = r_c - rt_total_blocks
                        T.copy(pool_bf16_ub[it % 2, :, :], q_pool_gm[head_c, iq_c : iq_c + 1, :])
                    T.set_flag("MTE3", "V", it % 2)  # out slot free
                T.wait_flag("V", "MTE2", 0)  # drain
                T.wait_flag("V", "MTE2", 1)

                # -- zero-pad pool rows (no MTE2 stage; simple serial chain) --
                rt_k_zero = cap_k_pool_pad - rt_total_blocks
                rt_q_zero = cap_q_pool_pad - rt_num_sq
                rt_per_head_zero = rt_k_zero + rt_q_zero
                if cap_per_head_zero > 0:
                    zero_task_count = cap_pool_tasks - pool_task_count
                    zero_iters = T.if_then_else(
                        worker < zero_task_count,
                        T.ceildiv(zero_task_count - worker, workers),
                        0,
                    )
                    for zit in T.serial(zero_iters):
                        ztask = worker + zit * workers
                        zhead = ztask // rt_per_head_zero
                        zr = ztask % rt_per_head_zero
                        T.wait_flag("MTE3", "V", 0)
                        T.tile.fill(pool_bf16_ub[0, :, :], T.cast(0, dtype))
                        T.set_flag("V", "MTE3", 0)
                        T.wait_flag("V", "MTE3", 0)
                        if zr < rt_k_zero:
                            T.copy(
                                pool_bf16_ub[0, :, :],
                                k_pool_gm[
                                    zhead,
                                    rt_total_blocks + zr : rt_total_blocks + zr + 1,
                                    :,
                                ],
                            )
                        else:
                            zq = zr - rt_k_zero
                            T.copy(
                                pool_bf16_ub[0, :, :],
                                q_pool_gm[
                                    zhead,
                                    rt_num_sq + zq : rt_num_sq + zq + 1,
                                    :,
                                ],
                            )
                        T.set_flag("MTE3", "V", 0)
                T.wait_flag("MTE3", "V", 0)  # drain
                T.wait_flag("MTE3", "V", 1)

                # The direct attention plan has eight count lanes per query block.
                # Initialize complete rows before per-head stores so lanes 4..7
                # stay zero without racing the independently scheduled heads.
                T.tile.fill(count_ub, 0)
                T.set_flag("V", "MTE3", 6)
                T.wait_flag("V", "MTE3", 6)
                count_row_iters = T.if_then_else(
                    worker < rt_total_blocks,
                    T.ceildiv(rt_total_blocks - worker, workers),
                    0,
                )
                for count_row_iter in T.serial(count_row_iters):
                    count_qb = worker + count_row_iter * workers
                    T.copy(count_ub, plan_count_gm[count_qb, :])
                T.set_flag("MTE3", "V", 6)
                T.wait_flag("MTE3", "V", 6)

            T.sync_all()  # pools visible to the Cube cores

            # ================================================================
            # Phase C — pooled-QK scores (AIC)  ||  Phase D consumes via ring
            # ================================================================
            # Identical to the frozen selector.
            with T.Scope("C"):
                T.set_flag("MTE1", "MTE2", 0)  # initial: q_l1/k_l1/l0* free
                T.set_flag("MTE1", "MTE2", 1)
                T.set_flag("MTE1", "MTE2", 2)
                T.set_flag("M", "MTE1", 0)
                T.set_flag("M", "MTE1", 1)
                T.set_flag("M", "MTE1", 2)
                T.set_flag("FIX", "M", 0)
                T.set_flag("FIX", "M", 1)
                rt_m_tiles = T.ceildiv(rt_num_sq, M_TILE)
                rt_score_tasks = heads * rt_m_tiles
                score_iters = T.if_then_else(cid < rt_score_tasks, T.ceildiv(rt_score_tasks - cid, num_cores), 0)
                for it in T.serial(score_iters):
                    task = cid + it * num_cores
                    head = task // rt_m_tiles
                    m_tile = task % rt_m_tiles
                    slot = it % RING_SLOTS
                    T.wait_cross_flag(SCORE_FREE)
                    # -- q tile: GM -> L1 -> L0A (once per task) --
                    T.wait_flag("MTE1", "MTE2", 0)
                    T.copy(q_pool_gm[head, m_tile * M_TILE : (m_tile + 1) * M_TILE, :], q_l1)
                    T.set_flag("MTE2", "MTE1", 0)
                    T.wait_flag("MTE2", "MTE1", 0)
                    T.wait_flag("M", "MTE1", 0)
                    T.copy(q_l1, l0a)
                    T.set_flag("MTE1", "M", 0)
                    T.set_flag("MTE1", "MTE2", 0)  # q_l1 free
                    # -- n tiles: k -> L0B -> mma -> L0C -> ring --
                    # Hoisted OUT of the unrolled loop: the loop is unrolled at the
                    # CAPACITY tile count, so these runtime values must be in scope for
                    # every unrolled instance and for the last-N guard below. Defining
                    # them inside the body left the guard referencing an out-of-scope
                    # value and codegen emitted an undeclared identifier.
                    # A Python-level symbolic expression does NOT survive codegen inside an
                    # unrolled loop (it emits an undeclared condval). Carry both values in
                    # EXPLICIT cross-loop state via T.alloc_var, the same mechanism count_var
                    # already uses in this kernel.
                    n_tiles_var = T.alloc_var("int32", init=0)
                    last_cols_var = T.alloc_var("int32", init=0)
                    n_tiles_var = T.ceildiv(rt_total_blocks, N_TILE)
                    last_cols_var = rt_total_blocks - (n_tiles_var - 1) * N_TILE
                    for n in T.unroll(cap_n_tiles):
                        # ACTIVE-N GUARD: the loop is unrolled at the CAPACITY tile count, which is
                        # >= the runtime tile count ceil(rt_total_blocks/N_TILE). Every tile at or beyond
                        # n_tiles_var is INACTIVE and must do nothing at all: no GM copy, no event
                        # set/wait, no MMA and no FIX store. Guarding only `cols` was insufficient --
                        # an inactive tile still evaluated cols=last_cols_var=1, so it copied one k_pool
                        # row, issued an MMA against l0a AFTER the last active tile had released it via
                        # set_flag("M","MTE1",0), and wrote one ring column. The l0a release therefore
                        # stays at the last ACTIVE n and no MMA can follow it. Each set/wait pair lives
                        # entirely inside one iteration, so skipping a whole iteration keeps the flag
                        # credit balance unchanged and the post-loop drain still consumes exactly the
                        # initial credits.
                        if n < n_tiles_var:
                            kside = n % 2
                            cols = T.if_then_else(n < n_tiles_var - 1, N_TILE, last_cols_var)
                            T.wait_flag("MTE1", "MTE2", 1 + kside)
                            T.copy(
                                k_pool_gm[head, n * N_TILE : n * N_TILE + cols, :],
                                k_l1[kside, 0:cols, :],
                            )
                            T.set_flag("MTE2", "MTE1", 1 + kside)
                            T.wait_flag("MTE2", "MTE1", 1 + kside)
                            T.wait_flag("M", "MTE1", 1 + kside)
                            T.copy(
                                k_l1[kside, 0:cols, :],
                                l0b[kside, :, :],
                                transpose=True,
                                real_k=dim,
                                real_n=cols,
                            )
                            T.set_flag("MTE1", "MTE2", 1 + kside)  # k_l1 slot free
                            T.set_flag("MTE1", "M", 1 + kside)
                            T.wait_flag("MTE1", "M", 1 + kside)
                            if n == 0:
                                T.wait_flag("MTE1", "M", 0)  # l0a ready (first mma only)
                            T.wait_flag("FIX", "M", kside)
                            T.mma(l0a, l0b[kside, :, :], l0c[kside, :, :], init=True, n_actual=cols)
                            T.set_flag("M", "MTE1", 1 + kside)  # l0b slot free
                            if n == n_tiles_var - 1:
                                T.set_flag("M", "MTE1", 0)  # l0a free after last mma
                            T.set_flag("M", "FIX", kside)
                            T.wait_flag("M", "FIX", kside)
                            T.copy(
                                l0c[kside, :, 0:cols],
                                ring_gm[cid, slot, :, n * N_TILE : n * N_TILE + cols],
                            )
                            T.set_flag("FIX", "M", kside)  # l0c slot free
                    T.set_cross_flag("FIX", SCORE_READY, 2)  # publish ring slot
                # drain local flags, then the 2 leftover SCORE_FREE credits
                T.wait_flag("MTE1", "MTE2", 0)
                T.wait_flag("MTE1", "MTE2", 1)
                T.wait_flag("MTE1", "MTE2", 2)
                T.wait_flag("M", "MTE1", 0)
                T.wait_flag("M", "MTE1", 1)
                T.wait_flag("M", "MTE1", 2)
                T.wait_flag("FIX", "M", 0)
                T.wait_flag("FIX", "M", 1)
                T.wait_cross_flag(SCORE_FREE)
                T.wait_cross_flag(SCORE_FREE)

            # ================================================================
            # Phase D — dense template fill + TopK union compaction (AIV)
            # ================================================================
            # Runs concurrently with Phase C via the ring cross flags.
            #
            # D.1: every worker builds the dense INT32 row template once and stores
            #     its strided share of the
            #     dense_row_tasks (qb, head) rows: count=total_blocks,
            #     ids=[0..total_blocks-1]. Dense qbs are [0,ff) and [vfb,total).
            # D.2 sparse loop (same ring protocol as the frozen selector). Per
            #     score tile: batch-load this AIV's ring rows, release
            #     SCORE_FREE, then per valid row sq (output row sq+ff_blocks):
            #       scale -> TopK -> sort IDs ascending
            #       merge first-frame + video TopK + dense context
            #       one fixed-size INT32 plan store per output.
            with T.Scope("V"):
                T.set_cross_flag("MTE2", SCORE_FREE, 2)  # 2 initial ring credits
                T.set_cross_flag("MTE2", SCORE_FREE, 2)
                T.set_flag("V", "MTE2", 0)  # initial: score_rows free
                T.set_flag("MTE3", "V", 1)  # initial: row buffers free
                T.set_flag("MTE3", "S", 2)  # initial: selection buffers free

                # ---- D.1 dense rows: static template, strided row stores ----
                worker = cid * VEC_NUM + vid
                T.tile.arith_progression(dense_i32_ub, 0, 1, rt_total_blocks)
                T.set_flag("V", "S", 5)
                T.wait_flag("V", "S", 5)
                for i in T.serial(capacity_pad - rt_total_blocks):
                    dense_i32_ub[rt_total_blocks + i] = -1
                dense_count_i32_ub[0] = rt_total_blocks
                T.set_flag("S", "MTE3", 1)
                T.wait_flag("S", "MTE3", 1)
                rt_ctx_blocks = rt_total_blocks - rt_vfb
                rt_dense_row_tasks = (rt_ff_blocks + rt_ctx_blocks) * heads
                dense_iters = T.if_then_else(
                    worker < rt_dense_row_tasks,
                    T.ceildiv(rt_dense_row_tasks - worker, workers),
                    0,
                )
                for dit in T.serial(dense_iters):
                    drow = worker + dit * workers
                    dq = drow // heads
                    dh = drow % heads
                    # dq enumerates dense qbs: [0,ff) then [vfb,total_blocks)
                    dqb = T.if_then_else(dq < rt_ff_blocks, dq, rt_vfb + dq - rt_ff_blocks)
                    T.copy(dense_i32_ub, plan_ids_gm[dqb, dh, :])
                    T.copy(dense_count_i32_ub[0:1], plan_count_gm[dqb, dh : dh + 1])

                # ---- D.2 sparse rows: ring consumption + TopK union ----
                rt_m_tiles = T.ceildiv(rt_num_sq, M_TILE)
                rt_score_tasks = heads * rt_m_tiles
                score_iters = T.if_then_else(cid < rt_score_tasks, T.ceildiv(rt_score_tasks - cid, num_cores), 0)
                for it in T.serial(score_iters):
                    task = cid + it * num_cores
                    head = task // rt_m_tiles
                    m_tile = task % rt_m_tiles
                    slot = it % RING_SLOTS
                    T.wait_cross_flag(SCORE_READY)
                    # Batch-load this AIV's ring rows, then release the slot.
                    T.wait_flag("V", "MTE2", 0)
                    T.copy(
                        ring_gm[cid, slot, vid * rows_per_vid : (vid + 1) * rows_per_vid, 0:cap_k_pool_pad],
                        score_rows_ub,
                    )
                    T.set_flag("MTE2", "V", 0)
                    T.set_cross_flag("MTE2", SCORE_FREE, 2)
                    T.wait_flag("MTE2", "V", 0)
                    for j in T.serial(rows_per_vid):
                        ring_row = vid * rows_per_vid + j
                        sq = m_tile * M_TILE + ring_row
                        if sq < rt_num_sq:
                            # count_var is declared directly in the prim_func.
                            T.wait_flag("MTE3", "V", 1)
                            T.copy(score_rows_ub[j, :], score_row_ub)
                            # -- vector stage: scale, TopK, gather, casts, fills --
                            T.tile.mul(score_row_ub, score_row_ub, sm_scale)
                            T.tile.topk(pairs_ub, score_row_ub, rt_topk_r, rt_total_blocks, tmp=topk_tmp_ub)
                            # Sort's actual_num MUST stay compile-time: tilelang derives
                            # repeatTimes = (actual_num+31)//32 from it and the InjectTmpBuffer
                            # pass Downcasts that workspace spec to IntImm, so a runtime scalar
                            # there fails with "Downcast from tir.FloorDiv to IntImm". Sort
                            # therefore runs over the CAPACITY K, and the tail beyond the logical
                            # K is neutralized to -inf so descending order pushes it past every
                            # real candidate. The merge loop below reads only [0, rt_topk_r).
                            for ti in T.serial(cap_topk_r):
                                if ti >= rt_topk_r:
                                    pairs_ub[2 * ti] = T.Cast(accum_dtype, -T.infinity(accum_dtype))
                                    pairs_ub[2 * ti + 1] = T.Cast(accum_dtype, 0)
                            T.tile.gather_mask(idx_f32_ub, pairs_ub, "P1010")
                            T.tile.sort(pairs_ub, idx_f32_ub, cap_topk_r, tmp=topk_tmp_ub)
                            T.tile.gather_mask(idx_f32_ub, pairs_ub, "P0101")
                            T.tile.cast(idx_i32_ub, idx_f32_ub, "CAST_ROUND", idx_pad)
                            T.tile.fill(sel_ub, -1)
                            T.set_flag("V", "S", 0)
                            T.wait_flag("V", "S", 0)
                            # -- scalar stage: merge sorted TopK with forced ranges --
                            T.wait_flag("MTE3", "S", 2)
                            count_var = 0
                            for b in T.serial(rt_ff_blocks):
                                sel_ub[count_var] = b
                                count_var = count_var + 1
                            for i in T.serial(rt_topk_r):
                                idv = idx_i32_ub[rt_topk_r - 1 - i]
                                if idv >= rt_ff_blocks and idv < rt_vfb:
                                    sel_ub[count_var] = idv
                                    count_var = count_var + 1
                            for i in T.serial(rt_total_blocks - rt_vfb):
                                sel_ub[count_var] = rt_vfb + i
                                count_var = count_var + 1
                            count_ub[0] = count_var
                            T.set_flag("S", "MTE3", 0)
                            T.wait_flag("S", "MTE3", 0)
                            T.copy(sel_ub, plan_ids_gm[sq + rt_ff_blocks, head, :])
                            T.copy(count_ub[0:1], plan_count_gm[sq + rt_ff_blocks, head : head + 1])
                            T.set_flag("MTE3", "V", 1)
                            T.set_flag("MTE3", "S", 2)
                    T.set_flag("V", "MTE2", 0)  # score_rows free for next tile
                T.wait_flag("V", "MTE2", 0)  # drain
                T.wait_flag("MTE3", "V", 1)
                T.wait_flag("MTE3", "S", 2)

    return main


__all__ = ["PASS_CONFIGS", "h3_ref2va_rf_v2_selector_rfa"]
