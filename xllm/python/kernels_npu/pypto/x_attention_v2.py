#!/usr/bin/env python3
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

"""x_attention: xLLM-style beam-decode attention op (delivery-form PyPTO kernel).

Single-launch kernel: core-split (shared||unshared) + SyncAll + CombineScale, built on
a FlashAttention pipeline using NBuffer + auto_mutex (NBuffer with current() auto-rotate
cursor, cross-core event_id synchronization for QK_READY/P_READY/PV_READY).

Semantics (matches xllm-ops x_attention, beam-search decode):

  * Decode step: each request (batch b) tracks ``beam`` beams; every beam contributes
    ONE query token.  Query layout is [batch*beam, Hq, D].
  * KV is split in two contiguous regions:
      - shared  : the request's prefix KV, SHARED by all beams of that request.
                  Either contiguous per-batch [sum(Lb), Hkv, D] (PagedShared == 0),
                  or a physical block cache [numBlocks*128, Hkv, D] gathered via
                  shared_block_table [B, maxBlocks] (PagedShared == 1).
      - unshared: each beam's own divergent KV, length decode_step.
                  Either direct logical-batch layout [B, beam, Hkv, maxDs, D]
                  (PagedUnshared == 0), or slot-gathered via unshared_block_table [B]
                  (PagedUnshared == 1).
  * GQA: Hq = group * Hkv; q-head h uses kv-head h // group.
  * Output = softmax over concat([shared_K, unshared_K]) @ concat([shared_V, unshared_V]),
    i.e. plain (non-causal) flash-attention over the two concatenated KV regions,
    computed as two partial (max, lse, out) results merged by the flash log-sum-exp
    combine (CombineScale), with a single cross-core barrier between.

Delivery-form interface (mirrors ops-transformer op_kernel conventions):

  * All I/O are raw ``pl.Ptr[pl.DT_UINT8]``; all shapes/params arrive in the
    ``tiling: OpTiling`` dataclass (host side == future C++ TILING_DATA_DEF layout).
  * Per-KV-mode variants are selected by the compile-time ``PagedShared`` /
    ``PagedUnshared`` tiling-key fields; each concrete key compiles one specialized
    kernel (dead addressing branch folded away by the parser).
  * The (o_s, m_s, l_s, o_u, m_u, l_u) partial workspaces live inside ONE external
    workspace buffer; offsets are derived in-kernel from the tiling shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pypto_pro.language as pl
import torch
from pypto_pro.language import Vf as vf  # noqa: N813
from pypto_pro.runtime.tilingkey import TilingKeyField

# ================================================================
#  Configuration
# ================================================================
QK_PRELOAD = 1
FIFO_SIZE = QK_PRELOAD + 1

# ================================================================
#  Tile dimensions and constants
# ================================================================
TS = 128
TKV = 128
TD = 128
TS_HALF = TS // 2
NEG_INF = -1e9

BLOCK_STRIDE_ND = TS >> 1 | 0x1
REPEAT_STRIDE_ND = 1
FLOAT_REP_SIZE = 64  # elements per fp32 register
# Width of the m/l partial path.  m and l are ONE fp32 scalar per row; the only reason
# they were ever 64 wide is that a vector store writes a full register.  combine reads
# element 0 alone (combine_row_vf's BRC_B32 loads), so the other lanes were pure padding
# -- 32KB of UB and, worse, 1KB of GM traffic per combine row for 16B of real data.
# 8 fp32 = 32B keeps every store/load 32B-aligned, the minimum granularity.
ML_W = 8
# combine block height: rows processed per iteration.  The combine sweep runs AFTER the
# global barrier, so it reuses the whole VEC space from address 0 (the pipeline buffers
# are dead by then) -- CMB_R is bounded by UB, not by the map above.
CMB_R = 32
D_LOOPS = TD // FLOAT_REP_SIZE
TAIL_D = TD % FLOAT_REP_SIZE
REDUCE_SIZE = 1

# Buffer sizes (bytes)
Q_F16 = TS * TD * 2
KT_F16 = TD * TKV * 2
V_F16 = TKV * TD * 2
P_F16 = TS * TKV * 2
QK_HALF_F32 = TS * TKV * 4
PV_HALF_F32 = TS * TD * 4

# VEC buffer sizes
VB4_KV = TS_HALF * TKV * 4
VB2_KV = TS_HALF * TKV * 2
VB4 = TS_HALF * TD * 4
VB2 = TS_HALF * TD * 2
VB6 = (TS_HALF + 1) * TKV * 2
VB_RED = TS_HALF * 1 * 4
VB_MASK = TS_HALF * TKV

# ================================================================
#  Buffer addresses
# ================================================================
# MAT (512KB) - L1 buffers
MA0_Q = 0
MA1_K = Q_F16 * 2
MA2_P = MA1_K + KT_F16 * 2
MA3_V = MA2_P + P_F16 * 3

# L0A/L0B/L0C addresses
LA0 = 0
LA1 = P_F16
RA0 = 0
RA1 = KT_F16
CA0 = 0
CA1 = QK_HALF_F32

# VEC (248KB) addresses
VA0 = 0
VA1 = VA0 + VB4_KV * 2
VA_GMAX0 = VA1 + VB6 * 2
VA_GMAX1 = VA_GMAX0 + VB_RED
VA_GMAX2 = VA_GMAX1 + VB_RED
VA_GSUM0 = VA_GMAX2 + VB_RED
VA_GSUM1 = VA_GSUM0 + VB_RED
VA_GSUM2 = VA_GSUM1 + VB_RED
VA_EXPMAX0 = VA_GSUM2 + VB_RED
VA_EXPMAX1 = VA_EXPMAX0 + VB_RED
VA_EXPMAX2 = VA_EXPMAX1 + VB_RED
VA7 = VA_EXPMAX2 + VB_RED
VA8 = VA7 + VB4
VA9 = VA8 + VB4 * 2
VB_ML = TS_HALF * ML_W * 4  # m_tile / l_tile, 2KB each (was 16KB at 64 wide)
VA10 = VA9 + VB2  # m_tile
VA11 = VA10 + VB_ML  # tmp_max
VA12 = VA11 + VB_RED  # tmp_sum
VA13 = VA12 + VB_RED  # l_tile
# Unshared merged-group mask tile (uint8, [TS_HALF, TKV]).
VA_UMASK = VA13 + VB_ML
# Per-row (w / we) window tables, int32 [TS, 8] (128 rows x 8 cols = 32B row
# width).  Built in-kernel once per launch by pl.setval (S pipe writes UB
# directly, no GM round-trip, no DCCI); the mask generator BRC-loads column 0
# of row (row_off + m) per item.  Two tiles: w_tbl_tile, we_tbl_tile.
VB_WTBL = TS * 8 * 4
VA_WTBL = VA_UMASK + VB_MASK
# ub_size for every Ascend950 variant is 253952 B (248KB) -- see the platform config,
# e.g. .../platform_config/Ascend950PR_9579.ini.  This is a hard limit, not a guideline.
UB_SIZE = 253952
assert VA_WTBL + 2 * VB_WTBL <= UB_SIZE, (VA_WTBL, 2 * VB_WTBL, UB_SIZE)


# ================================================================
#  Mutex IDs - Cube and Vector use independent buf_id spaces
# ================================================================
# Cube-only (inside section_cube): 0-11
#   Q L1: (0, 1), K L1: (2, 3), V L1: (4, 5)
#   L0A: (6, 7), L0B: (8, 9), L0C: (10, 11)
#
# Vector-only (inside section_vector): 0-11
#   tmp_vec: 0, p_f16: 1, reduce_dst: 2
#   gmax_rm: (3, 4), gsum: (5, 6)
#   exp_corr: (7, 8), running_o: 9, o_f16: 10, tile_nz: 11
#
# Cross-core shared (outside sections): 12-17
#   P MAT: (12, 13)
#   qk_vec UB: (14, 15)
#   pv_vec UB: (16, 17)

# Cross-core shared buffer IDs
QK_READY_FORWARD_IDS = (0, 1)
QK_READY_BARKWARD_IDS = (2, 3)
P_READY_FORWARD_IDS = (4, 5, 6)
PV_READY_FORWARD_IDS = (7, 8)
PV_READY_BARKWARD_IDS = (9, 10)


PV_CORE_STRIDE = 2 * FIFO_SIZE * TS


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign64(
    input_tile, dst_tile, max_tile, max_tile_st, sum_tile, s1_size, s2_size, scale
):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(
            dst_tile,
            vreg_dst_even_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf_unalign(
    input_tile, dst_tile, max_tile, max_tile_st, sum_tile, s1_size, s2_size, scale
):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)

        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_no_update_vf(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, s1_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)

        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign64(
    input_tile, dst_tile, max_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size, scale
):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(
            dst_tile,
            vreg_dst_even_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf_unalign(
    input_tile, dst_tile, max_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size, s2_size, scale
):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)

        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_nd_update_vf(input_tile, dst_tile, max_tile, tmp_max, tmp_max_st, tmp_exp_sum, s1_size, scale):
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_all)

        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_max, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(tmp_max, ureg_max, 0, post_update=True)
    vreg_in_max = vf.load_align(max_tile, 0)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    vreg_max_new = vf.load_align(tmp_max_st, 0)
    vreg_max_new = vf.max(vreg_in_max, vreg_max_new, preg_all)
    vf.store_align(tmp_max_st, vreg_max_new, preg_all)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)
    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(tmp_max_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(tmp_exp_sum, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(tmp_exp_sum, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def update_exp_sum(exp_diff, max_tile, tmp_max, sum_tile, tmp_sum):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)

    vreg_max = vf.load_align(max_tile, 0)
    vreg_max_tmp = vf.load_align(tmp_max, 0)
    vreg_exp_max = vf.exp_sub(vreg_max, vreg_max_tmp, preg_all)
    vf.store_align(exp_diff, vreg_exp_max, preg_all)
    vf.store_align(max_tile, vreg_max_tmp, preg_all)

    vreg_sum = vf.load_align(sum_tile, 0)
    vreg_sum_tmp = vf.load_align(tmp_sum, 0)
    vreg_exp_update = vf.mul(vreg_sum, vreg_exp_max, preg_all)
    vreg_exp_update = vf.add(vreg_exp_update, vreg_sum_tmp, preg_all)
    vf.store_align(sum_tile, vreg_exp_update, preg_all)


@pl.vector_function
def flash_update_basic_vf(dst_tile, cur_tile, pre_tile, exp_max_tile, s1_size, has_tail):
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(TAIL_D, dtype=pl.DT_FP32)
    for i in pl.range(0, s1_size):
        vreg_exp_max = vf.load_align(exp_max_tile, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        for j in pl.range(0, D_LOOPS):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + j * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_all)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_all)
            vf.store_align(dst_tile + (i * TD + j * FLOAT_REP_SIZE), vreg_add, preg_all)
        for _ in pl.range(0, has_tail):
            vreg_input_pre = vf.load_align(pre_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_input_cur = vf.load_align(cur_tile, i * TD + D_LOOPS * FLOAT_REP_SIZE)
            vreg_mul = vf.mul(vreg_exp_max, vreg_input_pre, preg_tail)
            vreg_add = vf.add(vreg_mul, vreg_input_cur, preg_tail)
            vf.store_align(dst_tile + (i * TD + D_LOOPS * FLOAT_REP_SIZE), vreg_add, preg_tail)


# ================================================================
#  Merged-group (cross-token x cross-kv-head) masked softmax variants.
#
#  The unshared path merges G groups -- a "group" being one (beam token, kv-head)
#  pair -- into ONE matmul: Q[G*group, D] @ K^T[D, G*maxDs].  Only the G diagonal
#  blocks are meaningful (row block i must only see col block i); everything else
#  is a cross-group product that must not enter the softmax.  Within a diagonal
#  block only the first `decode_step` of `max_decode_step` columns are real KV,
#  the rest is stale cache padding.  Both are expressed by one kernel-built mask
#  (0 = valid, 1 = invalid), applied by forcing invalid lanes to NEG_INF BEFORE
#  the row max, so exp(NEG_INF - max) == 0 and they contribute to neither the
#  row sum nor PV.
#
#  Masked lanes are written back to input_tile, so the second (exp/sum) pass
#  picks them up automatically -- exactly how the existing preg_tail masking works.
#
#  The mask itself is built IN-KERNEL (gen_umask_vf) -- byte-for-byte identical to
#  the old host-built tensor -- so no mask GM input exists anymore.
# ================================================================
@pl.vector_function
def gen_umask_vf(mask_tile, wtbl_w, wtbl_we, row_off, rows):
    """Build one work item's merged-group validity mask in UB.

    Byte-for-byte identical to the former host-built u_mask:
      mask[i*group : (i+1)*group,  i*maxds : i*maxds + ds] = 0   (valid)
      everything else = 1                                        (invalid)

    Per-row window bounds come from two kernel-built int32 tables `wtbl_w`
    (window start w) and `wtbl_we` (window end we), both indexed by ABSOLUTE
    item row.  The tables are UB-resident [TS, 8] int32 tiles written once per
    launch by pl.setval (S pipe -- no GM round-trip, no DCCI).  This sub-core's
    rows are absolute [row_off, row_off + rows); the generator BRC-loads
    column 0 of row (row_off + m), i.e. offset (row_off + m) * 8.  row_off and m
    are scalar offsets (flex `h*TS_HALF + m` precedent), so the generator stays
    in the provably-compilable flex decode_mask instruction set (BRC scalar
    loads + reg-reg compares; no div, no runtime state, no scalar-operand
    compares)."""
    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_all_b16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT16)
    merge_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    merge_unroll_bit = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    row_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    temp_reg = vf.create_mask(pattern=pl.MaskPattern.ALLF, dtype=pl.DT_UINT8)
    vreg_zero = vf.full(0, dtype=pl.DT_UINT16)
    vreg_one = vf.full(1, dtype=pl.DT_UINT16)
    index = vf.arange(0, dtype=pl.DT_INT32)  # columns 0..63
    index_unroll = vf.arange(64, dtype=pl.DT_INT32)  # columns 64..127

    for m in pl.range(0, rows):
        wreg = vf.load_align(wtbl_w, (row_off + m) * 8, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        wereg = vf.load_align(wtbl_we, (row_off + m) * 8, dist=pl.LoadDist.BRC_B32, dtype=pl.DT_INT32)
        merge_bit = vf.ge(index, wreg, preg_all)
        temp_reg = vf.lt(index, wereg, preg_all)
        merge_bit = vf.and_(merge_bit, temp_reg, preg_all)
        merge_unroll_bit = vf.ge(index_unroll, wreg, preg_all)
        temp_reg = vf.lt(index_unroll, wereg, preg_all)
        merge_unroll_bit = vf.and_(merge_unroll_bit, temp_reg, preg_all)
        row_reg, temp_reg = vf.de_interleave(merge_bit, merge_unroll_bit, dtype=pl.DT_UINT16)
        mask_b16 = vf.select(vreg_zero, vreg_one, row_reg)
        vf.store_align(mask_tile + m * TKV, mask_b16, preg_all_b16, dist=pl.StoreDist.PACK)


@pl.vector_function
def process_vec1_ug_unalign64(
    input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size, s2_size, scale
):
    """Merged-group masked softmax, s2_size <= 64 (single fp32 register)."""
    # Pre-declare the predicate register at UINT32 width so the LoadDist.DS mask
    # loads below resolve to a vsel-compatible register type.
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x = vf.muls(vreg_x, scale, preg_tail)

        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vreg_max = vf.reduce_max(vreg_x, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2 = vf.load_align(input_tile, i * TKV)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_tail)

        vreg_exp_sum = vf.reduce_sum(vreg_exp_even, preg_tail, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all_f16, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_dst_even_f16, vreg_dst_odd_f16 = vf.de_interleave(vreg_exp_even_f16, vreg_exp_even_f16)
        vf.store_align(
            dst_tile,
            vreg_dst_even_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


@pl.vector_function
def process_vec1_ug_unalign(input_tile, dst_tile, max_tile, max_tile_st, sum_tile, mask_tile, s1_size, s2_size, scale):
    """Merged-group masked softmax, 64 < s2_size <= 128 (two fp32 registers).
    s2_size == 128 also lands here: preg_tail becomes all-lanes, which is
    exactly the aligned case, so no separate ==128 variant is needed."""
    preg_mask = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    preg_mask_hi = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_UINT32)
    ureg_max = vf.unalign_reg_for_store()
    ureg_exp_sum = vf.unalign_reg_for_store()

    preg_all = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tail = vf.update_mask(s2_size - 64, dtype=pl.DT_FP32)
    preg_all_f16 = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=io_dtype)

    vreg_min = vf.full(NEG_INF)

    for m in pl.range(s1_size):
        vreg_x = vf.load_align(input_tile, m * TKV)
        vreg_x_unroll = vf.load_align(input_tile, m * TKV + 64)

        vreg_x = vf.muls(vreg_x, scale, preg_all)
        vreg_x_unroll = vf.muls(vreg_x_unroll, scale, preg_tail)

        vreg_x_unroll = vf.select(vreg_x_unroll, vreg_min, preg_tail)
        preg_mask = vf.load_align(mask_tile, m * TKV, dist=pl.LoadDist.DS)
        preg_mask_hi = vf.load_align(mask_tile, m * TKV + 64, dist=pl.LoadDist.DS)
        vreg_x = vf.select(vreg_min, vreg_x, preg_mask)
        vreg_x_unroll = vf.select(vreg_min, vreg_x_unroll, preg_mask_hi)
        vf.store_align(input_tile + m * TKV, vreg_x, preg_all)
        vf.store_align(input_tile + m * TKV + 64, vreg_x_unroll, preg_all)
        vreg_max_tmp = vf.max(vreg_x, vreg_x_unroll, preg_all)
        vreg_max = vf.reduce_max(vreg_max_tmp, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(max_tile, vreg_max, ureg_max, 1, post_update=True)
    vf.store_unalign_post(max_tile, ureg_max, 0, post_update=True)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)

    for i in pl.range(s1_size):
        vreg_max_2 = vf.load_align(max_tile_st, i, dist=pl.LoadDist.BRC_B32)
        vreg_x_2, vreg_x_unroll_2 = vf.load_align(input_tile, i * TKV, dist=pl.LoadDist.DINTLV_B32)
        vreg_exp_even = vf.exp_sub(vreg_x_2, vreg_max_2, preg_all)
        vreg_exp_odd = vf.exp_sub(vreg_x_unroll_2, vreg_max_2, preg_all)

        vreg_exp_sum = vf.add(vreg_exp_even, vreg_exp_odd, preg_all)
        vreg_exp_sum = vf.reduce_sum(vreg_exp_sum, preg_all, merge_mode=pl.MergeMode.ZEROING)
        vf.store_unalign(sum_tile, vreg_exp_sum, ureg_exp_sum, 1, post_update=True)

        vreg_exp_even_f16 = vf.astype(vreg_exp_even, preg_all, layout=pl.CastLayout.ZERO, dtype=io_dtype)
        vreg_exp_odd_f16 = vf.astype(vreg_exp_odd, preg_all, layout=pl.CastLayout.ONE, dtype=io_dtype)
        vreg_exp_f16 = vf.or_(vreg_exp_even_f16, vreg_exp_odd_f16, preg_all_f16)
        vf.store_align(
            dst_tile,
            vreg_exp_f16,
            preg_all_f16,
            block_stride=BLOCK_STRIDE_ND,
            repeat_stride=REPEAT_STRIDE_ND,
            data_copy_mode=pl.DataCopyMode.DATA_BLOCK_COPY,
            post_update=True,
        )
    vf.store_unalign_post(sum_tile, ureg_exp_sum, 0, post_update=True)


def compute_p_ug(
    ctx_p, sub_id, qk_vec_db, tile_nz_db, global_max, global_sum, p_mat_db, mask_db, w_tbl_tile, we_tbl_tile, scale
):
    """Softmax for the MERGED-GROUP unshared path.  Same structure as compute_p but
    (a) always the no_update form -- s2 = G*maxDs <= TKV so kv_loop is always 1,
    the ki>0 / running-max-update path is unreachable here -- and (b) the
    merged-group mask is applied (see process_vec1_ug_*).  The mask is built
    in-kernel by gen_umask_vf from UB-resident (w, we) tables (`w_tbl_tile` /
    `we_tbl_tile`) that are written once per launch by pl.setval (S pipe, no
    GM round-trip) -- replacing the former host-built mask."""
    p_eid = ctx_p.task_id % FIFO_SIZE

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    mask_buf = mask_db.next()
    q_idx_p = ctx_p.q_count % 3
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    row_off = ctx_p.first_s1 * sub_id

    # Mask rows follow the tile rows this sub-core owns; the (w, we) window
    # bounds for absolute rows [row_off, row_off + half) are BRC-read by
    # gen_umask_vf from the UB-resident tables (written once per launch by
    # pl.setval).  No per-item load: the tables are per-core UB, indexed by
    # ABSOLUTE row, so the generator needs row_off to locate its half's rows.
    pl.set_validshape(mask_buf, [ctx_p.half_s1, TKV])
    gen_umask_vf(mask_buf, w_tbl_tile, we_tbl_tile, row_off, ctx_p.half_s1)

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.s2_size <= 64:
        process_vec1_ug_unalign64(
            qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1, ctx_p.s2_size, scale
        )
    else:
        process_vec1_ug_unalign(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, mask_buf, ctx_p.half_s1, ctx_p.s2_size, scale)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2_size])
    pl.insert(cur_p_slot, tile_nz, [row_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


def compute_qk(
    ctx,
    ki,
    sq_off,
    q,
    k,
    cur_q_slot,
    k_l1_db,
    left_db,
    right_db,
    acc_db,
    qk_vec_db,
    task_id,
    left2,
    right2,
    acc2,
    sbt_t,
    sbt_stride,
):
    # --- compute_qk inlined ---
    if PagedShared == 1:  # noqa: F821
        # Paged shared KV: tile ki of batch b lives in physical block
        # shared_block_table[b*stride + ki] (blockSize == TKV == 128, so one KV
        # tile == one physical block; the final block's tail rows are handled by
        # the existing s2_size valid-shape).
        skv_off = pl.getval(sbt_t, ctx.b_idx * sbt_stride + ctx.ki) * TKV
    else:
        skv_off = ctx.s2SizeAcc + ctx.ki * TKV
    cur_k_slot = k_l1_db.next()
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

    if ki == 0:
        pl.set_validshape(cur_q_slot, [ctx.s1_size, TD])
        pl.load(cur_q_slot, q, [sq_off, ctx.n_idx, 0], order=[0, 2])
    pl.set_validshape(cur_k_slot, [TD, ctx.s2_size])
    pl.load(cur_k_slot, k, [skv_off, ctx.kv_n_idx, 0], order=[2, 0])

    pl.set_validshape(qk_left, [ctx.s1_size, TD])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [TD, ctx.s2_size])
    pl.move(qk_right, cur_k_slot)
    pl.set_validshape(qk_acc, [ctx.s1_size, ctx.s2_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [(ctx.s1_size + 1) // 2, ctx.s2_size])
    pl.set_validshape(qk_acc, [(ctx.s1_size + 1) // 2 * 2, (ctx.s2_size + 7) // 8 * 8])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, v, left2, right2, acc2, sbt_t, sbt_stride):
    if PagedShared == 1:  # noqa: F821
        sv_off = pl.getval(sbt_t, ctx.b_idx * sbt_stride + ctx.ki) * TKV
    else:
        sv_off = ctx.s2SizeAcc + ctx.ki * TKV
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    # # current() advances p_mat_db cursor to stay in sync with Vec,
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

    pl.set_validshape(cur_v_slot, [ctx.s2_size, TD])
    pl.load(cur_v_slot, v, [sv_off, ctx.kv_n_idx, 0], order=[0, 2])
    pl.set_validshape(cur_p_slot, [ctx.s1_size, ctx.s2_size])
    pl.set_validshape(pv_left, [ctx.s1_size, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)
    pl.set_validshape(pv_right, [ctx.s2_size, TD])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [ctx.s1_size, TD])
    pl.matmul(pv_acc, pv_left, pv_right)

    pv_slot = pv_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_slot, [(ctx.s1_size + 1) // 2, TD])
    pl.set_validshape(pv_acc, [(ctx.s1_size + 1) // 2 * 2, TD])
    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_qk_ug(
    ctx, ki, q_flat, uk5, cur_q_slot, k_l1_db, left_db, right_db, acc_db, qk_vec_db, task_id, left2, right2, acc2
):
    # A1 in-kernel unshared GATHER + GQA GROUP-FOLDING. One work item = (beam token,
    # kv-head ctx.kv_n_idx): the `group` q-heads that share this kv-head are folded into
    # the M dimension (s1_size = group), so QK is [group, D] @ [D, s2] instead of `group`
    # separate M=1 matmuls.
    #   Q: the group heads are consecutive rows in the flat view q_flat[B*beam*Hq, 1, D]
    #      at ctx.qrow -> load [group, D] via the proven order=[0,2] (a multi-row load
    #      over dim1/head is NOT reliable, hence the flat view).
    #   K^T: raw xllm 5D uk5[B, beam, Hkv, maxDs, D] at physical batch ctx.pb (block-
    #        table remapped), beam ctx.pm, kv-head ctx.kv_n_idx, tokens [ki*TKV:+s2];
    #        order=[4,3] => [D, s2].
    cur_k_slot = k_l1_db.next()
    qk_left = left_db.next()
    qk_right = right_db.next()
    qk_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

    if ki == 0:
        pl.set_validshape(cur_q_slot, [ctx.s1_size, TD])
        pl.load(cur_q_slot, q_flat, [ctx.qrow, 0, 0], order=[0, 2])
    pl.set_validshape(cur_k_slot, [TD, ctx.s2_size])
    pl.load(cur_k_slot, uk5, [ctx.pb, ctx.pm, 0], order=[2, 1])

    pl.set_validshape(qk_left, [ctx.s1_size, TD])
    pl.move(qk_left, cur_q_slot)
    pl.set_validshape(qk_right, [TD, ctx.s2_size])
    pl.move(qk_right, cur_k_slot)
    pl.set_validshape(qk_acc, [ctx.s1_size, ctx.s2_size])
    pl.matmul(qk_acc, qk_left, qk_right)

    qk_slot = qk_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_BARKWARD_IDS[task_id % 2])
    pl.set_validshape(qk_slot, [(ctx.s1_size + 1) // 2, ctx.s2_size])
    pl.set_validshape(qk_acc, [(ctx.s1_size + 1) // 2 * 2, (ctx.s2_size + 7) // 8 * 8])
    pl.move(qk_slot, qk_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=QK_READY_FORWARD_IDS[task_id % 2])


def compute_pv_ug(ctx, v_l1_db, p_mat_db, left_db, right_db, acc_db, pv_vec_db, uv5, left2, right2, acc2):
    # A1 in-kernel unshared GATHER: V loaded directly from the RAW xllm 5D layout
    # uv5[B, beam, Hkv, maxDs, D] at physical batch ctx.pb, beam ctx.pm, head
    # ctx.kv_n_idx, tokens [ki*TKV : +s2].  order=[3,4] => [s2, D] (non-transposed).
    pl.system.wait_cross_core(pipe=pl.PipeType.MTE1, event_id=P_READY_FORWARD_IDS[ctx.task_id % 3])
    cur_v_slot = v_l1_db.next()
    cur_p_slot = p_mat_db.next()
    pv_left = left_db.next()
    pv_right = right_db.next()
    pv_acc = acc_db.next()
    tmp1, tmp2, tmp3 = left2.next(), right2.next(), acc2.next()  # noqa: F841

    pl.set_validshape(cur_v_slot, [ctx.s2_size, TD])
    pl.load(cur_v_slot, uv5, [ctx.pb, ctx.pm, 0], order=[1, 2])
    pl.set_validshape(cur_p_slot, [ctx.s1_size, ctx.s2_size])
    pl.set_validshape(pv_left, [ctx.s1_size, ctx.s2_size])
    pl.move(pv_left, cur_p_slot)
    pl.set_validshape(pv_right, [ctx.s2_size, TD])
    pl.move(pv_right, cur_v_slot)
    pl.set_validshape(pv_acc, [ctx.s1_size, TD])
    pl.matmul(pv_acc, pv_left, pv_right)

    pv_slot = pv_vec_db.next()
    pl.system.wait_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_BARKWARD_IDS[ctx.task_id % 2])
    pl.set_validshape(pv_slot, [(ctx.s1_size + 1) // 2, TD])
    pl.set_validshape(pv_acc, [(ctx.s1_size + 1) // 2 * 2, TD])
    pl.move(pv_slot, pv_acc, acc_to_vec_mode=pl.AccToVecMode.DualModeSplitM)
    pl.system.set_cross_core(pipe=pl.PipeType.FIX, event_id=PV_READY_FORWARD_IDS[ctx.task_id % 2])


def compute_p(
    ctx_p, sub_id, qk_vec_db, tile_nz_db, tmp_max, tmp_sum, global_max, global_sum, exp_corr_db, p_mat_db, scale
) -> None:
    """Softmax on KQ tile -> P (full attention, no mask). Includes cross-core sync."""
    p_eid = ctx_p.task_id % FIFO_SIZE

    qk_slot = qk_vec_db.next()
    tile_nz = tile_nz_db.next()
    q_idx_p = ctx_p.q_count % 3
    gmax_p = global_max[q_idx_p]
    gsum_p = global_sum[q_idx_p]
    exp_diff = exp_corr_db[ctx_p.task_id % 3]
    row_off = ctx_p.first_s1 * sub_id

    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_FORWARD_IDS[p_eid])
    if ctx_p.ki == 0:
        if ctx_p.s2_size == 128:
            process_vec1_nd_no_update_vf(qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, ctx_p.half_s1, scale)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_no_update_vf_unalign64(
                qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, ctx_p.half_s1, ctx_p.s2_size, scale
            )
        else:
            process_vec1_nd_no_update_vf_unalign(
                qk_slot, tile_nz, gmax_p, gmax_p, gsum_p, ctx_p.half_s1, ctx_p.s2_size, scale
            )
    else:
        if ctx_p.s2_size == 128:
            process_vec1_nd_update_vf(qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1, scale)
        elif ctx_p.s2_size <= 64:
            process_vec1_nd_update_vf_unalign64(
                qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1, ctx_p.s2_size, scale
            )
        else:
            process_vec1_nd_update_vf_unalign(
                qk_slot, tile_nz, gmax_p, tmp_max, tmp_max, tmp_sum, ctx_p.half_s1, ctx_p.s2_size, scale
            )
        update_exp_sum(exp_diff, gmax_p, tmp_max, gsum_p, tmp_sum)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[p_eid])

    cur_p_slot = p_mat_db.next()
    pl.set_validshape(tile_nz, [ctx_p.half_s1, ctx_p.s2_size])
    pl.insert(cur_p_slot, tile_nz, [row_off, 0])
    pl.system.set_cross_core(pipe=pl.PipeType.MTE3, event_id=P_READY_FORWARD_IDS[ctx_p.task_id % 3])


@pl.vector_function
def bcast_one_vf(dst_tile, src_reduce, rows):
    """Broadcast per-row reduce [rows,1] into an ML_W-wide dst_tile [rows,ML_W].
    Only lane 0 is ever consumed (combine_row_vf broadcasts from it); ML_W is the
    32B store granularity, not a data requirement.  dst is a DEDICATED tile so the
    subsequent GM store never touches running_o -> no pipeline race."""
    preg = vf.update_mask(ML_W, dtype=pl.DT_FP32)
    for i in pl.range(rows):
        r = vf.load_align(src_reduce, i * REDUCE_SIZE, dist=pl.LoadDist.BRC_B32)
        vf.store_align(dst_tile + i * ML_W, r, preg)


def compute_gu_partial(
    ctx_gu,
    pv_vec_db,
    exp_corr_db,
    global_sum_buf,
    global_max_buf,
    running_o,
    o_f16,
    m_tile,
    l_tile,
    o_unnorm,
    m_out,
    l_out,
):
    """Like compute_gu but emits the UNNORMALIZED partial (O, m, l) for CombineScale:
    o_unnorm = sum_j exp(x_j - m)*V_j (no /l), m = row max (scaled), l = row sum.
    O via proven cast(running_o)->o_f16->store. m,l each broadcast into their OWN
    dedicated 64-wide tile (m_tile/l_tile), then stored to [T,N,64] (host reads
    [...,0]). Dedicated non-running_o tiles avoid the pipeline WAR race."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_off = ctx_gu.first_s1 * sub_id
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    gmax_gu = global_max_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
    has_tail = 0
    if TAIL_D != 0:
        has_tail = 1
    if ctx_gu.ki == 0:
        pl.move(running_o, pv_slot)
    else:
        # basic (no-div) accumulate on every tile, including the last
        flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, has_tail)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    if ctx_gu.ki == ctx_gu.kv_loop - 1:
        pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.store(o_unnorm, o_f16, [ctx_gu.q_off + row_off, ctx_gu.n_idx, 0], order=[0, 2])
        pl.set_validshape(gmax_gu, [ctx_gu.half_s1, 1])
        pl.set_validshape(m_tile, [ctx_gu.half_s1, ML_W])
        bcast_one_vf(m_tile, gmax_gu, ctx_gu.half_s1)
        pl.store(m_out, m_tile, [ctx_gu.q_off + row_off, ctx_gu.n_idx, 0], order=[0, 2])
        pl.set_validshape(gsum_gu, [ctx_gu.half_s1, 1])
        pl.set_validshape(l_tile, [ctx_gu.half_s1, ML_W])
        bcast_one_vf(l_tile, gsum_gu, ctx_gu.half_s1)
        pl.store(l_out, l_tile, [ctx_gu.q_off + row_off, ctx_gu.n_idx, 0], order=[0, 2])


def compute_gu_partial_ug(
    ctx_gu,
    pv_vec_db,
    exp_corr_db,
    global_sum_buf,
    global_max_buf,
    running_o,
    o_f16,
    m_tile,
    l_tile,
    o_unnorm,
    m_out,
    l_out,
):
    """GQA GROUP-FOLDED partial finalize for the unshared path. Identical online-softmax
    finalize as compute_gu_partial, but the M rows are the `group` q-heads of one beam
    token, so (O, m, l) are stored ALONG THE HEAD AXIS: o_out[ctx.tok, h0+row_off : +half_s1, :]
    via order=[1,2] (compute_gu_partial writes along the token axis, order=[0,2])."""
    pl.system.wait_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_FORWARD_IDS[ctx_gu.task_id % 2])

    sub_id = pl.get_subblock_idx()
    row_off = ctx_gu.first_s1 * sub_id
    pv_slot = pv_vec_db.next()
    gsum_gu = global_sum_buf[ctx_gu.q_count % 3]
    gmax_gu = global_max_buf[ctx_gu.q_count % 3]
    exp_corr_gu = exp_corr_db[ctx_gu.task_id % 3]
    pl.set_validshape(running_o, [ctx_gu.half_s1, TD])
    pl.set_validshape(pv_slot, [ctx_gu.half_s1, TD])
    has_tail = 0
    if TAIL_D != 0:
        has_tail = 1
    if ctx_gu.ki == 0:
        pl.move(running_o, pv_slot)
    else:
        flash_update_basic_vf(running_o, pv_slot, running_o, exp_corr_gu, ctx_gu.half_s1, has_tail)
    pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[ctx_gu.task_id % 2])
    if ctx_gu.ki == ctx_gu.kv_loop - 1:
        # Merged groups span MULTIPLE beam tokens, so the output rows are no longer
        # confined to one token's head axis -- they are a contiguous run in the
        # FLATTENED [T*Hq, 1, *] view starting at ctx_gu.tok (= the same flat row the
        # Q rows came from).  Storing along dim0 also avoids the multi-row dim1 store.
        o_row = ctx_gu.tok + row_off
        pl.set_validshape(o_f16, [ctx_gu.half_s1, TD])
        pl.cast(o_f16, running_o, mode=pl.RoundMode.CAST_ROUND)
        pl.store(o_unnorm, o_f16, [o_row, 0, 0], order=[0, 2])
        pl.set_validshape(gmax_gu, [ctx_gu.half_s1, 1])
        pl.set_validshape(m_tile, [ctx_gu.half_s1, ML_W])
        bcast_one_vf(m_tile, gmax_gu, ctx_gu.half_s1)
        pl.store(m_out, m_tile, [o_row, 0, 0], order=[0, 2])
        pl.set_validshape(gsum_gu, [ctx_gu.half_s1, 1])
        pl.set_validshape(l_tile, [ctx_gu.half_s1, ML_W])
        bcast_one_vf(l_tile, gsum_gu, ctx_gu.half_s1)
        pl.store(l_out, l_tile, [o_row, 0, 0], order=[0, 2])


# ================================================================
#  Tiling data (host-side layout; mirrors the future C++ TILING_DATA_DEF)
# ================================================================
@dataclass
class OpTiling:
    batch: int
    beam: int
    hq: int
    hkv: int
    shared_total: int
    u_maxds: int
    scale: float
    sbt_stride: int
    shared_core_num: int
    total_cores: int


# ================================================================
#  TilingKey: per-KV-mode compile-time variants
# ================================================================
class XAttnV2TilingKey:
    """KV addressing modes (parser folds each field per concrete key):

    PagedShared:   0 = contiguous shared prefix [sum(Lb), Hkv, D]
                   1 = paged block-table gather [numBlocks*128, Hkv, D]
    PagedUnshared: 0 = direct logical-batch addressing (no table)
                   1 = slot-table gather via unshared_block_table [B]
    """

    PagedShared = TilingKeyField(bits=1, values=[0, 1])
    PagedUnshared = TilingKeyField(bits=1, values=[0, 1])


@pl.jit(
    arch="a5",
    auto_mutex=True,
    compile_timeout=200,
    tiling_key=XAttnV2TilingKey,
    datatype={
        "query": "io_dtype",
        "shared_key_block": "io_dtype",
        "shared_value_block": "io_dtype",
        "unshared_key_block": "io_dtype",
        "unshared_value_block": "io_dtype",
        "attn_out": "io_dtype",
    },
)
def x_attention_v2_kernel(
    query: pl.Ptr[pl.DT_UINT8],
    shared_key_block: pl.Ptr[pl.DT_UINT8],
    shared_value_block: pl.Ptr[pl.DT_UINT8],
    unshared_key_block: pl.Ptr[pl.DT_UINT8],
    unshared_value_block: pl.Ptr[pl.DT_UINT8],
    unshared_block_table: pl.Ptr[pl.DT_UINT8],
    shared_kv_lens: pl.Ptr[pl.DT_UINT8],
    decode_step: pl.Ptr[pl.DT_UINT8],
    shared_block_table: pl.Ptr[pl.DT_UINT8],
    attn_out: pl.Ptr[pl.DT_UINT8],
    workspace: pl.Ptr[pl.DT_UINT8],
    tiling: OpTiling,
):
    # ========== Tiling-derived shapes ==========
    n_dim = tiling.hq
    kv_n_dim = tiling.hkv
    group = n_dim // kv_n_dim
    u_batch = tiling.batch
    u_beam = tiling.beam
    num_tokens = u_batch * u_beam
    u_maxds = tiling.u_maxds
    shared_total = tiling.shared_total
    scale = tiling.scale
    sbt_stride = tiling.sbt_stride
    shared_core_num = tiling.shared_core_num
    total_cores = tiling.total_cores
    u_gpb = u_beam * kv_n_dim  # groups per batch
    g_merge = pl.min(TKV // u_maxds, TS // group)
    g_merge = pl.min(g_merge, u_gpb)
    core_id = pl.get_block_idx() // pl.get_subblock_num()

    # ========== Reconstruct typed tensor views from Ptr (delivery pattern) ==========
    flat_rows = num_tokens * n_dim
    unshared_inner = u_beam * kv_n_dim * u_maxds
    q = pl.make_tensor(query, [num_tokens, n_dim, TD], [n_dim * TD, TD, 1], dtype=io_dtype)
    # q_flat is a flat [T*Hq, 1, D] view derived from the SAME q buffer (row stride
    # = D, so consecutive rows are adjacent head slots / next-token head 0) -- the
    # merged-group unshared path loads its Q rows contiguously this way.  No host
    # reshape needed: make_tensor reinterprets the q pointer with new strides.
    q_flat = pl.make_tensor(query, [flat_rows, 1, TD], [TD, TD, 1], dtype=io_dtype)
    shared_k = pl.make_tensor(shared_key_block, [shared_total, kv_n_dim, TD], [kv_n_dim * TD, TD, 1], dtype=io_dtype)
    shared_v = pl.make_tensor(shared_value_block, [shared_total, kv_n_dim, TD], [kv_n_dim * TD, TD, 1], dtype=io_dtype)
    unshared_k = pl.make_tensor(
        unshared_key_block, [u_batch, unshared_inner, TD], [unshared_inner * TD, TD, 1], dtype=io_dtype
    )
    unshared_v = pl.make_tensor(
        unshared_value_block, [u_batch, unshared_inner, TD], [unshared_inner * TD, TD, 1], dtype=io_dtype
    )
    ubt = pl.make_tensor(unshared_block_table, [u_batch], [1], dtype=pl.DT_INT32)
    skv_t = pl.make_tensor(shared_kv_lens, [u_batch], [1], dtype=pl.DT_INT32)
    ds_t = pl.make_tensor(decode_step, [1], [1], dtype=pl.DT_INT32)
    sbt = pl.make_tensor(shared_block_table, [u_batch * sbt_stride], [1], dtype=pl.DT_INT32)

    # ========== Single-workspace layout: [o_s, m_s, l_s, o_u, m_u, l_u] ==========
    # o_*: fp16 [T, Hq, D]; m_*/l_*: fp32 [T, Hq, ML_W] (fp32 = 4B).  All offsets in
    # bytes on the UINT8 workspace pointer (host tiling must size the buffer to
    # lu_off + flat_rows * ML_W * 4).
    os_off = 0
    ms_off = os_off + flat_rows * TD * 2
    ls_off = ms_off + flat_rows * ML_W * 4
    ou_off = ls_off + flat_rows * ML_W * 4
    mu_off = ou_off + flat_rows * TD * 2
    lu_off = mu_off + flat_rows * ML_W * 4
    o_s = pl.make_tensor(pl.addptr(workspace, os_off), [num_tokens, n_dim, TD], [n_dim * TD, TD, 1], dtype=io_dtype)
    m_s = pl.make_tensor(
        pl.addptr(workspace, ms_off), [num_tokens, n_dim, ML_W], [n_dim * ML_W, ML_W, 1], dtype=pl.DT_FP32
    )
    l_s = pl.make_tensor(
        pl.addptr(workspace, ls_off), [num_tokens, n_dim, ML_W], [n_dim * ML_W, ML_W, 1], dtype=pl.DT_FP32
    )
    o_u = pl.make_tensor(pl.addptr(workspace, ou_off), [num_tokens, n_dim, TD], [n_dim * TD, TD, 1], dtype=io_dtype)
    m_u = pl.make_tensor(
        pl.addptr(workspace, mu_off), [num_tokens, n_dim, ML_W], [n_dim * ML_W, ML_W, 1], dtype=pl.DT_FP32
    )
    l_u = pl.make_tensor(
        pl.addptr(workspace, lu_off), [num_tokens, n_dim, ML_W], [n_dim * ML_W, ML_W, 1], dtype=pl.DT_FP32
    )
    # Flat views over the same offsets (combine reads them as contiguous flat-row blocks).
    o_s_flat = pl.make_tensor(pl.addptr(workspace, os_off), [flat_rows, 1, TD], [TD, TD, 1], dtype=io_dtype)
    m_s_flat = pl.make_tensor(pl.addptr(workspace, ms_off), [flat_rows, 1, ML_W], [ML_W, ML_W, 1], dtype=pl.DT_FP32)
    l_s_flat = pl.make_tensor(pl.addptr(workspace, ls_off), [flat_rows, 1, ML_W], [ML_W, ML_W, 1], dtype=pl.DT_FP32)
    o_u_flat = pl.make_tensor(pl.addptr(workspace, ou_off), [flat_rows, 1, TD], [TD, TD, 1], dtype=io_dtype)
    m_u_flat = pl.make_tensor(pl.addptr(workspace, mu_off), [flat_rows, 1, ML_W], [ML_W, ML_W, 1], dtype=pl.DT_FP32)
    l_u_flat = pl.make_tensor(pl.addptr(workspace, lu_off), [flat_rows, 1, ML_W], [ML_W, ML_W, 1], dtype=pl.DT_FP32)
    o = pl.make_tensor(attn_out, [num_tokens, n_dim, TD], [n_dim * TD, TD, 1], dtype=io_dtype)
    o_flat = pl.make_tensor(attn_out, [flat_rows, 1, TD], [TD, TD, 1], dtype=io_dtype)

    # decode_step: EXTERNAL input [1], the number of valid tokens along the
    # unshared maxDs axis for this call (in {1,2,3} for this port's scope).
    u_ds = pl.getval(ds_t, 0)

    # ========== Cross-core shared buffers (UBNBuffer for double-buffer) ==========
    # P MAT - Vector insert, Cube PV read
    p_mat_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS, TKV], dtype=io_dtype, target_memory=pl.MemorySpace.Mat),
        addrs=MA2_P,
        mutex_ids=[14, 15, 16],
    )

    # qk_vec UB - Cube store from ACC, Vector softmax (double-buffer for FIFO)
    qk_vec_db = pl.make_tile_group(
        type=pl.TileType(
            shape=[TS_HALF, TKV],
            dtype=pl.DT_FP32,
            target_memory=pl.MemorySpace.Vec,
            valid_shape=[-1, -1],
            compact=1,
            pad=pl.TilePad.min,
        ),
        addrs=VA0,
        mutex_ids=[17, 18],
    )

    # pv_vec UB - Cube store from ACC, Vector GU (double-buffer for FIFO)
    pv_vec_db = pl.make_tile_group(
        type=pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
        addrs=VA8,
        mutex_ids=[19, 20],
    )

    with pl.section_cube():
        # Cube-only buffers (independent buf_id space: 0-11)
        q_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA0_Q,
            mutex_ids=[0, 1],
        )
        k_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Mat,
                layout=pl.ZN,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA1_K,
            mutex_ids=[2, 3],
        )
        v_l1_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Mat,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=MA3_V,
            mutex_ids=[4, 5],
        )

        left_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Left,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TD, TKV],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Right,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 65536, 131072, 196608],
            mutex_ids=[10, 11, 12, 13],
        )
        left_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TKV],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Left,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[6, 7],
        )
        right_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TKV, TD],
                dtype=io_dtype,
                target_memory=pl.MemorySpace.Right,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 32768],
            mutex_ids=[8, 9],
        )
        acc_db2 = pl.make_tile_group(
            type=pl.TileType(
                shape=[TS, TD],
                dtype=pl.DT_FP32,
                target_memory=pl.MemorySpace.Acc,
                valid_shape=[-1, -1],
                compact=1,
            ),
            addrs=[0, 65536, 131072, 196608],
            mutex_ids=[10, 11, 12, 13],
        )

        if core_id < shared_core_num:
            _cl = core_id
            # SHARED: 1 group/batch, sq = beam (beam-folded), kv = shared_kv_lens[b].
            # Cumulative offsets are derived here from shared_kv_lens (no external asq/askv).
            # Per-core work ranges are derived IN-KERNEL from the work count and the core
            # split (mirror of the host _split_work_ranges formula) -- no wr arrays.
            _batch = u_batch
            _k = shared_k
            _v = shared_v
            s_tiles = (u_beam + TS - 1) // TS  # shared Q-tiles per (batch, head)
            shared_work = u_batch * n_dim * s_tiles
            per_s = (shared_work + shared_core_num - 1) // shared_core_num
            work_start = pl.min(_cl * per_s, shared_work)
            work_end = pl.min((_cl + 1) * per_s, shared_work)
            task_id = 0
            b_idx = 0
            s1_size_acc = 0
            s2_size_acc = 0
            actual_s1 = 0
            actual_s2 = 0
            s1_o_acc = 0
            ctx_arr = pl.struct_array(
                4,
                "CubeCtx",
                n_idx=0,
                kv_n_idx=0,
                qi=0,
                ki=0,
                task_id=0,
                s1SizeAcc=0,
                s2SizeAcc=0,
                s1_size=0,
                s2_size=0,
                pb=0,
                pm=0,
                tok=0,
                h0=0,
                qrow=0,
                b_idx=0,
            )
            # calc start
            s1o_size = 0  # tmp-val
            for idx in pl.range(_batch):
                actual_s1 = u_beam
                actual_s2 = pl.getval(skv_t, idx)
                s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
                if work_start >= s1o_size:
                    s1_o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            for work_id in pl.range(work_start, work_end):
                for _ in pl.range(b_idx, _batch):
                    actual_s1 = u_beam
                    actual_s2 = pl.getval(skv_t, b_idx)
                    s1o_size = s1_o_acc + (actual_s1 + TS - 1) // TS * n_dim
                    if work_id >= s1o_size:
                        s1_o_acc = s1o_size
                        s1_size_acc = s1_size_acc + actual_s1
                        s2_size_acc = s2_size_acc + actual_s2
                        b_idx = b_idx + 1
                        continue
                    break
                cur_b_s1o = (actual_s1 + TS - 1) // TS
                s1o_size = work_id - s1_o_acc
                n_idx = s1o_size // cur_b_s1o
                s1_idx = s1o_size % cur_b_s1o
                s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

                sq_off = s1_size_acc + s1_idx * TS
                cur_q_slot = q_l1_db.next()
                # full (non-causal) attention: iterate all KV tiles
                kv_loop = (actual_s2 + TKV - 1) // TKV
                kv_end = actual_s2
                drain = 0
                if work_id == work_end - 1:
                    drain = 2
                for ki in pl.range(0, kv_loop + drain):
                    if ki < kv_loop:
                        # Save current context
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.task_id = task_id
                        ctx_curr.n_idx = n_idx
                        ctx_curr.kv_n_idx = n_idx // group
                        ctx_curr.ki = ki
                        ctx_curr.s1SizeAcc = s1_size_acc
                        ctx_curr.s2SizeAcc = s2_size_acc
                        ctx_curr.s1_size = s1_size
                        ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                        ctx_curr.b_idx = b_idx

                        # ========== compute_qk (current step) ==========
                        compute_qk(
                            ctx_curr,
                            ki,
                            sq_off,
                            q,
                            _k,
                            cur_q_slot,
                            k_l1_db,
                            left_db,
                            right_db,
                            acc_db,
                            qk_vec_db,
                            task_id,
                            left_db2,
                            right_db2,
                            acc_db2,
                            sbt,
                            sbt_stride,
                        )

                    # ========== compute_pv (delayed 1 step: uses ctx from task_id-1) ==========
                    if task_id > 1:
                        ctx_pre2 = ctx_arr[(task_id + 2) % 4]
                        compute_pv(
                            ctx_pre2,
                            v_l1_db,
                            p_mat_db,
                            left_db2,
                            right_db2,
                            acc_db2,
                            pv_vec_db,
                            _v,
                            left_db,
                            right_db,
                            acc_db,
                            sbt,
                            sbt_stride,
                        )
                    task_id = task_id + 1
        else:
            _cl = core_id - shared_core_num
            _k = unshared_k
            _v = unshared_v
            # MERGED-GROUP folding: one work item = G groups (a group = one
            # (beam token, kv-head) pair), merged BOTH across kv-heads and across beam
            # tokens.  Their Q rows stack into M (= G*group) and their KV concatenates
            # along N (= G*maxDs); only the G diagonal blocks are meaningful, which the
            # in-kernel mask (gen_umask_vf) enforces in vec1.  This trades redundant MACs (efficiency
            # 1/G) -- free, since M=group alone leaves the cube almost idle -- for G x
            # fewer work items, i.e. G x fewer cross-core handshakes and KV DMAs.
            # One work item = ONE merged block of up to g_merge groups, all inside a
            # single batch.  Item id -> (batch, block); no TND scan needed because the
            # per-item shape is a closed form of the item index.
            u_ipb = (u_gpb + g_merge - 1) // g_merge  # merged items per batch
            _batch = u_batch * u_ipb
            unshared_work = _batch
            u_cores = total_cores - shared_core_num
            per_u = (unshared_work + u_cores - 1) // u_cores
            work_start = pl.min(_cl * per_u, unshared_work)
            work_end = pl.min((_cl + 1) * per_u, unshared_work)
            task_id = 0
            ctx_arr = pl.struct_array(
                4,
                "CubeCtx",
                n_idx=0,
                kv_n_idx=0,
                qi=0,
                ki=0,
                task_id=0,
                s1SizeAcc=0,
                s2SizeAcc=0,
                s1_size=0,
                s2_size=0,
                pb=0,
                pm=0,
                tok=0,
                h0=0,
                qrow=0,
                b_idx=0,
            )
            for work_id in pl.range(work_start, work_end):
                bb = work_id // u_ipb
                blk = work_id % u_ipb
                gid0 = blk * g_merge  # first group, batch-local
                g_eff = pl.min(g_merge, u_gpb - gid0)  # tail block may be short
                s1_size = g_eff * group
                actual_s2 = g_eff * u_maxds

                cur_q_slot = q_l1_db.next()
                # full (non-causal) attention: iterate all KV tiles
                kv_loop = (actual_s2 + TKV - 1) // TKV
                kv_end = actual_s2
                drain = 0
                if work_id == work_end - 1:
                    drain = 2
                for ki in pl.range(0, kv_loop + drain):
                    if ki < kv_loop:
                        # Save current context
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.task_id = task_id
                        ctx_curr.ki = ki
                        ctx_curr.s1_size = s1_size
                        ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                        # Group g (batch-local) owns q_flat rows [g*group, +group) because
                        # Hq == Hkv*group, so the merged block's Q is the contiguous run
                        # starting at the global group index * group.
                        ctx_curr.qrow = (bb * u_gpb + gid0) * group
                        ctx_curr.tok = ctx_curr.qrow  # output uses the same flat row
                        # K/V: physical-batch remap (slot table) or direct logical batch,
                        # then the merged run's start row in the flattened
                        # (beam, Hkv, maxDs) axis.
                        if PagedUnshared == 1:  # noqa: F821
                            ctx_curr.pb = pl.getval(ubt, bb)
                        else:
                            ctx_curr.pb = bb
                        ctx_curr.pm = gid0 * u_maxds

                        # ========== compute_qk_ug (current step, gather + group-fold) ==========
                        compute_qk_ug(
                            ctx_curr,
                            ki,
                            q_flat,
                            _k,
                            cur_q_slot,
                            k_l1_db,
                            left_db,
                            right_db,
                            acc_db,
                            qk_vec_db,
                            task_id,
                            left_db2,
                            right_db2,
                            acc_db2,
                        )

                    # ========== compute_pv_ug (delayed 1 step: uses ctx from task_id-1) ==========
                    if task_id > 1:
                        ctx_pre2 = ctx_arr[(task_id + 2) % 4]
                        compute_pv_ug(
                            ctx_pre2,
                            v_l1_db,
                            p_mat_db,
                            left_db2,
                            right_db2,
                            acc_db2,
                            pv_vec_db,
                            _v,
                            left_db,
                            right_db,
                            acc_db,
                        )
                    task_id = task_id + 1

    with pl.section_vector():
        tile_nz_g = pl.make_tile_group(
            type=pl.TileType(
                shape=[65, 128], dtype=io_dtype, target_memory=pl.MemorySpace.Vec, valid_shape=[-1, -1], layout=pl.NZ
            ),
            addrs=VA1,
            mutex_ids=[0, 1],
        )

        running_o = pl.make_tile(
            pl.TileType(shape=[TS_HALF, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec), addr=VA7, size=VB4
        )

        o_f16_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=VA9,
            mutex_ids=[2],
        )
        o_f16 = o_f16_g.next()
        # dedicated 64-wide fp32 tiles for m and l stores (VA10=freed mask, VA13=free top).
        # tile_group with a mutex_id (like o_f16_g) so auto_mutex tracks the cross-task
        # WAR (next task's bcast waits for this task's store) -> no race.
        m_tile_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA10,
            mutex_ids=[3],
        )
        m_tile = m_tile_g.next()
        l_tile_g = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=VA13,
            mutex_ids=[4],
        )
        l_tile = l_tile_g.next()
        # merged-group mask (unshared path only); constant across work items
        umask_db = pl.make_tile_group(
            type=pl.TileType(shape=[TS_HALF, TKV], dtype=pl.DT_UINT8, target_memory=pl.MemorySpace.Vec),
            addrs=VA_UMASK,
            mutex_ids=[5],
        )
        # (w / we) window tables, [TS, 8] int32 BUILT IN-KERNEL once per launch
        # (pl.setval writes UB directly -- no GM, no DCCI).  Each table is
        # written once (S pipe) then only read (V pipe BRC), so no cross-item
        # WAR exists; bare make_tile (no mutex) suffices, gated by an explicit
        # S->V sync.  128 rows cover every absolute item row (s1_size <= TS).
        w_tbl_tile = pl.make_tile(
            pl.TileType(shape=[TS, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec), addr=VA_WTBL, size=VB_WTBL
        )
        we_tbl_tile = pl.make_tile(
            pl.TileType(shape=[TS, 8], dtype=pl.DT_INT32, target_memory=pl.MemorySpace.Vec),
            addr=VA_WTBL + VB_WTBL,
            size=VB_WTBL,
        )

        # Double-buffered global state (per Q tile) -use tile tuples for dynamic
        # indexing by q_count % 2, since StructArray ctx references need runtime index.
        red_type = pl.TileType(shape=[TS_HALF, 1], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec, layout=pl.DN)
        gmax_0 = pl.make_tile(red_type, addr=VA_GMAX0, size=VB_RED)
        gmax_1 = pl.make_tile(red_type, addr=VA_GMAX1, size=VB_RED)
        gmax_2 = pl.make_tile(red_type, addr=VA_GMAX2, size=VB_RED)
        global_max = (gmax_0, gmax_1, gmax_2)

        gsum_0 = pl.make_tile(red_type, addr=VA_GSUM0, size=VB_RED)
        gsum_1 = pl.make_tile(red_type, addr=VA_GSUM1, size=VB_RED)
        gsum_2 = pl.make_tile(red_type, addr=VA_GSUM2, size=VB_RED)
        global_sum = (gsum_0, gsum_1, gsum_2)

        tmp_max = pl.make_tile(red_type, addr=VA11, size=VB_RED)
        tmp_sum = pl.make_tile(red_type, addr=VA12, size=VB_RED)

        # FIFO exp_corr -use NBuffer with current() auto-rotate
        exp_max0 = pl.make_tile(red_type, addr=VA_EXPMAX0, size=VB_RED)
        exp_max1 = pl.make_tile(red_type, addr=VA_EXPMAX1, size=VB_RED)
        exp_max2 = pl.make_tile(red_type, addr=VA_EXPMAX2, size=VB_RED)
        exp_corr_db = (exp_max0, exp_max1, exp_max2)

        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=QK_READY_BARKWARD_IDS[1])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[0])
        pl.system.set_cross_core(pipe=pl.PipeType.V, event_id=PV_READY_BARKWARD_IDS[1])

        if core_id < shared_core_num:
            _cl = core_id
            # SHARED (vec): mirror of shared cube work-distribution, derived from
            # shared_kv_lens (sq = beam, kv = shared_kv_lens[b]); no external asq/askv.
            # Per-core work ranges derived in-kernel (same formula as the cube branch).
            _batch = u_batch
            _ou = o_s
            _mo = m_s
            _lo = l_s
            s_tiles = (u_beam + TS - 1) // TS  # mirror of the cube branch
            shared_work = u_batch * n_dim * s_tiles
            per_s = (shared_work + shared_core_num - 1) // shared_core_num
            work_start = pl.min(_cl * per_s, shared_work)
            work_end = pl.min((_cl + 1) * per_s, shared_work)
            sub_id = pl.get_subblock_idx()

            task_id = 0
            q_count = 0
            b_idx = 0
            s1_size_acc = 0
            s2_size_acc = 0
            actual_s1 = 0
            actual_s2 = 0
            s1_o_acc = 0
            segment_idx = 0

            # StructArray(3) for pipeline context tracking (same as original)
            ctx_arr = pl.struct_array(
                4,
                "VecCtx",
                b_idx=0,
                n_idx=0,
                s1_idx=0,
                q_off=0,
                ki=0,
                q_count=0,
                sub_id=0,
                task_id=0,
                kv_loop=0,
                half_s1=0,
                first_s1=0,
                s1_size=0,
                s2_size=0,
                segment_acc=0,
                segment_idx=0,
                cross_segment=0,
                tok=0,
                h0=0,
            )

            # calc start
            s1o_size = 0  # tmp-val
            for idx in pl.range(_batch):
                actual_s1 = u_beam
                actual_s2 = pl.getval(skv_t, idx)
                s1o_size = s1o_size + (actual_s1 + TS - 1) // TS * n_dim
                if work_start >= s1o_size:
                    s1_o_acc = s1o_size
                    s1_size_acc = s1_size_acc + actual_s1
                    s2_size_acc = s2_size_acc + actual_s2
                    b_idx = b_idx + 1
                    continue
                break
            for work_id in pl.range(work_start, work_end):
                for _ in pl.range(b_idx, _batch):
                    actual_s1 = u_beam
                    actual_s2 = pl.getval(skv_t, b_idx)
                    s1o_size = s1_o_acc + (actual_s1 + TS - 1) // TS * n_dim
                    if work_id >= s1o_size:
                        s1_o_acc = s1o_size
                        s1_size_acc = s1_size_acc + actual_s1
                        s2_size_acc = s2_size_acc + actual_s2
                        b_idx = b_idx + 1
                        continue
                    break
                cur_b_s1o = (actual_s1 + TS - 1) // TS
                s1o_size = work_id - s1_o_acc
                n_idx = s1o_size // cur_b_s1o
                s1_idx = s1o_size % cur_b_s1o
                s1_size = pl.min(TS, actual_s1 - s1_idx * TS)

                sq_acc = b_idx * u_beam
                q_off = sq_acc + s1_idx * TS
                # full (non-causal) attention: iterate all KV tiles
                kv_loop = (actual_s2 + TKV - 1) // TKV
                kv_end = actual_s2

                # Pipeline flush folded into the ki loop: on the LAST work item extend by
                # the max consumer lag (compute_gu = 3) with the producer gated off.
                # compute_p (lag 1) drains once -> gate to ki <= kv_loop; compute_gu
                # (lag 3) drains on all extra steps.  No separate epilogue.
                drain = 0
                if work_id == work_end - 1:
                    drain = 3
                for ki in pl.range(0, kv_loop + drain):
                    if ki < kv_loop:
                        # ===== producer: build this task's context =====
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.b_idx = b_idx
                        ctx_curr.n_idx = n_idx
                        ctx_curr.s1_idx = s1_idx
                        ctx_curr.q_off = q_off
                        ctx_curr.task_id = task_id
                        ctx_curr.ki = ki
                        ctx_curr.kv_loop = kv_loop
                        ctx_curr.q_count = q_count
                        ctx_curr.s1_size = s1_size
                        ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                        half_s1 = (ctx_curr.s1_size + 1) // 2
                        first_s1 = half_s1
                        if sub_id == 1:
                            half_s1 = ctx_curr.s1_size - half_s1
                        ctx_curr.first_s1 = first_s1
                        ctx_curr.half_s1 = half_s1

                    # ===== compute_p (consumer, lag 1: real steps + 1st drain step) =====
                    if task_id > 0:
                        if ki <= kv_loop:
                            ctx_p = ctx_arr[(task_id + 3) % 4]
                            compute_p(
                                ctx_p,
                                sub_id,
                                qk_vec_db,
                                tile_nz_g,
                                tmp_max,
                                tmp_sum,
                                global_max,
                                global_sum,
                                exp_corr_db,
                                p_mat_db,
                                scale,
                            )

                    # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                    if task_id > 2:
                        ctx_gu = ctx_arr[(task_id + 1) % 4]
                        compute_gu_partial(
                            ctx_gu,
                            pv_vec_db,
                            exp_corr_db,
                            global_sum,
                            global_max,
                            running_o,
                            o_f16,
                            m_tile,
                            l_tile,
                            _ou,
                            _mo,
                            _lo,
                        )

                    task_id = task_id + 1
                q_count = q_count + 1
        else:
            _cl = core_id - shared_core_num
            _ou = o_u_flat
            _mo = m_u_flat
            _lo = l_u_flat
            # MERGED-GROUP folding (mirror of the cube unshared branch).
            u_ipb = (u_gpb + g_merge - 1) // g_merge  # mirror of the cube branch
            _batch = u_batch * u_ipb
            unshared_work = _batch
            u_cores = total_cores - shared_core_num
            per_u = (unshared_work + u_cores - 1) // u_cores
            work_start = pl.min(_cl * per_u, unshared_work)
            work_end = pl.min((_cl + 1) * per_u, unshared_work)
            sub_id = pl.get_subblock_idx()

            # Build the (w, we) mask window tables ONCE per launch, at kernel
            # level (i64 div is fine here; it is NOT inside a vector function).
            # Both tables are written DIRECTLY into UB tiles by pl.setval (S-pipe
            # scalar store to UB -- no GM round-trip, no DCCI / cache-coherence
            # hazard, which was the failure mode when writing GM tensors).  The
            # tables are written once and only read afterwards, so no mutex is
            # needed; an explicit S->V sync pair gates the V-pipe BRC reads.
            for t_r in pl.range(0, TS):
                t_w = (t_r // group) * u_maxds
                pl.setval(w_tbl_tile, t_r * 8, t_w)
                pl.setval(we_tbl_tile, t_r * 8, t_w + u_ds)
            pl.system.sync_src(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.V, event_id=6)
            pl.system.sync_dst(set_pipe=pl.PipeType.S, wait_pipe=pl.PipeType.V, event_id=6)

            task_id = 0
            q_count = 0

            # StructArray(3) for pipeline context tracking (same as original)
            ctx_arr = pl.struct_array(
                4,
                "VecCtx",
                b_idx=0,
                n_idx=0,
                s1_idx=0,
                q_off=0,
                ki=0,
                q_count=0,
                sub_id=0,
                task_id=0,
                kv_loop=0,
                half_s1=0,
                first_s1=0,
                s1_size=0,
                s2_size=0,
                segment_acc=0,
                segment_idx=0,
                cross_segment=0,
                tok=0,
                h0=0,
            )

            for work_id in pl.range(work_start, work_end):
                bb = work_id // u_ipb
                blk = work_id % u_ipb
                gid0 = blk * g_merge
                g_eff = pl.min(g_merge, u_gpb - gid0)
                s1_size = g_eff * group
                actual_s2 = g_eff * u_maxds
                q_off = (bb * u_gpb + gid0) * group
                # full (non-causal) attention: iterate all KV tiles
                kv_loop = (actual_s2 + TKV - 1) // TKV
                kv_end = actual_s2

                # Pipeline flush folded into the ki loop: on the LAST work item extend by
                # the max consumer lag (compute_gu = 3) with the producer gated off.
                # compute_p (lag 1) drains once -> gate to ki <= kv_loop; compute_gu
                # (lag 3) drains on all extra steps.  No separate epilogue.
                drain = 0
                if work_id == work_end - 1:
                    drain = 3
                for ki in pl.range(0, kv_loop + drain):
                    if ki < kv_loop:
                        # ===== producer: build this task's context =====
                        ctx_curr = ctx_arr[task_id % 4]
                        ctx_curr.q_off = q_off
                        ctx_curr.task_id = task_id
                        ctx_curr.ki = ki
                        ctx_curr.kv_loop = kv_loop
                        ctx_curr.q_count = q_count
                        ctx_curr.s1_size = s1_size
                        ctx_curr.s2_size = pl.min(TKV, kv_end - ki * TKV)
                        # merged block -> a contiguous run of flat output rows
                        ctx_curr.tok = q_off
                        half_s1 = (ctx_curr.s1_size + 1) // 2
                        first_s1 = half_s1
                        if sub_id == 1:
                            half_s1 = ctx_curr.s1_size - half_s1
                        ctx_curr.first_s1 = first_s1
                        ctx_curr.half_s1 = half_s1

                    # ===== compute_p (consumer, lag 1: real steps + 1st drain step) =====
                    if task_id > 0:
                        if ki <= kv_loop:
                            ctx_p = ctx_arr[(task_id + 3) % 4]
                            compute_p_ug(
                                ctx_p,
                                sub_id,
                                qk_vec_db,
                                tile_nz_g,
                                global_max,
                                global_sum,
                                p_mat_db,
                                umask_db,
                                w_tbl_tile,
                                we_tbl_tile,
                                scale,
                            )

                    # ===== compute_gu (consumer, lag 3: every step incl. all drains) =====
                    if task_id > 2:
                        ctx_gu = ctx_arr[(task_id + 1) % 4]
                        compute_gu_partial_ug(
                            ctx_gu,
                            pv_vec_db,
                            exp_corr_db,
                            global_sum,
                            global_max,
                            running_o,
                            o_f16,
                            m_tile,
                            l_tile,
                            _ou,
                            _mo,
                            _lo,
                        )

                    task_id = task_id + 1
                q_count = q_count + 1

    pl.system.sync_all(core_type=pl.SyncCoreType.MIX)

    # =================== COMBINE SECTION (after global barrier) ===================
    with pl.section_vector():
        num_rows = flat_rows
        num_cores = total_cores
        # Partition by VECTOR SUB-CORE (section_vector runs on both sub-cores of a core,
        # so a core-granular split would have vec0/vec1 redo each other's rows), and give
        # each sub-core a CONTIGUOUS span rather than a grid-stride: row rr maps to flat
        # row rr of [T*Hq, D], so a contiguous span is one contiguous GM region and a
        # block of CMB_R rows loads in a SINGLE transfer instead of CMB_R of them.
        n_sub = pl.get_subblock_num()
        vec_id = pl.get_block_idx()
        num_vecs = num_cores * n_sub
        chunk = (num_rows + num_vecs - 1) // num_vecs
        lo = vec_id * chunk
        hi = pl.min(lo + chunk, num_rows)
        # Loaded/stored tiles are DOUBLE buffered so block k+1's loads (MTE2) overlap
        # block k's compute (V) and block k-1's store (MTE3); auto_mutex derives the
        # required cross-stage waits from the mutex ids, so no manual sync_src/dst pairs.
        # Intermediates touched only by V stay single-buffered (V is in-order anyway).
        RB = CMB_R
        os_h_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=[0x00000, 0x02000],
            mutex_ids=[0, 1],
        )
        ou_h_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=[0x04000, 0x06000],
            mutex_ids=[2, 3],
        )
        ms_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x08000, 0x08400],
            mutex_ids=[4, 5],
        )
        mu_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x08800, 0x08C00],
            mutex_ids=[6, 7],
        )
        ls_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x09000, 0x09400],
            mutex_ids=[8, 9],
        )
        lu_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, ML_W], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addrs=[0x09800, 0x09C00],
            mutex_ids=[10, 11],
        )
        oo16_g = pl.make_tile_group(
            type=pl.TileType(shape=[RB, TD], dtype=io_dtype, target_memory=pl.MemorySpace.Vec),
            addrs=[0x0A000, 0x0C000],
            mutex_ids=[12, 13],
        )
        os_f = pl.make_tile(
            pl.TileType(shape=[RB, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x10000,
            size=RB * TD * 4,
        )
        ou_f = pl.make_tile(
            pl.TileType(shape=[RB, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x14000,
            size=RB * TD * 4,
        )
        oo32 = pl.make_tile(
            pl.TileType(shape=[RB, TD], dtype=pl.DT_FP32, target_memory=pl.MemorySpace.Vec),
            addr=0x18000,
            size=RB * TD * 4,
        )
        for blk in pl.range(lo, hi, RB):
            rows = pl.min(RB, hi - blk)
            os_h = os_h_g.next()
            ou_h = ou_h_g.next()
            ms_c = ms_g.next()
            mu_c = mu_g.next()
            ls_c = ls_g.next()
            lu_c = lu_g.next()
            oo16 = oo16_g.next()
            pl.set_validshape(os_h, [rows, TD])
            pl.set_validshape(ou_h, [rows, TD])
            pl.load(os_h, o_s_flat, [blk, 0, 0], order=[0, 2])
            pl.load(ou_h, o_u_flat, [blk, 0, 0], order=[0, 2])
            pl.set_validshape(ms_c, [rows, ML_W])
            pl.set_validshape(mu_c, [rows, ML_W])
            pl.set_validshape(ls_c, [rows, ML_W])
            pl.set_validshape(lu_c, [rows, ML_W])
            pl.load(ms_c, m_s_flat, [blk, 0, 0], order=[0, 2])
            pl.load(mu_c, m_u_flat, [blk, 0, 0], order=[0, 2])
            pl.load(ls_c, l_s_flat, [blk, 0, 0], order=[0, 2])
            pl.load(lu_c, l_u_flat, [blk, 0, 0], order=[0, 2])
            pl.set_validshape(os_f, [rows, TD])
            pl.set_validshape(ou_f, [rows, TD])
            pl.cast(os_f, os_h, mode=pl.RoundMode.CAST_NONE)
            pl.cast(ou_f, ou_h, mode=pl.RoundMode.CAST_NONE)
            pl.set_validshape(oo32, [rows, TD])
            combine_blk_vf(oo32, os_f, ou_f, ms_c, mu_c, ls_c, lu_c, rows)
            pl.set_validshape(oo16, [rows, TD])
            pl.cast(oo16, oo32, mode=pl.RoundMode.CAST_ROUND)
            pl.store(o_flat, oo16, [blk, 0, 0], order=[0, 2])


@pl.vector_function
def combine_blk_vf(oout, os, ou, ms_t, mu_t, ls_t, lu_t, rows):
    """Flash-lse combine of two partials over a BLOCK of rows.
    os/ou/oout are [rows, TD] fp32; ms/mu/ls/lu are [rows, ML_W] (lane 0 used)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    for i in pl.range(rows):
        ms = vf.load_align(ms_t, i * ML_W, dist=pl.LoadDist.BRC_B32)
        mu = vf.load_align(mu_t, i * ML_W, dist=pl.LoadDist.BRC_B32)
        ls = vf.load_align(ls_t, i * ML_W, dist=pl.LoadDist.BRC_B32)
        lu = vf.load_align(lu_t, i * ML_W, dist=pl.LoadDist.BRC_B32)
        vm = vf.max(ms, mu, preg)
        cs = vf.exp_sub(ms, vm, preg)
        cu = vf.exp_sub(mu, vm, preg)
        lcs = vf.mul(ls, cs, preg)
        lcu = vf.mul(lu, cu, preg)
        lval = vf.add(lcs, lcu, preg)
        for j in pl.range(D_LOOPS):
            off = i * TD + j * FLOAT_REP_SIZE
            o1 = vf.load_align(os, off)
            o2 = vf.load_align(ou, off)
            n1 = vf.mul(o1, cs, preg)
            n2 = vf.mul(o2, cu, preg)
            num = vf.add(n1, n2, preg)
            outv = vf.div(num, lval, preg)
            vf.store_align(oout + off, outv, preg)


# ================================================================
#  Python host: external entry point (tiling + launch, NO return value)
#
#  x_attention() computes the host tiling (mirror of test_xattention_beam.py's
#  former xattn_op_prepare and of the (future) C++ tiling in
#  op_host/x_attention_tiling.cpp), then launches x_attention_v2 and writes the
#  result INTO the caller-provided attn_out tensor.
#
#  Device is NOT passed in: the caller sets the current device via
#  torch.npu.set_device(...) beforehand; this entry uses the current device.
# ================================================================
def _unshared_merge_factor(beam, hkv, group, max_decode_step):
    """G: how many groups (a group = one (beam token, kv-head) pair) one
    unshared work item merges.  Mirrors the kernel's in-kernel g_merge; the
    kernel-side constraints maxDecodeStep <= TKV and group <= TS are enforced
    by the tiling validation in x_attention() below."""
    return max(1, min(TS // max_decode_step, TS // group, beam * hkv))


def x_attention_v2(
    query: torch.Tensor,
    shared_k: torch.Tensor,
    shared_v: torch.Tensor,
    unshared_k: torch.Tensor,
    unshared_v: torch.Tensor,
    shared_kv_lens: torch.Tensor,
    decode_step: torch.Tensor,
    attn_out: torch.Tensor,
    unshared_block_table: torch.Tensor | None = None,
    shared_block_table: torch.Tensor | None = None,
    scale: float | None = None,
) -> None:
    """xLLM-style beam-decode attention, external entry point.

    Computes the host tiling, launches x_attention_v2 and writes the
    output into the CALLER-provided ``attn_out`` (no return value).

      query               [B*beam, Hq, D]                 fp16/bf16, NPU tensor
      shared_key/value    [sum(Lb), Hkv, D] contiguous, OR paged physical
                          block cache [numBlocks*128, Hkv, D] (when
                          shared_block_table is given). NPU tensors.
      unshared_key/value  [B, beam, Hkv, maxDecodeStep, D]  NPU tensors
      shared_kv_lens      [B] int32 NPU tensor: per-batch shared KV lengths
      decode_step         [1] int32 NPU tensor: valid unshared length ([1, maxDs])
      attn_out            pre-allocated NPU output [B*beam, Hq, D] (same shape
                          and dtype as query); written in place.
      unshared_block_table [B] int32 NPU tensor or None: slot gather (None =
                          direct logical-batch addressing, PagedUnshared=0).
      shared_block_table  [B, maxBlocks] int32 NPU tensor or None: paged shared
                          block table (None = contiguous layout, PagedShared=0).
      scale               optional scale_value; default 1/sqrt(D).

    ALL input tensors must already be on the CURRENT npu device: the entry
    performs no device transfer.  The caller sets the current device via
    torch.npu.set_device(...) beforehand and moves inputs there.

    Superset constraints enforced here (tiling-side validation): Hq % Hkv == 0,
    group = Hq/Hkv <= 128, maxDecodeStep <= 128, 1 <= decode_step <= maxDs,
    paged layout/capacity checks, attn_out shape/dtype == query.
    """
    import math  # noqa: E402

    from pypto_pro.runtime.platform import get_platform_info  # noqa: E402

    device = f"npu:{torch.npu.current_device()}"
    for name, t in [
        ("query", query),
        ("shared_k", shared_k),
        ("shared_v", shared_v),
        ("unshared_k", unshared_k),
        ("unshared_v", unshared_v),
        ("shared_kv_lens", shared_kv_lens),
        ("decode_step", decode_step),
        ("attn_out", attn_out),
    ]:
        assert t.is_npu, f"input '{name}' must be an NPU tensor (got {t.device})."
    assert shared_kv_lens.dtype == torch.int32, "shared_kv_lens must be int32."
    assert decode_step.dtype == torch.int32 and tuple(decode_step.shape) == (1,), (
        "decode_step must be an int32 tensor of shape [1]."
    )

    num_tokens, hq, d = query.shape
    batch = unshared_k.shape[0]
    beam = num_tokens // batch
    hkv = shared_k.shape[1]
    ds = int(decode_step.item())
    max_ds = unshared_k.shape[3]
    group = hq // hkv
    shared_lens = [int(x) for x in shared_kv_lens.cpu().tolist()]

    # ---------- tiling-side validation (constraints enforced here) ----------
    assert hq % hkv == 0, f"query heads {hq} must be divisible by kv heads {hkv}."
    assert 1 <= ds <= max_ds, f"decode_step must be in [1, maxDs={max_ds}], got {ds}."
    assert group <= 128, f"GQA group size {group} > 128 unsupported (requires Hq <= 128 * Hkv)."
    assert max_ds <= 128, (
        f"maxDecodeStep {max_ds} > 128 unsupported (kernel merge factor TKV//maxDs would divide by zero)."
    )
    assert tuple(attn_out.shape) == tuple(query.shape), (
        f"attn_out shape {tuple(attn_out.shape)} != query shape {tuple(query.shape)}."
    )
    assert attn_out.dtype == query.dtype, f"attn_out dtype {attn_out.dtype} != query dtype {query.dtype}."
    paged_shared = shared_block_table is not None
    paged_unshared = unshared_block_table is not None
    if paged_shared:
        assert shared_block_table.is_npu and shared_block_table.dtype == torch.int32, (
            "shared_block_table must be an int32 NPU tensor."
        )
        max_blocks = shared_block_table.shape[1]
        assert shared_block_table.shape[0] == batch, (
            f"shared_block_table rows {shared_block_table.shape[0]} != batch {batch}."
        )
        assert shared_k.shape[0] % TS == 0, (
            f"paged shared KV must be a multiple of block_size 128 [numBlocks*128, Hkv, D], got {shared_k.shape}."
        )
        assert max_blocks * TS >= max(shared_lens), (
            f"shared_block_table capacity {max_blocks * TS} < max shared len {max(shared_lens)}."
        )
    if paged_unshared:
        assert unshared_block_table.is_npu and unshared_block_table.dtype == torch.int32, (
            "unshared_block_table must be an int32 NPU tensor."
        )

    # ---------- host tiling (mirror of the C++ tiling) ----------
    # Only the core split is computed on the host; each core derives its own
    # [start, end) work range IN-KERNEL from (work, ncore) -- no wr arrays.
    shared_tiles_per_unit = max(1, (beam + TS - 1) // TS)
    g_merge = _unshared_merge_factor(beam, hkv, group, max_ds)
    items_per_batch = (beam * hkv + g_merge - 1) // g_merge
    max_prompt = max(shared_lens) if shared_lens else 128
    kv_loop = (max_prompt + TS - 1) // TS
    shared_work = batch * hq * shared_tiles_per_unit
    unshared_work = batch * items_per_batch
    total_cores = get_platform_info().core_num
    ratio = unshared_work / (shared_work * kv_loop + 0.001)
    unshared_cores = max(6, min(total_cores - 2, round(total_cores * ratio / (1 + ratio))))
    shared_cores = total_cores - unshared_cores
    s_cores = shared_cores
    u_cores = unshared_cores

    scale_value = scale if scale is not None else 1.0 / math.sqrt(d)
    tiling = OpTiling(
        batch=batch,
        beam=beam,
        hq=hq,
        hkv=hkv,
        shared_total=sum(shared_lens),
        u_maxds=max_ds,
        scale=scale_value,
        sbt_stride=0,
        shared_core_num=s_cores,
        total_cores=s_cores + u_cores,
    )
    if paged_shared:
        tiling = replace(tiling, sbt_stride=max_blocks, shared_total=shared_k.shape[0])

    # ---------- launch ----------
    flat_rows = num_tokens * hq
    ws_bytes = 2 * flat_rows * d * 2 + 4 * flat_rows * ML_W * 4
    workspace = torch.zeros(ws_bytes, dtype=torch.uint8, device=device)
    sbt_t = shared_block_table.reshape(-1) if paged_shared else torch.zeros(1, dtype=torch.int32, device=device)
    ubt_t = unshared_block_table if paged_unshared else torch.zeros(1, dtype=torch.int32, device=device)

    pl_dtype = pl.DT_FP16 if query.dtype == torch.float16 else pl.DT_BF16
    datatype = {
        "query": pl_dtype,
        "shared_key_block": pl_dtype,
        "shared_value_block": pl_dtype,
        "unshared_key_block": pl_dtype,
        "unshared_value_block": pl_dtype,
        "attn_out": pl_dtype,
    }
    x_attention_v2_kernel[
        None, tiling.total_cores, {"PagedShared": int(paged_shared), "PagedUnshared": int(paged_unshared)}, datatype
    ](
        query,
        shared_k,
        shared_v,
        unshared_k.contiguous(),
        unshared_v.contiguous(),
        ubt_t,
        shared_kv_lens,
        decode_step,
        sbt_t,
        attn_out,
        workspace,
        tiling,
    )
    torch.npu.synchronize()
