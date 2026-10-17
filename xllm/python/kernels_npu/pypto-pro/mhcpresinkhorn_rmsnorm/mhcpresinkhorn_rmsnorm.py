#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2024-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# You should have received a copy of the License along with this program. If not, see
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""mhcpresinkhorn_rmsnorm -- PyPTO-Pro implementation (v4-K: bf16 hc_fn + Cube/Vector overlap + single K-axis-split scheme).

Changes of v4 relative to v3 (realizing the avoidable items listed in the
"PyPTO-Pro framework-level overhead analysis report"):

  1. The hc_fn input becomes bf16 (the weight dtype conversion has been hoisted
     into the model side, the kernel no longer converts): the Cube runs the
     matmul directly as BF16xBF16->FP32, x no longer needs a cast to fp32, and
     the whole cast -> GM workspace -> L1 staging chain disappears (this avoids
     overhead item 1: the input is already bf16, so no AscendC-style TQue
     double buffering is needed to hide the cast either).
  2. The matmul is split along the K axis and load-balanced across all 28 cube
     cores:
       kstart = core*592 (27 full-block cores x 592 + a last core x 400 =
         16384), load-balanced over all 28 cube cores. K=16384 is sliced with a
         stride of 592 (kstart=core*592; 27 full-block cores x 592 + 1 last
         core x 400, both 16-aligned). Two tile shapes cover the whole load
         ([16,592]/[592,32] and [16,400]/[400,32], L0B totals 63.5KB), so there
         is no tail block and no zero padding; a single [16,K]x[K,32] matmul is
         issued, the 28 partials are stored in GM and Stage2 reduces them with
         pl.sum(dim=1).
         (Note: Mat tiles do not support fillpad (only Vec tiles do); the
          GM->L1 load in the cube section cannot use a bar barrier (both
          bar_mte1/2 generate PIPE_MTE1), so the zero-baseline scheme has a
          load race and was dropped in favour of the two tile shapes.)
  3. The matmul runs in parallel with RMSNorm: the Cube reads bf16 x straight
     from GM (it does not depend on any Vector-produced value) while the Vector
     computes ssq; each token group needs only 1 sync_all(MIX) (v3 needed 2).
     The Cube's mm(g+1) overlaps with the Vector's Stage2(g).
  4. VFs are unrolled per row and move 2xVL per load:
       ssq: a single DINTLV_B16 load_align moves 2*VL=256 bf16 (two rows of 128
            elements) and splits them into even/odd -> astype fp32 ->
            mul_add_dst FMA accumulation;
       y  : the HC=4 rows are explicitly unrolled with mul_add_dst FMA (a single
            multiply-add instruction, so the intermediate product is not
            truncated).
     Note: merging a pair of m rows into one load_align turns out to be
     infeasible after deriving the register lane geometry: after the bf16
     de-interleaving, the even/odd lane layout of the (m0,m1) row pair,
     [m0e|m1e]/[m0o|m1o], conflicts with the pre broadcast pattern, and the
     per-d summation would require adding across lanes (lane i with lane i+32),
     for which the VF has no primitive; so y keeps one load_align (128 bf16) +
     FMA per m row.
"""

import os

import torch
import torch_npu  # noqa: F401

import pypto_pro.language as pl
from pypto_pro.language import Vf as vf  # noqa: N813

# ======================================================================
# Contract constants (same as SPEC and golden; except the hc_fn dtype --
# the model side already converts it to bf16)
# ======================================================================
HC = 4
D = 4096
MIX_HC = (2 + HC) * HC       # 24
HC_D = HC * D                # 16384
SINKHORN_ITERS = 20
EPS = 1e-6
RMS_EPS = 1e-6               # eps of the fused RmsNorm (same as production RmsNorm)

# ======================================================================
# Multi-core / tiling constants
# ======================================================================
NB = 28                      # MIX logical blocks (28 AIC + 56 AIV, all cube cores used)
TG = 16                      # pass = token group (the cube M dim)
MIXC = 32                    # gating output 24 -> padded to 32
DC = 512                     # d-chunk width of y (tile width of x_sub/y_ub)
NDC = D // DC                # 8: number of d-chunks
SSQ_SLICES = 32              # number of ssq slices (both AIVs of blocks 0-15, 512 wide each)
SSQ_DK = HC_D // SSQ_SLICES  # 512

# ---- K-axis split (kstart=core*KA; full-block cores 592, last core 400) ----
# See the L1/L0 layout below for details

# ======================================================================
# UB address layout (vector side, ~36KB < 248KB; slots are spaced >=256B
# apart to prevent load_align from over-reading)
# ======================================================================
XSQ_UB = 0x00000             # [TG,SSQ_DK] bf16 16KB (ssq input)
SSQST_UB = 0x04000           # [TG,64] fp32 4KB
SSQTT_UB = 0x05000           # [64,TG] fp32 4KB
SLAB_UB = 0x06000            # [NB,MIXC] fp32 3.5KB (slab for reducing the 28 partials)
MIXR_UB = 0x06E00            # [1,MIXC] fp32 128B (a row of mixes)
SSQL_UB = 0x06F00            # [SSQ_SLICES,TG] fp32 2KB
SSQR_UB = 0x07700            # [1,TG] fp32 64B
SCALE3_UB = 0x07780          # [1,8] fp32 32B
SCALE24_UB = 0x07800         # [1,32] fp32 128B
PMUL24_UB = 0x07900          # [1,32] fp32 128B
EPS24_UB = 0x07A00           # [1,32] fp32 128B
BASE_UB = 0x07B00            # [1,32] fp32 128B
ZST_UB = 0x07C00             # [1,MIX_HC] fp32 96B (single-token zstage)
ZSD_UB = 0x07C80             # [MIX_HC,TG] fp32 1.5KB (after transpose)
CBT_UB = 0x08300             # [16,1] fp32 (sinkhorn comb output, 64B valid)
CBTT_UB = 0x08500            # [1,16] fp32 64B (after transpose)
PST_UB = 0x08540             # [HC,8] fp32 128B (post storage)
PSTT_UB = 0x08580            # [1,8] fp32 32B (after transpose)
PRE_UB = 0x085C0             # [1,8] fp32 32B (pre, for the y BRC)
XSUB_UB = 0x09000            # 2 slots [HC,DC] bf16 4KBx2 (x slices of y, double buffered)
YUB_UB = 0x0B000             # 2 slots [1,DC] bf16 1KBx2 (one y_norm chunk, double buffered)
YROW_UB = 0x0C000            # [1,D] bf16 8KB (staging of a whole y row: written by pass1 / read by pass2)
GAMMA_UB = 0x0E000           # [1,D] bf16 8KB (RmsNorm gamma, preloaded once per AIV)
RSTD_UB = 0x10000            # [1,64] fp32 256B (rstd broadcast, for the pass2 BRC)

# ---- L1/L0 (two tile shapes: full-block cores 592 / last core 400, full
# coverage with no tail block) ----
KA = 592                      # K slice of a full-block core (37x16); 27 cores x 592 + last core 400
KLAST = HC_D - (NB - 1) * KA  # K slice of the last core = 400 (25x16)
LHS_SRC_L1 = 0x00000          # [TG,KA] bf16 NZ 18.5KB
LHS_SRC2_L1 = 0x06000         # [TG,KLAST] bf16 NZ 12.5KB
RHS_SRC_L1 = 0x0A000          # [KA,MIXC] bf16 ZN 37KB
RHS_SRC2_L1 = 0x14000         # [KLAST,MIXC] bf16 ZN 25KB


# ======================================================================
# VF sub-functions
# ======================================================================

@pl.vector_function
def _ssq_vf(x_src, ssq_stage, tg):
    """x [tg,SSQ_DK] bf16 -> per-token sum of squares (fp32); row j is stored at ssq_stage+j*64.

    A single DINTLV_B16 load_align moves 2*VL=256 bf16 (two rows of 128
    elements) and splits them into two registers by even/odd index; after the
    astype expansion to fp32 they are accumulated with mul_add_dst FMA
    (acc = x*x + acc, a single multiply-add instruction).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    for j in pl.range(0, tg):
        acc = vf.full(0.0, preg, dtype=pl.DT_FP32)
        ev0, od0 = vf.load_align(x_src, j * SSQ_DK + 0,
                                 dist=pl.LoadDist.DINTLV_B16)
        e0 = vf.astype(ev0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e0, e0, preg)
        e1 = vf.astype(ev0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(e1, e1, preg)
        o0 = vf.astype(od0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(o0, o0, preg)
        o1 = vf.astype(od0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o1, o1, preg)
        ev1, od1 = vf.load_align(x_src, j * SSQ_DK + 256,
                                 dist=pl.LoadDist.DINTLV_B16)
        e2 = vf.astype(ev1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(e2, e2, preg)
        e3 = vf.astype(ev1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(e3, e3, preg)
        o2 = vf.astype(od1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
        acc = vf.mul_add_dst(o2, o2, preg)
        o3 = vf.astype(od1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
        acc = vf.mul_add_dst(o3, o3, preg)
        sval = vf.reduce_sum(acc, preg)
        vf.store_align(ssq_stage + j * 64, sval, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _build_aux_vf(scale3, scale24, pmul24, eps24):
    """Auxiliary vectors: scale24=[s0]*4+[s1]*4+[s2]*16; pmul24=[1]*4+[2]*4+[1]*16; eps24=[eps]*4+[0]*20."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg24 = vf.update_mask(24, dtype=pl.DT_FP32)
    sel4 = vf.update_mask(4, dtype=pl.DT_FP32)
    sel8 = vf.update_mask(8, dtype=pl.DT_FP32)
    s0 = vf.load_align(scale3, 0, dist=pl.LoadDist.BRC_B32)
    s1 = vf.load_align(scale3, 1, dist=pl.LoadDist.BRC_B32)
    s2 = vf.load_align(scale3, 2, dist=pl.LoadDist.BRC_B32)
    v1 = vf.select(s0, s1, sel4)
    sc = vf.select(v1, s2, sel8)
    vf.store_align(scale24, sc, preg24)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    two = vf.full(2.0, preg, dtype=pl.DT_FP32)
    u1 = vf.select(one, two, sel4)
    pm = vf.select(u1, one, sel8)
    vf.store_align(pmul24, pm, preg24)
    ee = vf.full(EPS, preg, dtype=pl.DT_FP32)
    zz = vf.full(0.0, preg, dtype=pl.DT_FP32)
    ev = vf.select(ee, zz, sel4)
    vf.store_align(eps24, ev, preg24)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _gates1_vf(mixr, ssqr, jloc, scale24, pmul24, eps24, base,
               pre_ub, post_ub, zst):
    """Single-token gating decomposition (called independently per AIV, lane=channel 0..23).

    mixr [1,MIXC]: lane 0-23 = mixes[j,:] (after reducing the 28 partials);
    ssqr [1,TG]: BRC(jloc) fetches r. Outputs: pre_ub [1,HC] (for the y BRC);
    post_ub [1,HC]; zst [1,MIX_HC] (24 values).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg24 = vf.update_mask(24, dtype=pl.DT_FP32)
    preg8 = vf.update_mask(8, dtype=pl.DT_FP32)
    preg4 = vf.update_mask(4, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    zero = vf.full(0.0, preg, dtype=pl.DT_FP32)
    scale_r = vf.load_align(scale24, 0)
    pmul_r = vf.load_align(pmul24, 0)
    eps_r = vf.load_align(eps24, 0)
    base_r = vf.load_align(base, 0)
    # r_j = 1/sqrt(ssq_j/16384 + eps)
    ssq_b = vf.load_align(ssqr, jloc, dist=pl.LoadDist.BRC_B32)
    m1 = vf.muls(ssq_b, 1.0 / HC_D, preg)
    m2 = vf.adds(m1, EPS, preg)
    m3 = vf.sqrt(m2, preg)
    rj = vf.div(one, m3, preg)
    # z = mix * r * scale + base
    mix_r = vf.load_align(mixr, 0)
    z1 = vf.mul(mix_r, scale_r, preg24)
    z2 = vf.mul(rj, z1, preg24)
    z = vf.add(z2, base_r, preg24)
    # sigmoid = 1/(1+exp(-z))
    e = vf.exp_sub(zero, z, preg24)
    d = vf.adds(e, 1.0, preg24)
    sig = vf.div(one, d, preg24)
    # Piecewise: lane0-7 sigmoid, lane8-23 linear
    pp0 = vf.select(sig, z, preg8)
    pp1 = vf.mul(pp0, pmul_r, preg24)
    pp = vf.add(pp1, eps_r, preg24)
    # Store zst (24 values, for transpose -> sinkhorn)
    vf.store_align(zst, pp, preg24)
    # Extract pre (lane0-3) into a separate small tile
    # (post is produced by the sinkhorn path)
    pre_r = vf.select(pp, zero, preg4)   # lane0-3: pp, 4+: 0
    vf.store_align(pre_ub, pre_r, preg4)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _zero64_vf(t64):
    """Zero a 64-lane fp32 tile (resets the yssq accumulator for each token)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    zz = vf.full(0.0, preg, dtype=pl.DT_FP32)
    vf.store_align(t64 + 0, zz, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _y_pass1_vf(pre_ub, x_sub, y_row, yssq, off):
    """y pass1 (single chunk, DC=512): y[off:off+DC] = sum_m pre[m]*x[m,:].

    Same 4-row x 4-segment FMA structure as the non-fused version; the
    difference is that pk is no longer written to GM but staged in the
    y_row UB tile (at offset off), and the **bf16-rounded** pk is re-read
    as even/odd halves to accumulate the square FMA into yssq (a 64-lane
    accumulator carried across chunks) -- matching the semantics of the
    production RmsNorm, which reads the bf16 y.
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    p0 = vf.load_align(pre_ub, 0, dist=pl.LoadDist.BRC_B32)
    p1 = vf.load_align(pre_ub, 1, dist=pl.LoadDist.BRC_B32)
    p2 = vf.load_align(pre_ub, 2, dist=pl.LoadDist.BRC_B32)
    p3 = vf.load_align(pre_ub, 3, dist=pl.LoadDist.BRC_B32)
    b = vf.load_align(x_sub, 0)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye0 = vf.mul(e, p0, preg)
    yo0 = vf.mul(o, p0, preg)
    b = vf.load_align(x_sub, 128)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye1 = vf.mul(e, p0, preg)
    yo1 = vf.mul(o, p0, preg)
    b = vf.load_align(x_sub, 256)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye2 = vf.mul(e, p0, preg)
    yo2 = vf.mul(o, p0, preg)
    b = vf.load_align(x_sub, 384)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye3 = vf.mul(e, p0, preg)
    yo3 = vf.mul(o, p0, preg)

    b = vf.load_align(x_sub, 512)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye0 = vf.mul_add_dst(e, p1, preg)
    yo0 = vf.mul_add_dst(o, p1, preg)
    b = vf.load_align(x_sub, 640)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye1 = vf.mul_add_dst(e, p1, preg)
    yo1 = vf.mul_add_dst(o, p1, preg)
    b = vf.load_align(x_sub, 768)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye2 = vf.mul_add_dst(e, p1, preg)
    yo2 = vf.mul_add_dst(o, p1, preg)
    b = vf.load_align(x_sub, 896)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye3 = vf.mul_add_dst(e, p1, preg)
    yo3 = vf.mul_add_dst(o, p1, preg)

    b = vf.load_align(x_sub, 1024)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye0 = vf.mul_add_dst(e, p2, preg)
    yo0 = vf.mul_add_dst(o, p2, preg)
    b = vf.load_align(x_sub, 1152)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye1 = vf.mul_add_dst(e, p2, preg)
    yo1 = vf.mul_add_dst(o, p2, preg)
    b = vf.load_align(x_sub, 1280)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye2 = vf.mul_add_dst(e, p2, preg)
    yo2 = vf.mul_add_dst(o, p2, preg)
    b = vf.load_align(x_sub, 1408)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye3 = vf.mul_add_dst(e, p2, preg)
    yo3 = vf.mul_add_dst(o, p2, preg)

    b = vf.load_align(x_sub, 1536)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye0 = vf.mul_add_dst(e, p3, preg)
    yo0 = vf.mul_add_dst(o, p3, preg)
    b = vf.load_align(x_sub, 1664)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye1 = vf.mul_add_dst(e, p3, preg)
    yo1 = vf.mul_add_dst(o, p3, preg)
    b = vf.load_align(x_sub, 1792)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye2 = vf.mul_add_dst(e, p3, preg)
    yo2 = vf.mul_add_dst(o, p3, preg)
    b = vf.load_align(x_sub, 1920)
    e = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(b, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ye3 = vf.mul_add_dst(e, p3, preg)
    yo3 = vf.mul_add_dst(o, p3, preg)
    # ---- bf16 packing + y_row staging + yssq square accumulation
    # (re-reading the rounded pk) ----
    a = vf.astype(ye0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(yo0, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk0 = vf.add(a, c, preg_bf)
    vf.store_align(y_row + off + 0, pk0, preg_bf)

    a = vf.astype(ye1, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(yo1, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk1 = vf.add(a, c, preg_bf)
    vf.store_align(y_row + off + 128, pk1, preg_bf)

    a = vf.astype(ye2, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(yo2, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk2 = vf.add(a, c, preg_bf)
    vf.store_align(y_row + off + 256, pk2, preg_bf)

    a = vf.astype(ye3, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(yo3, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk3 = vf.add(a, c, preg_bf)
    vf.store_align(y_row + off + 384, pk3, preg_bf)
    acc = vf.load_align(yssq, 0)
    e = vf.astype(pk0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    acc = vf.mul_add_dst(e, e, preg)
    o = vf.astype(pk0, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    acc = vf.mul_add_dst(o, o, preg)

    e = vf.astype(pk1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    acc = vf.mul_add_dst(e, e, preg)
    o = vf.astype(pk1, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    acc = vf.mul_add_dst(o, o, preg)

    e = vf.astype(pk2, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    acc = vf.mul_add_dst(e, e, preg)
    o = vf.astype(pk2, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    acc = vf.mul_add_dst(o, o, preg)

    e = vf.astype(pk3, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    acc = vf.mul_add_dst(e, e, preg)
    o = vf.astype(pk3, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    acc = vf.mul_add_dst(o, o, preg)
    vf.store_align(yssq + 0, acc, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _rstd_vf(yssq, rstd_t):
    """rstd = 1/sqrt(sum(y^2)/4096 + RMS_EPS), broadcast over 64 lanes into rstd_t."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    one = vf.full(1.0, preg, dtype=pl.DT_FP32)
    acc = vf.load_align(yssq, 0)
    sval = vf.reduce_sum(acc, preg)
    m1 = vf.muls(sval, 1.0 / D, preg)
    m2 = vf.adds(m1, RMS_EPS, preg)
    m3 = vf.sqrt(m2, preg)
    r = vf.div(one, m3, preg)
    vf.store_align(rstd_t + 0, r, preg)
    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


@pl.vector_function
def _y_norm_vf(y_row, gamma, rstd_t, y_ub, off):
    """y pass2 (single chunk, DC=512): y_norm = y_bf16 * rstd * gamma -> bf16.

    Reads 128-element segments from the y_row/gamma UB tiles (ZERO/ONE
    even/odd expansion, so the lanes are naturally aligned); rstd is
    broadcast via BRC_B32; y_ub is a double-buffered slot (rotated by the
    kernel body and written to GM with pl.store).
    """
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_bf = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_BF16)
    r = vf.load_align(rstd_t, 0, dist=pl.LoadDist.BRC_B32)
    pk = vf.load_align(y_row, off + 0)
    e = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    g = vf.load_align(gamma, off + 0)
    ge = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    go = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ne = vf.mul(e, r, preg)
    ne = vf.mul(ne, ge, preg)
    no = vf.mul(o, r, preg)
    no = vf.mul(no, go, preg)
    a = vf.astype(ne, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(no, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk = vf.add(a, c, preg_bf)
    vf.store_align(y_ub + 0, pk, preg_bf)

    pk = vf.load_align(y_row, off + 128)
    e = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    g = vf.load_align(gamma, off + 128)
    ge = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    go = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ne = vf.mul(e, r, preg)
    ne = vf.mul(ne, ge, preg)
    no = vf.mul(o, r, preg)
    no = vf.mul(no, go, preg)
    a = vf.astype(ne, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(no, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk = vf.add(a, c, preg_bf)
    vf.store_align(y_ub + 128, pk, preg_bf)

    pk = vf.load_align(y_row, off + 256)
    e = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    g = vf.load_align(gamma, off + 256)
    ge = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    go = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ne = vf.mul(e, r, preg)
    ne = vf.mul(ne, ge, preg)
    no = vf.mul(o, r, preg)
    no = vf.mul(no, go, preg)
    a = vf.astype(ne, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(no, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk = vf.add(a, c, preg_bf)
    vf.store_align(y_ub + 256, pk, preg_bf)

    pk = vf.load_align(y_row, off + 384)
    e = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    o = vf.astype(pk, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    g = vf.load_align(gamma, off + 384)
    ge = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ZERO)
    go = vf.astype(g, preg_bf, dtype=pl.DT_FP32, layout=pl.CastLayout.ONE)
    ne = vf.mul(e, r, preg)
    ne = vf.mul(ne, ge, preg)
    no = vf.mul(o, r, preg)
    no = vf.mul(no, go, preg)
    a = vf.astype(ne, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ZERO)
    c = vf.astype(no, preg, dtype=pl.DT_BF16, layout=pl.CastLayout.ONE)
    pk = vf.add(a, c, preg_bf)
    vf.store_align(y_ub + 384, pk, preg_bf)


@pl.vector_function
def _sinkhorn_vf(zst, zsd, cbt, pst, tg):
    """Single-token Sinkhorn: zsd [24,1] rows 8-23 = comb_flag, 4x4 over 20 iterations (tg=1, single lane)."""
    preg = vf.create_mask(pattern=pl.MaskPattern.ALL, dtype=pl.DT_FP32)
    preg_tg = vf.update_mask(tg, dtype=pl.DT_FP32)

    z00 = vf.load_align(zsd, (2 * HC + 0) * TG)
    z01 = vf.load_align(zsd, (2 * HC + 1) * TG)
    z02 = vf.load_align(zsd, (2 * HC + 2) * TG)
    z03 = vf.load_align(zsd, (2 * HC + 3) * TG)
    z10 = vf.load_align(zsd, (2 * HC + 4) * TG)
    z11 = vf.load_align(zsd, (2 * HC + 5) * TG)
    z12 = vf.load_align(zsd, (2 * HC + 6) * TG)
    z13 = vf.load_align(zsd, (2 * HC + 7) * TG)
    z20 = vf.load_align(zsd, (2 * HC + 8) * TG)
    z21 = vf.load_align(zsd, (2 * HC + 9) * TG)
    z22 = vf.load_align(zsd, (2 * HC + 10) * TG)
    z23 = vf.load_align(zsd, (2 * HC + 11) * TG)
    z30 = vf.load_align(zsd, (2 * HC + 12) * TG)
    z31 = vf.load_align(zsd, (2 * HC + 13) * TG)
    z32 = vf.load_align(zsd, (2 * HC + 14) * TG)
    z33 = vf.load_align(zsd, (2 * HC + 15) * TG)

    qa0 = vf.max(z00, z01, preg_tg)
    qb0 = vf.max(z02, z03, preg_tg)
    rmx0 = vf.max(qa0, qb0, preg_tg)
    qa1 = vf.max(z10, z11, preg_tg)
    qb1 = vf.max(z12, z13, preg_tg)
    rmx1 = vf.max(qa1, qb1, preg_tg)
    qa2 = vf.max(z20, z21, preg_tg)
    qb2 = vf.max(z22, z23, preg_tg)
    rmx2 = vf.max(qa2, qb2, preg_tg)
    qa3 = vf.max(z30, z31, preg_tg)
    qb3 = vf.max(z32, z33, preg_tg)
    rmx3 = vf.max(qa3, qb3, preg_tg)

    e00 = vf.exp_sub(z00, rmx0, preg_tg)
    e01 = vf.exp_sub(z01, rmx0, preg_tg)
    e02 = vf.exp_sub(z02, rmx0, preg_tg)
    e03 = vf.exp_sub(z03, rmx0, preg_tg)
    e10 = vf.exp_sub(z10, rmx1, preg_tg)
    e11 = vf.exp_sub(z11, rmx1, preg_tg)
    e12 = vf.exp_sub(z12, rmx1, preg_tg)
    e13 = vf.exp_sub(z13, rmx1, preg_tg)
    e20 = vf.exp_sub(z20, rmx2, preg_tg)
    e21 = vf.exp_sub(z21, rmx2, preg_tg)
    e22 = vf.exp_sub(z22, rmx2, preg_tg)
    e23 = vf.exp_sub(z23, rmx2, preg_tg)
    e30 = vf.exp_sub(z30, rmx3, preg_tg)
    e31 = vf.exp_sub(z31, rmx3, preg_tg)
    e32 = vf.exp_sub(z32, rmx3, preg_tg)
    e33 = vf.exp_sub(z33, rmx3, preg_tg)

    sa0 = vf.add(e00, e01, preg_tg)
    sb0 = vf.add(e02, e03, preg_tg)
    rs0 = vf.add(sa0, sb0, preg_tg)
    sa1 = vf.add(e10, e11, preg_tg)
    sb1 = vf.add(e12, e13, preg_tg)
    rs1 = vf.add(sa1, sb1, preg_tg)
    sa2 = vf.add(e20, e21, preg_tg)
    sb2 = vf.add(e22, e23, preg_tg)
    rs2 = vf.add(sa2, sb2, preg_tg)
    sa3 = vf.add(e30, e31, preg_tg)
    sb3 = vf.add(e32, e33, preg_tg)
    rs3 = vf.add(sa3, sb3, preg_tg)

    dv00 = vf.div(e00, rs0, preg_tg)
    c00 = vf.adds(dv00, EPS, preg_tg)
    dv01 = vf.div(e01, rs0, preg_tg)
    c01 = vf.adds(dv01, EPS, preg_tg)
    dv02 = vf.div(e02, rs0, preg_tg)
    c02 = vf.adds(dv02, EPS, preg_tg)
    dv03 = vf.div(e03, rs0, preg_tg)
    c03 = vf.adds(dv03, EPS, preg_tg)
    dv10 = vf.div(e10, rs1, preg_tg)
    c10 = vf.adds(dv10, EPS, preg_tg)
    dv11 = vf.div(e11, rs1, preg_tg)
    c11 = vf.adds(dv11, EPS, preg_tg)
    dv12 = vf.div(e12, rs1, preg_tg)
    c12 = vf.adds(dv12, EPS, preg_tg)
    dv13 = vf.div(e13, rs1, preg_tg)
    c13 = vf.adds(dv13, EPS, preg_tg)
    dv20 = vf.div(e20, rs2, preg_tg)
    c20 = vf.adds(dv20, EPS, preg_tg)
    dv21 = vf.div(e21, rs2, preg_tg)
    c21 = vf.adds(dv21, EPS, preg_tg)
    dv22 = vf.div(e22, rs2, preg_tg)
    c22 = vf.adds(dv22, EPS, preg_tg)
    dv23 = vf.div(e23, rs2, preg_tg)
    c23 = vf.adds(dv23, EPS, preg_tg)
    dv30 = vf.div(e30, rs3, preg_tg)
    c30 = vf.adds(dv30, EPS, preg_tg)
    dv31 = vf.div(e31, rs3, preg_tg)
    c31 = vf.adds(dv31, EPS, preg_tg)
    dv32 = vf.div(e32, rs3, preg_tg)
    c32 = vf.adds(dv32, EPS, preg_tg)
    dv33 = vf.div(e33, rs3, preg_tg)
    c33 = vf.adds(dv33, EPS, preg_tg)

    ua0 = vf.add(c00, c10, preg_tg)
    ub0 = vf.add(c20, c30, preg_tg)
    wq0 = vf.add(ua0, ub0, preg_tg)
    cs0 = vf.adds(wq0, EPS, preg_tg)
    ua1 = vf.add(c01, c11, preg_tg)
    ub1 = vf.add(c21, c31, preg_tg)
    wq1 = vf.add(ua1, ub1, preg_tg)
    cs1 = vf.adds(wq1, EPS, preg_tg)
    ua2 = vf.add(c02, c12, preg_tg)
    ub2 = vf.add(c22, c32, preg_tg)
    wq2 = vf.add(ua2, ub2, preg_tg)
    cs2 = vf.adds(wq2, EPS, preg_tg)
    ua3 = vf.add(c03, c13, preg_tg)
    ub3 = vf.add(c23, c33, preg_tg)
    wq3 = vf.add(ua3, ub3, preg_tg)
    cs3 = vf.adds(wq3, EPS, preg_tg)

    c00 = vf.div(c00, cs0, preg_tg)
    c01 = vf.div(c01, cs1, preg_tg)
    c02 = vf.div(c02, cs2, preg_tg)
    c03 = vf.div(c03, cs3, preg_tg)
    c10 = vf.div(c10, cs0, preg_tg)
    c11 = vf.div(c11, cs1, preg_tg)
    c12 = vf.div(c12, cs2, preg_tg)
    c13 = vf.div(c13, cs3, preg_tg)
    c20 = vf.div(c20, cs0, preg_tg)
    c21 = vf.div(c21, cs1, preg_tg)
    c22 = vf.div(c22, cs2, preg_tg)
    c23 = vf.div(c23, cs3, preg_tg)
    c30 = vf.div(c30, cs0, preg_tg)
    c31 = vf.div(c31, cs1, preg_tg)
    c32 = vf.div(c32, cs2, preg_tg)
    c33 = vf.div(c33, cs3, preg_tg)

    for _ in pl.range(0, SINKHORN_ITERS - 1):
        aa0 = vf.add(c00, c01, preg_tg)
        ab0 = vf.add(c02, c03, preg_tg)
        wr0 = vf.add(aa0, ab0, preg_tg)
        rs0 = vf.adds(wr0, EPS, preg_tg)
        aa1 = vf.add(c10, c11, preg_tg)
        ab1 = vf.add(c12, c13, preg_tg)
        wr1 = vf.add(aa1, ab1, preg_tg)
        rs1 = vf.adds(wr1, EPS, preg_tg)
        aa2 = vf.add(c20, c21, preg_tg)
        ab2 = vf.add(c22, c23, preg_tg)
        wr2 = vf.add(aa2, ab2, preg_tg)
        rs2 = vf.adds(wr2, EPS, preg_tg)
        aa3 = vf.add(c30, c31, preg_tg)
        ab3 = vf.add(c32, c33, preg_tg)
        wr3 = vf.add(aa3, ab3, preg_tg)
        rs3 = vf.adds(wr3, EPS, preg_tg)
        c00 = vf.div(c00, rs0, preg_tg)
        c01 = vf.div(c01, rs0, preg_tg)
        c02 = vf.div(c02, rs0, preg_tg)
        c03 = vf.div(c03, rs0, preg_tg)
        c10 = vf.div(c10, rs1, preg_tg)
        c11 = vf.div(c11, rs1, preg_tg)
        c12 = vf.div(c12, rs1, preg_tg)
        c13 = vf.div(c13, rs1, preg_tg)
        c20 = vf.div(c20, rs2, preg_tg)
        c21 = vf.div(c21, rs2, preg_tg)
        c22 = vf.div(c22, rs2, preg_tg)
        c23 = vf.div(c23, rs2, preg_tg)
        c30 = vf.div(c30, rs3, preg_tg)
        c31 = vf.div(c31, rs3, preg_tg)
        c32 = vf.div(c32, rs3, preg_tg)
        c33 = vf.div(c33, rs3, preg_tg)
        ba0 = vf.add(c00, c10, preg_tg)
        bb0 = vf.add(c20, c30, preg_tg)
        wc0 = vf.add(ba0, bb0, preg_tg)
        cs0 = vf.adds(wc0, EPS, preg_tg)
        ba1 = vf.add(c01, c11, preg_tg)
        bb1 = vf.add(c21, c31, preg_tg)
        wc1 = vf.add(ba1, bb1, preg_tg)
        cs1 = vf.adds(wc1, EPS, preg_tg)
        ba2 = vf.add(c02, c12, preg_tg)
        bb2 = vf.add(c22, c32, preg_tg)
        wc2 = vf.add(ba2, bb2, preg_tg)
        cs2 = vf.adds(wc2, EPS, preg_tg)
        ba3 = vf.add(c03, c13, preg_tg)
        bb3 = vf.add(c23, c33, preg_tg)
        wc3 = vf.add(ba3, bb3, preg_tg)
        cs3 = vf.adds(wc3, EPS, preg_tg)
        c00 = vf.div(c00, cs0, preg_tg)
        c01 = vf.div(c01, cs1, preg_tg)
        c02 = vf.div(c02, cs2, preg_tg)
        c03 = vf.div(c03, cs3, preg_tg)
        c10 = vf.div(c10, cs0, preg_tg)
        c11 = vf.div(c11, cs1, preg_tg)
        c12 = vf.div(c12, cs2, preg_tg)
        c13 = vf.div(c13, cs3, preg_tg)
        c20 = vf.div(c20, cs0, preg_tg)
        c21 = vf.div(c21, cs1, preg_tg)
        c22 = vf.div(c22, cs2, preg_tg)
        c23 = vf.div(c23, cs3, preg_tg)
        c30 = vf.div(c30, cs0, preg_tg)
        c31 = vf.div(c31, cs1, preg_tg)
        c32 = vf.div(c32, cs2, preg_tg)
        c33 = vf.div(c33, cs3, preg_tg)

    vf.store_align(cbt + 0, c00, preg_tg)
    vf.store_align(cbt + 8, c01, preg_tg)
    vf.store_align(cbt + 16, c02, preg_tg)
    vf.store_align(cbt + 24, c03, preg_tg)
    vf.store_align(cbt + 32, c10, preg_tg)
    vf.store_align(cbt + 40, c11, preg_tg)
    vf.store_align(cbt + 48, c12, preg_tg)
    vf.store_align(cbt + 56, c13, preg_tg)
    vf.store_align(cbt + 64, c20, preg_tg)
    vf.store_align(cbt + 72, c21, preg_tg)
    vf.store_align(cbt + 80, c22, preg_tg)
    vf.store_align(cbt + 88, c23, preg_tg)
    vf.store_align(cbt + 96, c30, preg_tg)
    vf.store_align(cbt + 104, c31, preg_tg)
    vf.store_align(cbt + 112, c32, preg_tg)
    vf.store_align(cbt + 120, c33, preg_tg)
    hp0 = vf.load_align(zsd, (HC + 0) * TG)
    vf.store_align(pst + 0, hp0, preg_tg)
    hp1 = vf.load_align(zsd, (HC + 1) * TG)
    vf.store_align(pst + 8, hp1, preg_tg)
    hp2 = vf.load_align(zsd, (HC + 2) * TG)
    vf.store_align(pst + 16, hp2, preg_tg)
    hp3 = vf.load_align(zsd, (HC + 3) * TG)
    vf.store_align(pst + 24, hp3, preg_tg)

    vf.mem_bar(mode=pl.MemBarMode.VST_VLD)


# ======================================================================
# Common tile declarations (vector side, layout shared by both kernels)
# ======================================================================

def _make_vector_tiles():
    xsq = pl.make_tile_group(
        type=pl.TileType(shape=[TG, SSQ_DK], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=XSQ_UB, mutex_ids=[9])
    st = pl.make_tile_group(
        type=pl.TileType(shape=[TG, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=SSQST_UB, mutex_ids=[10])
    tt = pl.make_tile_group(
        type=pl.TileType(shape=[64, TG], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SSQTT_UB, mutex_ids=[11])
    slab = pl.make_tile_group(
        type=pl.TileType(shape=[NB, MIXC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=SLAB_UB, mutex_ids=[12])
    mixr = pl.make_tile_group(
        type=pl.TileType(shape=[1, MIXC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=MIXR_UB, mutex_ids=[13])
    ql = pl.make_tile_group(
        type=pl.TileType(shape=[SSQ_SLICES, TG], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SSQL_UB, mutex_ids=[14])
    qr = pl.make_tile_group(
        type=pl.TileType(shape=[1, TG], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SSQR_UB, mutex_ids=[15])
    sc3 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=SCALE3_UB, mutex_ids=[16])
    sc24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=SCALE24_UB, mutex_ids=[16])
    pm24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=PMUL24_UB, mutex_ids=[16])
    ep24 = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=EPS24_UB, mutex_ids=[16])
    base = pl.make_tile_group(
        type=pl.TileType(shape=[1, 32], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=BASE_UB, mutex_ids=[16])
    zst = pl.make_tile_group(
        type=pl.TileType(shape=[1, MIX_HC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=ZST_UB, mutex_ids=[21])
    zsd = pl.make_tile_group(
        type=pl.TileType(shape=[MIX_HC, TG], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=ZSD_UB, mutex_ids=[22])
    cbt = pl.make_tile_group(
        type=pl.TileType(shape=[16, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=CBT_UB, mutex_ids=[23])
    cbtt = pl.make_tile_group(
        type=pl.TileType(shape=[1, 16], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=CBTT_UB, mutex_ids=[24])
    pst = pl.make_tile_group(
        type=pl.TileType(shape=[HC, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=PST_UB, mutex_ids=[25])
    pstt = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec,
                         valid_shape=[-1, -1]),
        addrs=PSTT_UB, mutex_ids=[26])
    pre = pl.make_tile_group(
        type=pl.TileType(shape=[1, 8], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=PRE_UB, mutex_ids=[27])
    xsub = pl.make_tile_group(
        type=pl.TileType(shape=[HC, DC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=XSUB_UB, mutex_ids=[28, 29])
    yub = pl.make_tile_group(
        type=pl.TileType(shape=[1, DC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=YUB_UB, mutex_ids=[30, 31])
    yrow = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=YROW_UB, mutex_ids=[17])
    gam = pl.make_tile_group(
        type=pl.TileType(shape=[1, D], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Vec),
        addrs=GAMMA_UB, mutex_ids=[16])
    rstdg = pl.make_tile_group(
        type=pl.TileType(shape=[1, 64], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Vec),
        addrs=RSTD_UB, mutex_ids=[18])
    return (xsq, st, tt, slab, mixr, ql, qr, sc3, sc24, pm24, ep24, base,
            zst, zsd, cbt, cbtt, pst, pstt, pre, xsub, yub, yrow, gam, rstdg)


# ======================================================================
# Kernel: K-axis split (27x592 + 400 on the last core), load balanced over
# all 28 cube cores
# ======================================================================

@pl.jit(auto_mutex=True)
def mhcpresinkhorn_rmsnorm_kernel_k(
    x_2d:    pl.Tensor[[pl.DYNAMIC, HC_D], pl.DT_BF16],
    x_3d:    pl.Tensor[[pl.DYNAMIC, HC, D], pl.DT_BF16],
    hc_fn:   pl.Tensor[[MIX_HC, HC_D], pl.DT_BF16],
    hc_scale: pl.Tensor[[1, 3], pl.DT_FP32],
    hc_base:  pl.Tensor[[1, MIX_HC], pl.DT_FP32],
    gamma:    pl.Tensor[[1, D], pl.DT_BF16],
    y:        pl.Tensor[[pl.DYNAMIC, D], pl.DT_BF16],
    rstd:     pl.Tensor[[pl.DYNAMIC, 1], pl.DT_FP32],
    post:     pl.Tensor[[pl.DYNAMIC, HC], pl.DT_FP32],
    comb16:   pl.Tensor[[pl.DYNAMIC, MIX_HC - 8], pl.DT_FP32],
    pmix3:    pl.Tensor[[NB, pl.DYNAMIC, MIXC], pl.DT_FP32],
    pmix3f:   pl.Tensor[[NB, pl.DYNAMIC], pl.DT_FP32],
    ssq_ws:   pl.Tensor[[SSQ_SLICES, pl.DYNAMIC], pl.DT_FP32],
):
    t = x_2d.shape[0]
    ng = (t + TG - 1) // TG

    # ---- Cube tiles (two shapes: full-block cores [16,592]/[592,32], last
    #      core [16,400]/[400,32]; fully covered load with no tail block.
    #      L0B totals 63.5KB <= 64KB. Note: fillpad is not supported for Mat
    #      tiles (Vec only), and the GM->L1 load cannot use a bar barrier
    #      either (the generated code lowers both bar_mte1/2 in the cube
    #      section to PIPE_MTE1), hence the zero-baseline approach is not
    #      used) ----
    lhs_src = pl.make_tile_group(
        type=pl.TileType(shape=[TG, KA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=LHS_SRC_L1, mutex_ids=[0])
    lhs_src2 = pl.make_tile_group(
        type=pl.TileType(shape=[TG, KLAST], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=LHS_SRC2_L1, mutex_ids=[1])
    rhs_src = pl.make_tile_group(
        type=pl.TileType(shape=[KA, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                         valid_shape=[-1, -1]),
        addrs=RHS_SRC_L1, mutex_ids=[2])
    rhs_src2 = pl.make_tile_group(
        type=pl.TileType(shape=[KLAST, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Mat, layout=pl.ZN,
                         valid_shape=[-1, -1]),
        addrs=RHS_SRC2_L1, mutex_ids=[3])
    l0a = pl.make_tile_group(
        type=pl.TileType(shape=[TG, KA], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x0, mutex_ids=[4])
    l0a2 = pl.make_tile_group(
        type=pl.TileType(shape=[TG, KLAST], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Left, layout=pl.NZ),
        addrs=0x5000, mutex_ids=[5])
    l0b = pl.make_tile_group(
        type=pl.TileType(shape=[KA, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x0, mutex_ids=[6])
    l0b2 = pl.make_tile_group(
        type=pl.TileType(shape=[KLAST, MIXC], dtype=pl.DT_BF16,
                         target_memory=pl.MemorySpace.Right, layout=pl.ZN),
        addrs=0x9800, mutex_ids=[7])
    l0c = pl.make_tile_group(
        type=pl.TileType(shape=[TG, MIXC], dtype=pl.DT_FP32,
                         target_memory=pl.MemorySpace.Acc, layout=pl.NZ,
                         valid_shape=[-1, -1]),
        addrs=0x0, mutex_ids=[8])
    (xsq, st, tt, slab, mixr, ql, qr, sc3, sc24, pm24, ep24, base,
     zst, zsd, cbt, cbtt, pst, pstt, pre, xsub, yub, yrow, gam, rstdg) =         _make_vector_tiles()

    # ========== Cube: K-split matmul (runs in parallel with the Vector ssq) ==========
    with pl.section_cube():
        core = pl.get_block_idx()
        kstart = core * KA
        c_t = l0c.current()
        if core < NB - 1:
            # ---- Full-block core (K=592): fully covered, no tail block ----
            lsrc = lhs_src.current()
            rsrc = rhs_src.current()
            a_t = l0a.current()
            b_t = l0b.current()
            pl.set_validshape(rsrc, [KA, MIX_HC])
            pl.load(rsrc, hc_fn, [0, kstart], order=[1, 0])
            pl.move(b_t, rsrc)
            for g in pl.range(0, ng):
                tg = pl.min(TG, t - g * TG)
                pl.set_validshape(lsrc, [tg, KA])
                pl.load(lsrc, x_2d, [g * TG, kstart])
                pl.move(a_t, lsrc)
                pl.matmul(c_t, a_t, b_t)
                pl.set_validshape(c_t, [tg, MIX_HC])
                pl.store(pmix3, c_t, [core, g * TG, 0])
                # B(g): matmul(g) done (the Vector sync point is after ssq(g))
                pl.system.sync_all(mode=pl.SyncAllMode.HARD,
                                   core_type=pl.SyncCoreType.MIX)
        else:
            # ---- Last core (K=400): full coverage with the small tile ----
            lsrc2 = lhs_src2.current()
            rsrc2 = rhs_src2.current()
            a_t2 = l0a2.current()
            b_t2 = l0b2.current()
            pl.set_validshape(rsrc2, [KLAST, MIX_HC])
            pl.load(rsrc2, hc_fn, [0, kstart], order=[1, 0])
            pl.move(b_t2, rsrc2)
            for g in pl.range(0, ng):
                tg = pl.min(TG, t - g * TG)
                pl.set_validshape(lsrc2, [tg, KLAST])
                pl.load(lsrc2, x_2d, [g * TG, kstart])
                pl.move(a_t2, lsrc2)
                pl.matmul(c_t, a_t2, b_t2)
                pl.set_validshape(c_t, [tg, MIX_HC])
                pl.store(pmix3, c_t, [core, g * TG, 0])
                pl.system.sync_all(mode=pl.SyncAllMode.HARD,
                                   core_type=pl.SyncCoreType.MIX)

    # ========== Vector: RMSNorm || Cube, token-distributed Stage2 ==========
    with pl.section_vector():
        gidx = pl.get_block_idx()
        xq_t = xsq.current()
        st_t = st.current()
        tt_t = tt.current()
        sl_t = slab.current()
        mr_t = mixr.current()
        ql_t = ql.current()
        qr_t = qr.current()
        sc3_t = sc3.current()
        sc24_t = sc24.current()
        pm24_t = pm24.current()
        ep24_t = ep24.current()
        base_tile = base.current()
        zs_t = zst.current()
        zsd_t = zsd.current()
        cb_t = cbt.current()
        cbt_t = cbtt.current()
        ps_t = pst.current()
        pst_t = pstt.current()
        pr_t = pre.current()
        yr_t = yrow.current()
        gam_t = gam.current()
        rt_t = rstdg.current()

        # Build aux vectors (once per AIV)
        pl.set_validshape(sc3_t, [1, 3])
        pl.load(sc3_t, hc_scale, [0, 0])
        pl.set_validshape(base_tile, [1, MIX_HC])
        pl.load(base_tile, hc_base, [0, 0])
        _build_aux_vf(sc3_t, sc24_t, pm24_t, ep24_t)
        # Preload the RmsNorm gamma (token invariant, once per AIV)
        pl.set_validshape(gam_t, [1, D])
        pl.load(gam_t, gamma, [0, 0])

        for g in pl.range(0, ng):
            tg = pl.min(TG, t - g * TG)
            # ---- RMSNorm partial (parallel to the Cube matmul, no cross dependency) ----
            if gidx < SSQ_SLICES:
                pl.set_validshape(xq_t, [tg, SSQ_DK])
                pl.load(xq_t, x_2d, [g * TG, gidx * SSQ_DK])
                _ssq_vf(xq_t, st_t, tg)
                pl.set_validshape(tt_t, [64, tg])
                pl.transpose(tt_t, st_t)
                pl.set_validshape(tt_t, [1, tg])
                pl.store(ssq_ws, tt_t, [gidx, g * TG])
            # ---- B(g): both the Cube matmul and the Vector ssq are done ----
            pl.system.sync_all(mode=pl.SyncAllMode.HARD,
                               core_type=pl.SyncCoreType.MIX)
            # ---- Stage2 (parallel to the Cube matmul of g+1) ----
            j = g * TG + gidx
            if gidx < tg:
                jloc = gidx
                # Prefetch y chunk0 (the GM->UB MTE2 overlaps with the
                # reduction / gates computation below)
                xs_cur = xsub.next()
                pl.load(xs_cur, x_3d, [j, 0, 0])
                # mixes: 28-way partial reduction
                pl.set_validshape(sl_t, [NB, MIXC])
                pl.load(sl_t, pmix3f, [0, g * TG * MIXC + jloc * MIXC])
                pl.set_validshape(mr_t, [1, MIX_HC])
                pl.sum(mr_t, sl_t, sl_t, dim=1)
                # r: 32-way ssq reduction
                pl.set_validshape(ql_t, [SSQ_SLICES, tg])
                pl.load(ql_t, ssq_ws, [0, g * TG])
                pl.set_validshape(qr_t, [1, tg])
                pl.sum(qr_t, ql_t, ql_t, dim=1)
                # Gating decomposition (single token)
                _gates1_vf(mr_t, qr_t, jloc, sc24_t, pm24_t, ep24_t,
                           base_tile, pr_t, ps_t, zs_t)
                # ---- y pass1 (first half of the fused RmsNorm): compute y,
                # stage it in UB and accumulate the sum of squares ----
                # (the chunk0 prefetch slot was already issued before the
                #  reductions; the x double buffering stays unchanged)
                _zero64_vf(rt_t)
                _y_pass1_vf(pr_t, xs_cur, yr_t, rt_t, 0)
                for dc in pl.range(1, NDC):
                    xs_cur = xsub.next()
                    pl.load(xs_cur, x_3d, [j, 0, dc * DC])
                    _y_pass1_vf(pr_t, xs_cur, yr_t, rt_t, dc * DC)
                # ---- rstd = 1/sqrt(mean(y^2)+rms_eps), store to GM [t,1] ----
                _rstd_vf(rt_t, rt_t)
                pl.set_validshape(rt_t, [1, 1])
                pl.store(rstd, rt_t, [j, 0])
                # ---- y pass2: y_norm = y*rstd*gamma, double-buffered
                # store to GM ----
                for dc in pl.range(0, NDC):
                    yu_cur = yub.next()
                    _y_norm_vf(yr_t, gam_t, rt_t, yu_cur, dc * DC)
                    pl.store(y, yu_cur, [j, dc * DC])
                # Sinkhorn (zstage transposed -> 16 registers)
                pl.set_validshape(zs_t, [1, MIX_HC])
                pl.set_validshape(zsd_t, [MIX_HC, 1])
                pl.transpose(zsd_t, zs_t)
                _sinkhorn_vf(zs_t, zsd_t, cb_t, ps_t, 1)
                # comb transposed output
                pl.set_validshape(cb_t, [16, 1])
                pl.set_validshape(cbt_t, [1, 16])
                pl.transpose(cbt_t, cb_t)
                pl.store(comb16, cbt_t, [j, 0])
                # post transposed output
                pl.set_validshape(ps_t, [HC, 1])
                pl.set_validshape(pst_t, [1, HC])
                pl.transpose(pst_t, ps_t)
                pl.store(post, pst_t, [j, 0])


# ======================================================================
# Host wrapper
# ======================================================================

_WS_CACHE = {}
_WS_CACHE_MAX = 8


def _get_workspaces(t, device):
    key = (t, device.index if device.index is not None
           else torch.npu.current_device())
    ws = _WS_CACHE.get(key)
    if ws is None:
        if len(_WS_CACHE) >= _WS_CACHE_MAX:
            _WS_CACHE.pop(next(iter(_WS_CACHE)))
        ng = (t + TG - 1) // TG
        ssq_ws = torch.empty((SSQ_SLICES, ng * TG), dtype=torch.float32,
                             device=device)
        pmix3 = torch.empty((NB, ng * TG, MIXC),
                            dtype=torch.float32, device=device)
        ws = (pmix3, pmix3.view(NB, ng * TG * MIXC), ssq_ws)
        _WS_CACHE[key] = ws
    return ws


def mhcpresinkhorn_rmsnorm(x, hc_fn, hc_scale, hc_base, gamma, hc_mult=4,
                sinkhorn_iters=SINKHORN_ITERS, eps=EPS, rms_eps=RMS_EPS,
                **kwargs):
    """Public entry of mhcpresinkhorn_rmsnorm PyPTO-Pro v4-K + RmsNorm fusion (bf16 hc_fn).

    The matmul splits the K axis to balance the load across all 28 cube
    cores (27x592 + 400 on the last core); RmsNorm is fused in place right
    after y is computed (y_norm = y*rstd*gamma,
    rstd = 1/sqrt(mean(y^2)+rms_eps)), so y carries the normalized values
    and the original bf16 y is no longer emitted separately (it is only
    consumed by RmsNorm in the production chain).
    """
    if hc_mult != HC:
        raise ValueError(f"hc_mult={hc_mult} != {HC}")
    if sinkhorn_iters != SINKHORN_ITERS:
        raise ValueError(f"sinkhorn_iters={sinkhorn_iters} != "
                         f"{SINKHORN_ITERS}")
    if eps != EPS:
        raise ValueError(f"eps={eps} != {EPS}")
    if x.dtype != torch.bfloat16 or x.dim() != 3 \
            or x.shape[1] != HC or x.shape[2] != D:
        raise ValueError(f"mhcpresinkhorn_rmsnorm: x must be [t,{HC},{D}] bf16, "
                         f"got {tuple(x.shape)}/{x.dtype}")
    if not x.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: x must be contiguous")
    if hc_fn.dtype != torch.bfloat16 or tuple(hc_fn.shape) != (MIX_HC, HC_D):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: hc_fn must be [{MIX_HC},{HC_D}] bf16 "
                         f"(model side converts weight dtype), got "
                         f"{tuple(hc_fn.shape)}/{hc_fn.dtype}")
    if not hc_fn.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_fn must be contiguous")
    if hc_scale.dtype != torch.float32 or tuple(hc_scale.shape) != (3,):
        raise ValueError("mhcpresinkhorn_rmsnorm: hc_scale must be [3] fp32")
    if hc_base.dtype != torch.float32 or tuple(hc_base.shape) != (MIX_HC,):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: hc_base must be [{MIX_HC}] fp32")
    if gamma.dtype != torch.bfloat16 or tuple(gamma.shape) != (D,):
        raise ValueError(f"mhcpresinkhorn_rmsnorm: gamma must be [{D}] bf16")
    if not gamma.is_contiguous():
        raise ValueError("mhcpresinkhorn_rmsnorm: gamma must be contiguous")

    device = torch.device(
        f"npu:{int(os.environ.get('TILE_FWK_DEVICE_ID', '0'))}")
    torch.npu.set_device(device)
    x = x.to(device)
    hc_fn = hc_fn.to(device)
    hc_scale = hc_scale.to(device)
    hc_base = hc_base.to(device)
    gamma = gamma.to(device)

    t = x.shape[0]
    x_2d = x.reshape(t, HC_D)
    y = torch.empty((t, D), dtype=torch.bfloat16, device=device)
    rstd = torch.empty((t, 1), dtype=torch.float32, device=device)
    post = torch.empty((t, HC), dtype=torch.float32, device=device)
    comb = torch.empty((t, HC, HC), dtype=torch.float32, device=device)
    comb16 = comb.view(t, HC * HC)

    pmix3, pmix3f, ssq_ws = _get_workspaces(t, device)
    mhcpresinkhorn_rmsnorm_kernel_k[None, NB](
        x_2d, x, hc_fn, hc_scale.view(1, 3), hc_base.view(1, MIX_HC),
        gamma.view(1, D), y, rstd, post, comb16, pmix3, pmix3f, ssq_ws)
    return y, rstd, post, comb
