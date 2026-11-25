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
"""TileLang-Ascend DCP sparse-index remap used by JIT and AOT callers."""

import tilelang.language as T
from tilelang import tvm

from xllm_kernel.ops.attention.dsa import SFA_DCP_REMAP_MAX_TOKENS, SFA_DCP_REMAP_TOPK

VEC_NUM = 2
REMAP_EVT_SRC0 = 0
REMAP_EVT_IDX = 2
REMAP_EVT_OUT = 3
REMAP_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.disable_safe_memory_legalize": True,
}


def build_remap_topk_kernel(
    *,
    vec_core_num: int,
    topk: int,
) -> tvm.tir.PrimFunc:
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")
    if topk != SFA_DCP_REMAP_TOPK:
        raise ValueError(f"topk must be {SFA_DCP_REMAP_TOPK}, got {topk}")

    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num
    max_elems = SFA_DCP_REMAP_MAX_TOKENS * SFA_DCP_REMAP_TOPK
    full_mask_len = topk // 8
    neg_inf = -1.0e30
    gather_elem_bytes = 4

    @T.prim_func
    def remap_topk_kernel(
        topk_in: T.Tensor((max_elems,), "int32"),
        topk_out: T.Tensor((max_elems,), "int32"),
        idx_scratch: T.Tensor((max_elems,), "int32"),
        idx_scratch_u: T.Tensor((max_elems,), "uint32"),
        num_tokens: T.int32,
        physical_block_size: T.int32,
        shard_size: T.int32,
        shard_rank: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            per_task = (num_tokens + task_num - 1) // task_num
            token_start = task_id * per_task
            tokens_left = T.if_then_else(num_tokens > token_start, num_tokens - token_start, 0)
            token_count = T.if_then_else(tokens_left < per_task, tokens_left, per_task)

            with T.Scope("V"):
                full_src_ub = T.alloc_ub((topk,), "int32")
                full_off_ub = T.alloc_ub((topk,), "int32")
                full_idx_u_ub = T.alloc_ub((topk,), "uint32")
                full_fp_ub = T.alloc_ub((topk,), "float32")
                full_keys_ub = T.alloc_ub((topk,), "float32")
                full_idx_fp_ub = T.alloc_ub((topk,), "float32")
                full_out_fp_ub = T.alloc_ub((topk,), "float32")
                full_sort_ub = T.alloc_ub((topk * 2,), "float32")
                full_ge_mask = T.alloc_ub((full_mask_len,), "uint8")
                full_owner_mask = T.alloc_ub((full_mask_len,), "uint8")
                work0_fp_ub = T.alloc_ub((topk,), "float32")
                logical_fp = T.Cast("float32", physical_block_size * shard_size)
                physical_fp = T.Cast("float32", physical_block_size)
                T.set_flag("v", "mte2", REMAP_EVT_SRC0)
                T.wait_flag("v", "mte2", REMAP_EVT_SRC0)
                for local_token in T.serial(token_count):
                    token = token_start + local_token
                    src_off = token * topk
                    with T.If(local_token > 0), T.Then():
                        T.wait_flag("mte3", "v", REMAP_EVT_OUT)
                        T.wait_flag("mte3", "mte2", REMAP_EVT_OUT)
                    T.copy(topk_in[src_off], full_src_ub)
                    T.set_flag("mte2", "v", REMAP_EVT_SRC0)
                    T.wait_flag("mte2", "v", REMAP_EVT_SRC0)
                    T.tile.cast(full_fp_ub, full_src_ub, "CAST_NONE", topk)
                    T.tile.compare(full_ge_mask, full_fp_ub, T.float32(0.0), "GE")
                    T.tile.select(
                        full_keys_ub,
                        full_ge_mask,
                        full_fp_ub,
                        T.float32(0.0),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.tile.div(full_idx_fp_ub, full_keys_ub, logical_fp)
                    T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_FLOOR", topk)
                    T.tile.cast(work0_fp_ub, full_off_ub, "CAST_NONE", topk)
                    T.tile.mul(full_idx_fp_ub, work0_fp_ub, logical_fp)
                    T.tile.sub(full_keys_ub, full_keys_ub, full_idx_fp_ub)
                    T.tile.div(full_idx_fp_ub, full_keys_ub, physical_fp)
                    T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_FLOOR", topk)
                    T.tile.cast(full_idx_fp_ub, full_off_ub, "CAST_NONE", topk)
                    T.tile.compare(
                        full_owner_mask,
                        full_idx_fp_ub,
                        T.Cast("float32", shard_rank),
                        "EQ",
                    )
                    T.tile.mul(full_out_fp_ub, full_idx_fp_ub, physical_fp)
                    T.tile.sub(full_keys_ub, full_keys_ub, full_out_fp_ub)
                    T.tile.mul(full_out_fp_ub, work0_fp_ub, physical_fp)
                    T.tile.add(full_fp_ub, full_out_fp_ub, full_keys_ub)
                    T.tile.select(
                        full_fp_ub,
                        full_ge_mask,
                        full_fp_ub,
                        T.float32(-1.0),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.tile.select(
                        full_fp_ub,
                        full_owner_mask,
                        full_fp_ub,
                        T.float32(-1.0),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.tile.compare(full_ge_mask, full_fp_ub, T.float32(0.0), "GE")
                    T.tile.arith_progression(
                        full_keys_ub,
                        T.float32(topk),
                        T.float32(-1.0),
                        topk,
                    )
                    T.tile.select(
                        full_keys_ub,
                        full_ge_mask,
                        full_keys_ub,
                        T.float32(neg_inf),
                        "VSEL_TENSOR_SCALAR_MODE",
                    )
                    T.tile.sort(full_sort_ub, full_keys_ub, topk)
                    T.tile.gather_mask(full_idx_fp_ub, full_sort_ub, "P1010")
                    T.tile.cast(full_off_ub, full_idx_fp_ub, "CAST_RINT", topk)
                    T.tile.mul(full_off_ub, full_off_ub, T.int32(gather_elem_bytes))
                    idx_base = token * topk
                    T.set_flag("v", "mte3", REMAP_EVT_IDX)
                    T.wait_flag("v", "mte3", REMAP_EVT_IDX)
                    T.copy(full_off_ub, idx_scratch[idx_base])
                    T.set_flag("mte3", "mte2", REMAP_EVT_IDX)
                    T.wait_flag("mte3", "mte2", REMAP_EVT_IDX)
                    T.copy(idx_scratch_u[idx_base], full_idx_u_ub)
                    T.set_flag("mte2", "v", REMAP_EVT_IDX)
                    T.wait_flag("mte2", "v", REMAP_EVT_IDX)
                    T.tile.gather(full_out_fp_ub, full_fp_ub, full_idx_u_ub, 0)
                    T.tile.cast(full_src_ub, full_out_fp_ub, "CAST_RINT", topk)
                    T.set_flag("v", "mte3", REMAP_EVT_OUT)
                    T.wait_flag("v", "mte3", REMAP_EVT_OUT)
                    T.copy(full_src_ub, topk_out[src_off])
                    T.set_flag("mte3", "mte2", REMAP_EVT_OUT)
                    T.set_flag("mte3", "v", REMAP_EVT_OUT)
                with T.If(token_count > 0), T.Then():
                    T.wait_flag("mte3", "v", REMAP_EVT_OUT)
                    T.wait_flag("mte3", "mte2", REMAP_EVT_OUT)

    return remap_topk_kernel
