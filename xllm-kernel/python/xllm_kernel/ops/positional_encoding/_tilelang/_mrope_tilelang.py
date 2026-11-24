#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
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

"""TileLang DSL for fused Q/K normalization and partial mRoPE."""

import enum

import tilelang.language as T
from tilelang import tvm

from xllm_kernel.ops.positional_encoding._tilelang._mrope_geometry import (
    COMPILE_MAX_TOKENS,
    SUPPORTED_HEAD_SPECS,
    VEC_NUM,
)


class SyncEvent(enum.IntEnum):
    """Event IDs for manual pipeline sync in the fused kernel."""

    # MTE2 -> V
    INIT_WEIGHTS = 1
    COS_SIN_AXES = 3
    HEAD_ROW_Q = 4
    HEAD_ROW_K = 5

    # V -> MTE2
    AXES_FREE = 2

    # V -> MTE3
    Q_CAST = 6
    K_CAST = 7

    # MTE3 -> MTE2 (per-buffer free signals)
    Q_FREE = 0
    G_FREE = 1
    K_FREE = 2
    V_FREE = 3

    # MTE2 -> MTE3
    GATE_READY = 3
    V_READY = 4


KERNEL_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": True,
    "tl.ascend_auto_cv_combine": True,
}


def _select_task_num(*, max_num_tokens: int, vec_core_num: int) -> int:
    raw = min(max_num_tokens, vec_core_num)
    aligned = ((raw + VEC_NUM - 1) // VEC_NUM) * VEC_NUM
    return min(aligned, vec_core_num)


def _validate_head_spec(head_size: int, rope_dim: int) -> None:
    if (head_size, rope_dim) not in SUPPORTED_HEAD_SPECS:
        raise ValueError(
            "split_qkv_rmsnorm_mrope only supports "
            f"{SUPPORTED_HEAD_SPECS}, got head_size={head_size}, rope_dim={rope_dim}"
        )
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim({rope_dim}) must be even")
    if rope_dim > head_size:
        raise ValueError(f"rope_dim({rope_dim}) must be <= head_size({head_size})")


def build_split_qkv_rmsnorm_mrope_kernel(
    *,
    head_size: int,
    rope_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    vec_core_num: int,
    max_num_tokens: int = COMPILE_MAX_TOKENS,
    dynamic_launch: bool = False,
) -> tvm.tir.PrimFunc:
    _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    if vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be divisible by VEC_NUM({VEC_NUM})")
    if num_q_heads <= 0:
        raise ValueError(f"num_q_heads({num_q_heads}) must be > 0")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads({num_kv_heads}) must be > 0")

    half_rope_dim = rope_dim // 2
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    gate_size = q_size
    qkv_stride = q_size * 2 + kv_size * 2
    qkv_head_slots = num_q_heads * 2 + num_kv_heads * 2

    acc_dtype = "float32"
    input_dtype = "bfloat16"
    task_num = _select_task_num(max_num_tokens=max_num_tokens, vec_core_num=vec_core_num)
    m_num = task_num // VEC_NUM
    gather_pad_dim = ((3 * rope_dim + 127) // 128) * 128
    E = SyncEvent

    @T.macro
    def batch_rmsnorm(
        heads_fp32_ub,
        heads_half_ub,
        square_ub,
        rms_vec_ub,
        rms_vec_2d_ub,
        weight_2d_fp32_ub,
        num_heads,
        eps,
    ):
        T.tile.cast(heads_fp32_ub, heads_half_ub, "CAST_NONE", num_heads * head_size)
        T.tile.mul(square_ub, heads_fp32_ub, heads_fp32_ub)
        T.reduce_sum(square_ub, rms_vec_ub, dim=-1)
        T.tile.mul(rms_vec_ub, rms_vec_ub, 1.0 / head_size)
        T.tile.add(rms_vec_ub, rms_vec_ub, eps)
        T.tile.rsqrt(rms_vec_ub, rms_vec_ub)
        T.tile.broadcast(rms_vec_2d_ub, rms_vec_ub)
        T.tile.mul(heads_fp32_ub, heads_fp32_ub, rms_vec_2d_ub)
        T.tile.mul(heads_fp32_ub, heads_fp32_ub, weight_2d_fp32_ub)

    @T.macro
    def assemble_cos_sin_v(
        axes_ub,
        gather_offset_ub,
        gathered_ub,
        assembled_cos_sin_ub,
        cos_full_ub,
        sin_signed_ub,
        sin_neg_tmp_ub,
    ):
        """V-pipe: gather from axes_ub then construct cos/sin."""
        T.wait_flag("mte2", "v", E.COS_SIN_AXES)
        T.tile.gather(gathered_ub, axes_ub, gather_offset_ub, 0)
        T.tile.cast(assembled_cos_sin_ub, gathered_ub, "CAST_NONE", rope_dim)
        # cos_full = [cos, cos]
        T.copy(assembled_cos_sin_ub[0, 0], cos_full_ub[0, 0:half_rope_dim])
        T.copy(
            assembled_cos_sin_ub[0, 0],
            cos_full_ub[0, half_rope_dim:rope_dim],
        )
        # sin_signed = [-sin, sin]
        T.copy(
            assembled_cos_sin_ub[0, half_rope_dim],
            sin_signed_ub[0, half_rope_dim:rope_dim],
        )
        T.copy(assembled_cos_sin_ub[0, half_rope_dim], sin_neg_tmp_ub[0, :])
        T.tile.mul(sin_neg_tmp_ub, sin_neg_tmp_ub, -1.0)
        T.copy(sin_neg_tmp_ub[0, :], sin_signed_ub[0, 0:half_rope_dim])

    @T.macro
    def apply_mrope_batch(
        heads_fp32_ub,
        num_heads,
        cos_full_ub,
        sin_signed_ub,
        rope_orig_ub,
        rope_swapped_ub,
        rope_cos_2d_ub,
        rope_sin_2d_ub,
    ):
        T.tile.broadcast(rope_cos_2d_ub, cos_full_ub)
        T.tile.broadcast(rope_sin_2d_ub, sin_signed_ub)
        for head_idx in T.serial(num_heads):
            T.copy(heads_fp32_ub[head_idx, 0], rope_orig_ub[head_idx, :])
            T.copy(
                heads_fp32_ub[head_idx, half_rope_dim],
                rope_swapped_ub[head_idx, 0:half_rope_dim],
            )
            T.copy(
                heads_fp32_ub[head_idx, 0],
                rope_swapped_ub[head_idx, half_rope_dim:rope_dim],
            )
        T.tile.mul(rope_orig_ub, rope_orig_ub, rope_cos_2d_ub)
        T.tile.mul(rope_swapped_ub, rope_swapped_ub, rope_sin_2d_ub)
        T.tile.add(rope_orig_ub, rope_orig_ub, rope_swapped_ub)
        for head_idx in T.serial(num_heads):
            T.copy(rope_orig_ub[head_idx, :], heads_fp32_ub[head_idx, 0])

    @T.prim_func
    def split_qkv_rmsnorm_mrope_kernel(
        # Redeclare the flat QKVG input row as (total_heads, head_size) so
        # T.copy can issue per-tensor bulk DMA instead of treating the first
        # UB dimension as a cross-token repeat.
        qkvg_in: T.Tensor((COMPILE_MAX_TOKENS, qkv_head_slots, head_size), input_dtype),
        q_weight: T.Tensor((1, head_size), input_dtype),
        k_weight: T.Tensor((1, head_size), input_dtype),
        cos_sin: T.Tensor((COMPILE_MAX_TOKENS, 3 * rope_dim), input_dtype),
        gather_pattern: T.Tensor((gather_pad_dim,), "uint32"),
        q_out: T.Tensor((COMPILE_MAX_TOKENS, num_q_heads, head_size), input_dtype),
        k_out: T.Tensor((COMPILE_MAX_TOKENS, num_kv_heads, head_size), input_dtype),
        v_out: T.Tensor((COMPILE_MAX_TOKENS, num_kv_heads, head_size), input_dtype),
        gate_out: T.Tensor((COMPILE_MAX_TOKENS, num_q_heads, head_size), input_dtype),
        num_tokens: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(T.min(m_num, T.ceildiv(num_tokens, VEC_NUM)) if dynamic_launch else m_num, is_npu=True) as (
            cid,
            vid,
        ):
            task_id = cid * VEC_NUM + vid
            active_tasks = T.min(task_num, T.ceildiv(num_tokens, VEC_NUM) * VEC_NUM) if dynamic_launch else task_num
            block_m = (num_tokens + active_tasks - 1) // active_tasks
            row_start = task_id * block_m
            rows_left = T.if_then_else(num_tokens > row_start, num_tokens - row_start, 0)
            num_rows_per_vec = T.if_then_else(
                rows_left < block_m,
                rows_left,
                block_m,
            )

            with T.Scope("V"):
                q_weight_half_ub = T.alloc_shared((1, head_size), input_dtype)
                q_weight_fp32_ub = T.alloc_shared((1, head_size), acc_dtype)
                q_weight_2d_fp32_ub = T.alloc_shared((num_q_heads, head_size), acc_dtype)
                k_weight_half_ub = T.alloc_shared((1, head_size), input_dtype)
                k_weight_fp32_ub = T.alloc_shared((1, head_size), acc_dtype)
                k_weight_2d_fp32_ub = T.alloc_shared((num_kv_heads, head_size), acc_dtype)

                q_heads_half_ub = T.alloc_shared((num_q_heads, head_size), input_dtype)
                q_heads_fp32_ub = T.alloc_shared((num_q_heads, head_size), acc_dtype)
                q_square_ub = T.alloc_shared((num_q_heads, head_size), acc_dtype)
                q_rms_vec_ub = T.alloc_shared((num_q_heads, 1), acc_dtype)
                q_rms_vec_2d_ub = T.alloc_shared((num_q_heads, head_size), acc_dtype)
                k_heads_half_ub = T.alloc_shared((num_kv_heads, head_size), input_dtype)
                k_heads_fp32_ub = T.alloc_shared((num_kv_heads, head_size), acc_dtype)
                k_square_ub = T.alloc_shared((num_kv_heads, head_size), acc_dtype)
                k_rms_vec_ub = T.alloc_shared((num_kv_heads, 1), acc_dtype)
                k_rms_vec_2d_ub = T.alloc_shared((num_kv_heads, head_size), acc_dtype)
                # Separate passthrough buffers for gate and V. Keeping these
                # independent from q/k_heads_half_ub lets MTE2 issue their
                # loads right after Q/K loads (while V is still doing Q/K
                # compute), so the passthrough DMA is fully hidden behind
                # V-pipe work.
                gate_heads_half_ub = T.alloc_shared((num_q_heads, head_size), input_dtype)
                v_heads_half_ub = T.alloc_shared((num_kv_heads, head_size), input_dtype)

                # Gather buffers for merged cos+sin assembly.
                axes_ub = T.alloc_shared((1, gather_pad_dim), input_dtype)
                gathered_ub = T.alloc_shared((1, gather_pad_dim), input_dtype)
                assembled_cos_sin_ub = T.alloc_shared((1, rope_dim), acc_dtype)
                gather_offset_ub = T.alloc_shared((gather_pad_dim,), "uint32")
                cos_full_ub = T.alloc_shared((1, rope_dim), acc_dtype)
                sin_signed_ub = T.alloc_shared((1, rope_dim), acc_dtype)
                sin_neg_tmp_ub = T.alloc_shared((1, half_rope_dim), acc_dtype)
                q_rope_orig_ub = T.alloc_shared((num_q_heads, rope_dim), acc_dtype)
                q_rope_swapped_ub = T.alloc_shared((num_q_heads, rope_dim), acc_dtype)
                q_rope_cos_2d_ub = T.alloc_shared((num_q_heads, rope_dim), acc_dtype)
                q_rope_sin_2d_ub = T.alloc_shared((num_q_heads, rope_dim), acc_dtype)

                k_rope_orig_ub = T.alloc_shared((num_kv_heads, rope_dim), acc_dtype)
                k_rope_swapped_ub = T.alloc_shared((num_kv_heads, rope_dim), acc_dtype)
                k_rope_cos_2d_ub = T.alloc_shared((num_kv_heads, rope_dim), acc_dtype)
                k_rope_sin_2d_ub = T.alloc_shared((num_kv_heads, rope_dim), acc_dtype)

                # Init: load gather pattern, weights.
                T.copy(gather_pattern, gather_offset_ub)
                T.copy(q_weight[0, 0], q_weight_half_ub[0, :])
                T.copy(k_weight[0, 0], k_weight_half_ub[0, :])
                T.set_flag("mte2", "v", E.INIT_WEIGHTS)
                T.wait_flag("mte2", "v", E.INIT_WEIGHTS)
                T.tile.cast(q_weight_fp32_ub, q_weight_half_ub, "CAST_NONE", head_size)
                T.tile.cast(k_weight_fp32_ub, k_weight_half_ub, "CAST_NONE", head_size)
                T.tile.broadcast(q_weight_2d_fp32_ub, q_weight_fp32_ub)
                T.tile.broadcast(k_weight_2d_fp32_ub, k_weight_fp32_ub)

                for row_local in T.serial(num_rows_per_vec):
                    row = row_start + row_local
                    # === Load phase: cos_sin load needs axes_ub free (skip
                    # wait on first iter since axes_ub starts free).
                    with T.If(row_local > 0), T.Then():
                        T.wait_flag("v", "mte2", E.AXES_FREE)
                    T.copy(cos_sin[row, 0], axes_ub[0, 0 : 3 * rope_dim])
                    T.set_flag("mte2", "v", E.COS_SIN_AXES)

                    # Load/store order: Q→G→K→V (matches [Q|G|K|V] layout).

                    # load q
                    with T.If(row_local > 0), T.Then():
                        T.wait_flag("mte3", "mte2", E.Q_FREE)
                    T.copy(qkvg_in[row, 0:num_q_heads, 0:head_size], q_heads_half_ub)
                    T.set_flag("mte2", "v", E.HEAD_ROW_Q)
                    # load g
                    with T.If(row_local > 0), T.Then():
                        T.wait_flag("mte3", "mte2", E.G_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads : num_q_heads * 2,
                            0:head_size,
                        ],
                        gate_heads_half_ub,
                    )
                    T.set_flag("mte2", "mte3", E.GATE_READY)

                    # load k
                    with T.If(row_local > 0), T.Then():
                        T.wait_flag("mte3", "mte2", E.K_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads * 2 : num_q_heads * 2 + num_kv_heads,
                            0:head_size,
                        ],
                        k_heads_half_ub,
                    )
                    T.set_flag("mte2", "v", E.HEAD_ROW_K)

                    # load v
                    with T.If(row_local > 0), T.Then():
                        T.wait_flag("mte3", "mte2", E.V_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads * 2 + num_kv_heads : qkv_head_slots,
                            0:head_size,
                        ],
                        v_heads_half_ub,
                    )
                    T.set_flag("mte2", "mte3", E.V_READY)

                    # === cos/sin assembly (V pipe): gather + release axes_ub,
                    # then construct cos/sin vectors for rope. axes_ub is freed
                    # right after gather so MTE2 can start next row's cos_sin
                    # load while V is still computing rope.
                    assemble_cos_sin_v(
                        axes_ub,
                        gather_offset_ub,
                        gathered_ub,
                        assembled_cos_sin_ub,
                        cos_full_ub,
                        sin_signed_ub,
                        sin_neg_tmp_ub,
                    )
                    # Release axes_ub after gather (skip on last iter —
                    # keeps SetFlag/WaitFlag counts balanced).
                    with T.If(row_local < num_rows_per_vec - 1), T.Then():
                        T.set_flag("v", "mte2", E.AXES_FREE)

                    # === Q compute (V pipe) ===
                    T.wait_flag("mte2", "v", E.HEAD_ROW_Q)
                    batch_rmsnorm(
                        q_heads_fp32_ub,
                        q_heads_half_ub,
                        q_square_ub,
                        q_rms_vec_ub,
                        q_rms_vec_2d_ub,
                        q_weight_2d_fp32_ub,
                        num_q_heads,
                        eps,
                    )
                    apply_mrope_batch(
                        q_heads_fp32_ub,
                        num_q_heads,
                        cos_full_ub,
                        sin_signed_ub,
                        q_rope_orig_ub,
                        q_rope_swapped_ub,
                        q_rope_cos_2d_ub,
                        q_rope_sin_2d_ub,
                    )
                    T.tile.cast(q_heads_half_ub, q_heads_fp32_ub, "CAST_RINT", q_size)
                    T.set_flag("v", "mte3", E.Q_CAST)

                    # === K compute (V pipe) ===
                    T.wait_flag("mte2", "v", E.HEAD_ROW_K)
                    batch_rmsnorm(
                        k_heads_fp32_ub,
                        k_heads_half_ub,
                        k_square_ub,
                        k_rms_vec_ub,
                        k_rms_vec_2d_ub,
                        k_weight_2d_fp32_ub,
                        num_kv_heads,
                        eps,
                    )
                    apply_mrope_batch(
                        k_heads_fp32_ub,
                        num_kv_heads,
                        cos_full_ub,
                        sin_signed_ub,
                        k_rope_orig_ub,
                        k_rope_swapped_ub,
                        k_rope_cos_2d_ub,
                        k_rope_sin_2d_ub,
                    )
                    T.tile.cast(k_heads_half_ub, k_heads_fp32_ub, "CAST_RINT", kv_size)
                    T.set_flag("v", "mte3", E.K_CAST)

                    # === Store phase (MTE3 pipe, Q→G→K→V order matching
                    # load order). Each store signals buffer free immediately.
                    T.wait_flag("v", "mte3", E.Q_CAST)
                    T.copy(q_heads_half_ub, q_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1), T.Then():
                        T.set_flag("mte3", "mte2", E.Q_FREE)

                    T.wait_flag("mte2", "mte3", E.GATE_READY)
                    T.copy(gate_heads_half_ub, gate_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1), T.Then():
                        T.set_flag("mte3", "mte2", E.G_FREE)

                    T.wait_flag("v", "mte3", E.K_CAST)
                    T.copy(k_heads_half_ub, k_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1), T.Then():
                        T.set_flag("mte3", "mte2", E.K_FREE)

                    T.wait_flag("mte2", "mte3", E.V_READY)
                    T.copy(v_heads_half_ub, v_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1), T.Then():
                        T.set_flag("mte3", "mte2", E.V_FREE)

    return split_qkv_rmsnorm_mrope_kernel
