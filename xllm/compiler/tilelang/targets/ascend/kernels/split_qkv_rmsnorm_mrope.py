#!/usr/bin/env python3

# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import enum
from pathlib import Path

import tilelang
import tilelang.language as T

from scripts.logger import logger

from compiler.tilelang.common.spec import (
    DispatchField,
    TilelangKernel,
    register_kernel,
)
from compiler.tilelang.targets.ascend.kernels.utils import (
    detect_vec_core_num,
    mte2_notify_mte3,
    mte2_notify_v,
    mte2_wait_mte3,
    mte2_wait_v,
    mte3_notify_mte2,
    mte3_notify_v,
    mte3_wait_mte2,
    mte3_wait_v,
    v_notify_mte2,
    v_notify_mte3,
    v_wait_mte2,
    v_wait_mte3,
)


COS_SIN_MERGED_LAYOUT = "token_3rope"
DEFAULT_DTYPE = "bf16"
SUPPORTED_HEAD_SPECS = (
    (256, 64),
)
COMPILE_MAX_TOKENS = 100_000_000
VEC_NUM = 2
MAX_VEC_CORE_NUM = detect_vec_core_num()
MAX_LAUNCH_NUM_TOKENS = (MAX_VEC_CORE_NUM // VEC_NUM) * VEC_NUM
NUM_TOKEN_SPECIALIZATIONS = tuple(
    range(VEC_NUM, MAX_LAUNCH_NUM_TOKENS + 1, VEC_NUM)
)
REF_CHECK_NUM_TOKENS = 16
REF_CHECK_EPS = 1e-6
DEFAULT_MROPE_SECTION = (11, 11, 10)
DEFAULT_NUM_Q_HEADS = 16
DEFAULT_NUM_KV_HEADS = 4

# All Qwen3.5/3.6 model original head configurations:
#   Model              num_attention_heads  num_key_value_heads
#   Qwen3.5-0.8B/2B          8                    2
#   Qwen3.5-4B/9B           16                    4
#   Qwen3.5-27B             24                    4
#   Qwen3.5-35B-A3B         16                    2
#   Qwen3.5-122B/397B       32                    2
MODEL_HEAD_CONFIGS: tuple[tuple[int, int], ...] = (
    (8, 2),
    (16, 4),
    (24, 4),
    (16, 2),
    (32, 2),
)

SUPPORTED_TP_SIZES = (1, 2, 4, 8, 16)


def compute_tp_split_head_configs(
    model_head_configs: tuple[tuple[int, int], ...],
    tp_sizes: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Compute deduplicated (q_heads, kv_heads) after TP split.

    For each (total_q, total_kv) and each tp:
      - q_heads = total_q / tp  (skip if not divisible)
      - kv_heads = total_kv / tp if total_kv >= tp,
                   else 1 if tp % total_kv == 0
    """
    seen: set[tuple[int, int]] = set()
    result: list[tuple[int, int]] = []
    for total_q, total_kv in model_head_configs:
        for tp in tp_sizes:
            if total_q % tp != 0:
                continue
            q_heads = total_q // tp
            if total_kv >= tp:
                if total_kv % tp != 0:
                    continue
                kv_heads = total_kv // tp
            else:
                if tp % total_kv != 0:
                    continue
                kv_heads = 1
            pair = (q_heads, kv_heads)
            if pair not in seen:
                seen.add(pair)
                result.append(pair)
    result.sort(key=lambda p: (-p[0], -p[1]))
    return result


# Deduplicated TP-split shapes:
#   0.8B/2B   (8,2):  tp1=(8,2)  tp2=(4,1)  tp4=(2,1)
#   4B/9B    (16,4):  tp1=(16,4) tp2=(8,2)  tp4=(4,1)  tp8=(2,1)
#   27B      (24,4):  tp1=(24,4) tp2=(12,2) tp4=(6,1)  tp8=(3,1)
#   35B-A3B  (16,2):  tp1=(16,2) tp2=(8,1)  tp4=(4,1)  tp8=(2,1)
#   122B/397B(32,2):  tp1=(32,2) tp2=(16,1) tp4=(8,1)  tp8=(4,1) tp16=(2,1)
ALL_HEAD_CONFIGS = compute_tp_split_head_configs(MODEL_HEAD_CONFIGS, SUPPORTED_TP_SIZES)
REF_CHECK_HEAD_CONFIGS = tuple(ALL_HEAD_CONFIGS)



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
    K_FREE = 9
    V_FREE = 10

    # MTE2 -> MTE3
    GATE_READY = 3
    V_READY = 8


KERNEL_PASS_CONFIGS = {
    "tl.ascend_auto_sync": True,
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
        raise ValueError(
            f"rope_dim({rope_dim}) must be <= head_size({head_size})"
        )



def build_split_qkv_rmsnorm_mrope_kernel(
    *,
    head_size: int,
    rope_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    vec_core_num: int,
    max_num_tokens: int = COMPILE_MAX_TOKENS,
):
    _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    if vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num({vec_core_num}) must be divisible by VEC_NUM({VEC_NUM})"
        )
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
    task_num = _select_task_num(
        max_num_tokens=max_num_tokens, vec_core_num=vec_core_num
    )
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
        gather_index_ub,
        gathered_ub,
        assembled_cos_sin_ub,
        cos_full_ub,
        sin_signed_ub,
        sin_neg_tmp_ub,
    ):
        """V-pipe: gather from axes_ub then construct cos/sin."""
        v_wait_mte2(E.COS_SIN_AXES)
        T.tile.gather_mask(gathered_ub, axes_ub, gather_index_ub)
        T.tile.cast(
            assembled_cos_sin_ub, gathered_ub, "CAST_NONE", rope_dim
        )
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
        # Re-declare the flat QKVG input row as (total_heads, head_size) so
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
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            block_m = (num_tokens + task_num - 1) // task_num
            row_start = task_id * block_m
            rows_left = T.if_then_else(
                num_tokens > row_start, num_tokens - row_start, 0
            )
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
                gate_heads_half_ub = T.alloc_shared(
                    (num_q_heads, head_size), input_dtype
                )
                v_heads_half_ub = T.alloc_shared(
                    (num_kv_heads, head_size), input_dtype
                )

                # Gather buffers for merged cos+sin assembly.
                axes_ub = T.alloc_shared((1, gather_pad_dim), input_dtype)
                gathered_ub = T.alloc_shared((1, gather_pad_dim), input_dtype)
                assembled_cos_sin_ub = T.alloc_shared((1, rope_dim), acc_dtype)
                gather_index_ub = T.alloc_shared((1, gather_pad_dim), "uint32")
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
                T.copy(gather_pattern, gather_index_ub[0, :])
                T.tile.fill(axes_ub, 0.0)
                T.copy(q_weight[0, 0], q_weight_half_ub[0, :])
                T.copy(k_weight[0, 0], k_weight_half_ub[0, :])
                mte2_notify_v(E.INIT_WEIGHTS)
                v_wait_mte2(E.INIT_WEIGHTS)
                T.tile.cast(q_weight_fp32_ub, q_weight_half_ub, "CAST_NONE", head_size)
                T.tile.cast(k_weight_fp32_ub, k_weight_half_ub, "CAST_NONE", head_size)
                T.tile.broadcast(q_weight_2d_fp32_ub, q_weight_fp32_ub)
                T.tile.broadcast(k_weight_2d_fp32_ub, k_weight_fp32_ub)

                for row_local in T.serial(num_rows_per_vec):
                    row = row_start + row_local
                    # === Load phase: cos_sin load needs axes_ub free (skip
                    # wait on first iter since axes_ub starts free).
                    with T.If(row_local > 0):
                        with T.Then():
                            mte2_wait_v(E.AXES_FREE)
                    T.copy(cos_sin[row, 0], axes_ub[0, 0:3 * rope_dim])
                    mte2_notify_v(E.COS_SIN_AXES)

                    # Load/store order: Q→G→K→V (matches [Q|G|K|V] layout).

                    # load q
                    with T.If(row_local > 0):
                        with T.Then():
                            mte2_wait_mte3(E.Q_FREE)
                    T.copy(qkvg_in[row, 0:num_q_heads, 0:head_size], q_heads_half_ub)
                    mte2_notify_v(E.HEAD_ROW_Q)
                    # load g
                    with T.If(row_local > 0):
                        with T.Then():
                            mte2_wait_mte3(E.G_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads : num_q_heads * 2,
                            0:head_size,
                        ],
                        gate_heads_half_ub,
                    )
                    mte2_notify_mte3(E.GATE_READY)

                    # load k
                    with T.If(row_local > 0):
                        with T.Then():
                            mte2_wait_mte3(E.K_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads * 2 : num_q_heads * 2 + num_kv_heads,
                            0:head_size,
                        ],
                        k_heads_half_ub,
                    )
                    mte2_notify_v(E.HEAD_ROW_K)

                    # load v
                    with T.If(row_local > 0):
                        with T.Then():
                            mte2_wait_mte3(E.V_FREE)
                    T.copy(
                        qkvg_in[
                            row,
                            num_q_heads * 2 + num_kv_heads : qkv_head_slots,
                            0:head_size,
                        ],
                        v_heads_half_ub,
                    )
                    mte2_notify_mte3(E.V_READY)

                    # === cos/sin assembly (V pipe): gather + release axes_ub,
                    # then construct cos/sin vectors for rope. axes_ub is freed
                    # right after gather so MTE2 can start next row's cos_sin
                    # load while V is still computing rope.
                    assemble_cos_sin_v(
                        axes_ub,
                        gather_index_ub,
                        gathered_ub,
                        assembled_cos_sin_ub,
                        cos_full_ub,
                        sin_signed_ub,
                        sin_neg_tmp_ub,
                    )
                    # Release axes_ub after gather (skip on last iter —
                    # keeps SetFlag/WaitFlag counts balanced).
                    with T.If(row_local < num_rows_per_vec - 1):
                        with T.Then():
                            v_notify_mte2(E.AXES_FREE)

                    # === Q compute (V pipe) ===
                    v_wait_mte2(E.HEAD_ROW_Q)
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
                    v_notify_mte3(E.Q_CAST)

                    # === K compute (V pipe) ===
                    v_wait_mte2(E.HEAD_ROW_K)
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
                    v_notify_mte3(E.K_CAST)

                    # === Store phase (MTE3 pipe, Q→G→K→V order matching
                    # load order). Each store signals buffer free immediately.
                    mte3_wait_v(E.Q_CAST)
                    T.copy(q_heads_half_ub, q_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1):
                        with T.Then():
                            mte3_notify_mte2(E.Q_FREE)

                    mte3_wait_mte2(E.GATE_READY)
                    T.copy(gate_heads_half_ub, gate_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1):
                        with T.Then():
                            mte3_notify_mte2(E.G_FREE)

                    mte3_wait_v(E.K_CAST)
                    T.copy(k_heads_half_ub, k_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1):
                        with T.Then():
                            mte3_notify_mte2(E.K_FREE)

                    mte3_wait_mte2(E.V_READY)
                    T.copy(v_heads_half_ub, v_out[row, 0, 0])
                    with T.If(row_local < num_rows_per_vec - 1):
                        with T.Then():
                            mte3_notify_mte2(E.V_FREE)

    return split_qkv_rmsnorm_mrope_kernel


@tilelang.jit(pass_configs=KERNEL_PASS_CONFIGS, target="pto")
def split_qkv_rmsnorm_mrope_kernel_jit(
    head_size: int,
    rope_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    vec_core_num: int,
    max_num_tokens: int = COMPILE_MAX_TOKENS,
):
    return build_split_qkv_rmsnorm_mrope_kernel(
        head_size=head_size,
        rope_dim=rope_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vec_core_num=vec_core_num,
        max_num_tokens=max_num_tokens,
    )


def _validate_specialization_num_tokens(num_tokens: int) -> None:
    if num_tokens not in NUM_TOKEN_SPECIALIZATIONS:
        raise ValueError(
            "split_qkv_rmsnorm_mrope only supports specialization num_tokens in "
            f"{NUM_TOKEN_SPECIALIZATIONS}, got {num_tokens}"
        )

@register_kernel
class SplitQkvRmsnormMropeKernel(TilelangKernel):
    KERNEL_NAME = "split_qkv_rmsnorm_mrope"
    DISPATCH_SCHEMA = [
        DispatchField("head_size", "int32"),
        DispatchField("rope_dim", "int32"),
        DispatchField("num_tokens", "int32"),
        DispatchField("num_q_heads", "int32"),
        DispatchField("num_kv_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": (
                f"hs{head_size}_rd{rope_dim}_nt{num_tokens}_"
                f"qh{num_q_heads}_kvh{num_kv_heads}_bf16"
            ),
            "head_size": head_size,
            "rope_dim": rope_dim,
            "num_tokens": num_tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "dtype": DEFAULT_DTYPE,
        }
        for head_size, rope_dim in SUPPORTED_HEAD_SPECS
        for num_tokens in NUM_TOKEN_SPECIALIZATIONS
        for num_q_heads, num_kv_heads in ALL_HEAD_CONFIGS
    ]

    @staticmethod
    def generate_source(
        head_size: int,
        rope_dim: int,
        num_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        dtype: str,
    ) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(
                "split_qkv_rmsnorm_mrope only supports "
                f"dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
        _validate_specialization_num_tokens(num_tokens)
        tilelang.disable_cache()
        vec_core_num = detect_vec_core_num()
        tilelang_kernel = build_split_qkv_rmsnorm_mrope_kernel(
            head_size=head_size,
            rope_dim=rope_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            vec_core_num=vec_core_num,
            max_num_tokens=num_tokens,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=KERNEL_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel, target="pto")
        return kernel.kernel_source



def _validate_mrope_section(
    *,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
) -> None:
    if len(mrope_section) != 3:
        raise ValueError(
            f"mrope_section must have 3 items [t, h, w], got {mrope_section}"
        )
    if any(section < 0 for section in mrope_section):
        raise ValueError(f"mrope_section values must be >= 0, got {mrope_section}")
    if sum(mrope_section) != rope_dim // 2:
        raise ValueError(
            "sum(mrope_section) must equal rope_dim // 2, got "
            f"mrope_section={mrope_section}, rope_dim={rope_dim}"
        )


def _validate_runtime_config(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    del is_interleaved
    _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens({num_tokens}) must be > 0")
    if num_tokens > COMPILE_MAX_TOKENS:
        raise ValueError(
            "num_tokens("
            f"{num_tokens}) must be <= COMPILE_MAX_TOKENS({COMPILE_MAX_TOKENS})"
        )
    if num_q_heads <= 0:
        raise ValueError(f"num_q_heads({num_q_heads}) must be > 0")
    if num_kv_heads <= 0:
        raise ValueError(f"num_kv_heads({num_kv_heads}) must be > 0")
    _validate_mrope_section(rope_dim=rope_dim, mrope_section=mrope_section)





def _torch_rms_norm(
    x: "torch.Tensor",
    weight: "torch.Tensor",
    eps: float,
) -> "torch.Tensor":
    import torch

    x_fp32 = x.to(torch.float32)
    weight_fp32 = weight.to(torch.float32)
    reciprocal_std = torch.rsqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x_fp32 * reciprocal_std * weight_fp32


def _assemble_non_interleaved_mrope_rows(
    cos_sin: "torch.Tensor",
    mrope_section: tuple[int, int, int],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    half_rope_dim = cos_sin.shape[-1] // 2
    t_len, h_len, w_len = mrope_section
    h_end = t_len + h_len
    w_end = h_end + w_len

    cos_axes = cos_sin[:, :, :half_rope_dim].to(torch.float32)
    sin_axes = cos_sin[:, :, half_rope_dim:].to(torch.float32)
    num_tokens = cos_sin.shape[1]
    cos_rows = torch.zeros(
        (num_tokens, half_rope_dim), device=cos_sin.device, dtype=torch.float32
    )
    sin_rows = torch.zeros(
        (num_tokens, half_rope_dim), device=cos_sin.device, dtype=torch.float32
    )

    if t_len > 0:
        cos_rows[:, :t_len] = cos_axes[0, :, :t_len]
        sin_rows[:, :t_len] = sin_axes[0, :, :t_len]
    if h_len > 0:
        cos_rows[:, t_len:h_end] = cos_axes[1, :, t_len:h_end]
        sin_rows[:, t_len:h_end] = sin_axes[1, :, t_len:h_end]
    if w_len > 0:
        cos_rows[:, h_end:w_end] = cos_axes[2, :, h_end:w_end]
        sin_rows[:, h_end:w_end] = sin_axes[2, :, h_end:w_end]
    return cos_rows, sin_rows


def _assemble_interleaved_mrope_rows(
    cos_sin: "torch.Tensor",
    mrope_section: tuple[int, int, int],
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch

    half_rope_dim = cos_sin.shape[-1] // 2
    _, h_len, w_len = mrope_section

    cos_axes = cos_sin[:, :, :half_rope_dim].to(torch.float32)
    sin_axes = cos_sin[:, :, half_rope_dim:].to(torch.float32)

    cos_rows = cos_axes[0].clone()
    sin_rows = sin_axes[0].clone()

    cos_rows[:, 1 : h_len * 3 : 3] = cos_axes[1, :, 1 : h_len * 3 : 3]
    sin_rows[:, 1 : h_len * 3 : 3] = sin_axes[1, :, 1 : h_len * 3 : 3]

    cos_rows[:, 2 : w_len * 3 : 3] = cos_axes[2, :, 2 : w_len * 3 : 3]
    sin_rows[:, 2 : w_len * 3 : 3] = sin_axes[2, :, 2 : w_len * 3 : 3]

    return cos_rows, sin_rows


def _apply_partial_mrope(
    x: "torch.Tensor",
    cos_rows: "torch.Tensor",
    sin_rows: "torch.Tensor",
    rope_dim: int,
) -> "torch.Tensor":
    half_rope_dim = rope_dim // 2

    out = x.clone()
    x1 = x[:, :, :half_rope_dim]
    x2 = x[:, :, half_rope_dim:rope_dim]
    cos = cos_rows.unsqueeze(1)
    sin = sin_rows.unsqueeze(1)
    out[:, :, :half_rope_dim] = x1 * cos - x2 * sin
    out[:, :, half_rope_dim:rope_dim] = x2 * cos + x1 * sin
    return out


def _torch_split_qkv_rmsnorm_mrope(
    *,
    qkv: "torch.Tensor",
    q_weight: "torch.Tensor",
    k_weight: "torch.Tensor",
    cos_sin: "torch.Tensor",
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    import torch

    _validate_runtime_config(
        num_tokens=qkv.shape[0],
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size
    q = qkv[:, :q_width].reshape(qkv.shape[0], num_q_heads, head_size)
    gate = qkv[:, q_width : q_width * 2]
    k = qkv[:, q_width * 2 : q_width * 2 + kv_width].reshape(
        qkv.shape[0], num_kv_heads, head_size
    )
    v = qkv[:, q_width * 2 + kv_width : q_width * 2 + kv_width * 2]

    q_norm = _torch_rms_norm(q, q_weight[0], eps)
    k_norm = _torch_rms_norm(k, k_weight[0], eps)
    if is_interleaved:
        cos_rows, sin_rows = _assemble_interleaved_mrope_rows(
            cos_sin, mrope_section
        )
    else:
        cos_rows, sin_rows = _assemble_non_interleaved_mrope_rows(
            cos_sin, mrope_section
        )

    q_out = _apply_partial_mrope(q_norm, cos_rows, sin_rows, rope_dim)
    k_out = _apply_partial_mrope(k_norm, cos_rows, sin_rows, rope_dim)
    return (
        q_out.reshape(qkv.shape[0], q_width).to(torch.bfloat16),
        k_out.reshape(qkv.shape[0], kv_width).to(torch.bfloat16),
        v.to(torch.bfloat16),
        gate.to(torch.bfloat16),
    )


def _parse_head_config(text: str) -> tuple[int, int]:
    q_heads_text, sep, kv_heads_text = text.partition(":")
    if sep != ":":
        raise argparse.ArgumentTypeError(
            f"Invalid head config '{text}'. Expected format <num_q_heads>:<num_kv_heads>."
        )
    try:
        return int(q_heads_text), int(kv_heads_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid head config '{text}'. Expected integer pair."
        ) from exc


def build_mrope_gather_pattern_merged(
    half_rope_dim: int,
    rope_dim: int,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
    gather_pad_dim: int,
    device: str = "cpu",
) -> "torch.Tensor":
    """Build uint32 element-index pattern for merged cos+sin gather."""
    import torch

    t_len, h_len, w_len = mrope_section
    axis_id = [0] * half_rope_dim
    if is_interleaved:
        for i in range(half_rope_dim):
            if i % 3 == 1 and i <= 3 * h_len:
                axis_id[i] = 1
            elif i % 3 == 2 and i <= 3 * w_len:
                axis_id[i] = 2
    else:
        t_end = t_len
        h_end = t_end + h_len
        for i in range(half_rope_dim):
            if i >= h_end:
                axis_id[i] = 2
            elif i >= t_end:
                axis_id[i] = 1

    pattern = torch.zeros(gather_pad_dim, dtype=torch.int32, device="cpu")
    for i in range(half_rope_dim):
        pattern[i] = axis_id[i] * rope_dim + i
        pattern[half_rope_dim + i] = (
            axis_id[i] * rope_dim + half_rope_dim + i
        )

    return pattern.to(device).view(torch.uint32)


def _run_ref_check(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning(
            "Skip split_qkv_rmsnorm_mrope reference check: "
            "NPU is not available"
        )
        return

    _validate_runtime_config(
        num_tokens=num_tokens,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    torch.manual_seed(42)
    device = torch.device("npu")
    half_rope_dim = rope_dim // 2
    q_width = num_q_heads * head_size
    kv_width = num_kv_heads * head_size
    qkv_stride = q_width * 2 + kv_width * 2

    qkv = torch.randn((num_tokens, qkv_stride), device=device, dtype=torch.bfloat16)
    q_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    k_weight = torch.randn((1, head_size), device=device, dtype=torch.bfloat16)
    phase = torch.randn((3, num_tokens, half_rope_dim), device=device)
    cos_sin = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1).to(
        torch.bfloat16
    )

    q_out = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)
    k_out = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
    v_out = torch.empty((num_tokens, kv_width), device=device, dtype=torch.bfloat16)
    gate_out = torch.empty((num_tokens, q_width), device=device, dtype=torch.bfloat16)

    kernel = split_qkv_rmsnorm_mrope_kernel_jit(
        head_size=head_size,
        rope_dim=rope_dim,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        vec_core_num=detect_vec_core_num(),
        max_num_tokens=num_tokens,
    )
    cos_sin_reshaped = cos_sin.permute(1, 0, 2).contiguous().view(num_tokens, 3 * rope_dim)
    gather_pad_dim = ((3 * rope_dim + 127) // 128) * 128
    gather_pat = build_mrope_gather_pattern_merged(
        half_rope_dim=half_rope_dim,
        rope_dim=rope_dim,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
        gather_pad_dim=gather_pad_dim,
        device=str(device),
    )
    kernel(
        qkv,
        q_weight,
        k_weight,
        cos_sin_reshaped,
        gather_pat,
        q_out,
        k_out,
        v_out,
        gate_out,
        num_tokens,
        eps,
    )
    torch.npu.synchronize()

    q_ref, k_ref, v_ref, gate_ref = _torch_split_qkv_rmsnorm_mrope(
        qkv=qkv,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin=cos_sin,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        rope_dim=rope_dim,
        eps=eps,
        mrope_section=mrope_section,
        is_interleaved=is_interleaved,
    )

    torch.testing.assert_close(
        q_out.to(torch.float32),
        q_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(
        k_out.to(torch.float32),
        k_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    torch.testing.assert_close(v_out, v_ref, rtol=0, atol=0)
    torch.testing.assert_close(gate_out, gate_ref, rtol=0, atol=0)
    il_tag = "interleaved" if is_interleaved else "non-interleaved"
    logger.info(
        "split_qkv_rmsnorm_mrope output matches torch reference for "
        f"num_tokens={num_tokens}, num_q_heads={num_q_heads}, "
        f"num_kv_heads={num_kv_heads}, {il_tag}"
    )


def _run_ref_suite(
    *,
    num_tokens: int,
    head_configs: list[tuple[int, int]],
    head_size: int,
    rope_dim: int,
    eps: float,
    mrope_section: tuple[int, int, int],
    is_interleaved: bool,
) -> None:
    for num_q_heads, num_kv_heads in head_configs:
        _run_ref_check(
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            rope_dim=rope_dim,
            eps=eps,
            mrope_section=mrope_section,
            is_interleaved=is_interleaved,
        )


def parse_args() -> argparse.Namespace:
    default_head_size, default_rope_dim = SUPPORTED_HEAD_SPECS[0]
    parser = argparse.ArgumentParser(
        description=(
            "Generate TileLang AscendC source for split_qkv_rmsnorm_mrope "
            "AOT kernel."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-size", type=int, default=default_head_size)
    parser.add_argument("--rope-dim", type=int, default=default_rope_dim)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=NUM_TOKEN_SPECIALIZATIONS[-1],
        help=(
            "Launch token specialization bucket used for source generation. "
            f"Supported values: {NUM_TOKEN_SPECIALIZATIONS}"
        ),
    )
    parser.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--eps", type=float, default=REF_CHECK_EPS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-tokens",
        type=int,
        default=REF_CHECK_NUM_TOKENS,
        help="Token count used by the optional torch-reference test suite.",
    )
    parser.add_argument(
        "--ref-head-configs",
        type=_parse_head_config,
        nargs="+",
        default=list(REF_CHECK_HEAD_CONFIGS),
        help=(
            "Head configurations covered by the optional runtime check. "
            "Each item must use <num_q_heads>:<num_kv_heads>."
        ),
    )
    parser.add_argument(
        "--mrope-section",
        type=int,
        nargs=3,
        default=list(DEFAULT_MROPE_SECTION),
        metavar=("T", "H", "W"),
        help="MRoPE section [t h w] used by the optional test.",
    )
    parser.add_argument(
        "--is-interleaved",
        action="store_true",
        default=True,
        help="Use interleaved MRoPE layout (default: True for Qwen3.5).",
    )
    parser.add_argument(
        "--no-interleaved",
        dest="is_interleaved",
        action="store_false",
        help="Use non-interleaved MRoPE layout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_mrope_section(
        rope_dim=args.rope_dim,
        mrope_section=tuple(args.mrope_section),
    )
    source = SplitQkvRmsnormMropeKernel.generate_source(
        head_size=args.head_size,
        rope_dim=args.rope_dim,
        num_tokens=args.num_tokens,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_suite(
            num_tokens=args.ref_num_tokens,
            head_configs=args.ref_head_configs,
            head_size=args.head_size,
            rope_dim=args.rope_dim,
            eps=args.eps,
            mrope_section=tuple(args.mrope_section),
            is_interleaved=args.is_interleaved,
        )


if __name__ == "__main__":
    main()
