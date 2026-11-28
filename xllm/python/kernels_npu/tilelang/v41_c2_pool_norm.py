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

"""Fuse ratio-two FP32 gated pooling and the BF16 compressor RMSNorm."""

from typing import Any

import tilelang.language as T

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

C2_HEAD_DIM = 512
C2_MAX_GROUPS = 65536
C2_MAX_STATE_ROWS = 1 << 21
C2_PASS_CONFIGS = {**DEFAULT_ASCEND_PASS_CONFIGS, "tl.disable_safe_memory_legalize": True}


@T.macro
def _pool_norm(left, right, gate_left, gate_right, denom, weight_fp32, rounded, rms, eps):
    T.tile.max(denom, gate_left, gate_right)
    T.tile.sub(gate_left, gate_left, denom)
    T.tile.sub(gate_right, gate_right, denom)
    T.tile.exp(gate_left, gate_left)
    T.tile.exp(gate_right, gate_right)
    T.tile.add(denom, gate_left, gate_right)
    T.tile.div(gate_left, gate_left, denom)
    T.tile.div(gate_right, gate_right, denom)
    T.tile.mul(left, left, gate_left)
    T.tile.mul(right, right, gate_right)
    T.tile.add(left, left, right)

    # Match the checkpoint's BF16 rounding between pooling and RMSNorm.
    T.tile.cast(rounded, left, "CAST_RINT", C2_HEAD_DIM)
    T.tile.cast(left, rounded, "CAST_NONE", C2_HEAD_DIM)
    T.tile.mul(right, left, left)
    T.reduce_sum(right, rms, dim=-1)
    T.tile.mul(rms, rms, 1.0 / C2_HEAD_DIM)
    T.tile.add(rms, rms, eps)
    T.tile.rsqrt(rms, rms)
    T.tile.broadcast(denom, rms)
    T.tile.mul(left, left, denom)
    T.tile.mul(left, left, weight_fp32)
    T.tile.cast(rounded, left, "CAST_RINT", C2_HEAD_DIM)


def build_c2_pool_norm_kernel(*, vec_core_num: int) -> Any:
    dim = C2_HEAD_DIM
    max_groups = C2_MAX_GROUPS
    vectors_per_block = 2

    @T.prim_func
    def c2_pool_norm(
        kv: T.Tensor((max_groups, 2, dim), "float32"),
        score: T.Tensor((max_groups, 2, dim), "float32"),
        weight: T.Tensor((dim,), "bfloat16"),
        output: T.Tensor((max_groups, dim), "bfloat16"),
        groups: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(vec_core_num // vectors_per_block, is_npu=True) as (cid, vid):
            worker = cid * vectors_per_block + vid
            with T.Scope("V"):
                left = T.alloc_ub((1, dim), "float32")
                right = T.alloc_ub((1, dim), "float32")
                gate_left = T.alloc_ub((1, dim), "float32")
                gate_right = T.alloc_ub((1, dim), "float32")
                denom = T.alloc_ub((1, dim), "float32")
                weight_bf16 = T.alloc_ub((1, dim), "bfloat16")
                weight_fp32 = T.alloc_ub((1, dim), "float32")
                rounded = T.alloc_ub((1, dim), "bfloat16")
                rms = T.alloc_ub((1,), "float32")
                T.copy(weight[0], weight_bf16[0, :])
                T.tile.cast(weight_fp32, weight_bf16, "CAST_NONE", dim)
                for batch in T.serial(T.ceildiv(groups, vec_core_num)):
                    row = batch * vec_core_num + worker
                    if row < groups:
                        T.copy(kv[row, 0, 0], left[0, :])
                        T.copy(kv[row, 1, 0], right[0, :])
                        T.copy(score[row, 0, 0], gate_left[0, :])
                        T.copy(score[row, 1, 0], gate_right[0, :])
                        _pool_norm(left, right, gate_left, gate_right, denom, weight_fp32, rounded, rms, eps)
                        T.copy(rounded[0, :], output[row, 0])

    return c2_pool_norm


def build_c2_decode_pool_norm_kernel(*, vec_core_num: int) -> Any:
    dim = C2_HEAD_DIM
    max_groups = C2_MAX_GROUPS
    max_state_rows = C2_MAX_STATE_ROWS

    @T.prim_func
    def c2_decode_pool_norm(
        kv: T.Tensor((max_groups, dim), "float32"),
        score: T.Tensor((max_groups, dim), "float32"),
        state_kv: T.Tensor((max_state_rows, dim), "float32"),
        state_score: T.Tensor((max_state_rows, dim), "float32"),
        previous_rows: T.Tensor((max_groups,), "int32"),
        weight: T.Tensor((dim,), "bfloat16"),
        output: T.Tensor((max_groups, dim), "bfloat16"),
        groups: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(vec_core_num // 2, is_npu=True) as (cid, vid), T.Scope("V"):
            worker = cid * 2 + vid
            left = T.alloc_ub((1, dim), "float32")
            right = T.alloc_ub((1, dim), "float32")
            gate_left = T.alloc_ub((1, dim), "float32")
            gate_right = T.alloc_ub((1, dim), "float32")
            denom = T.alloc_ub((1, dim), "float32")
            weight_bf16 = T.alloc_ub((1, dim), "bfloat16")
            weight_fp32 = T.alloc_ub((1, dim), "float32")
            rounded = T.alloc_ub((1, dim), "bfloat16")
            rms = T.alloc_ub((1,), "float32")
            T.copy(weight[0], weight_bf16[0, :])
            T.tile.cast(weight_fp32, weight_bf16, "CAST_NONE", dim)
            for batch in T.serial(T.ceildiv(groups, vec_core_num)):
                row = batch * vec_core_num + worker
                if row < groups:
                    previous = previous_rows[row]
                    # Cache rows are nonnegative. A negative value encodes
                    # the index of a predecessor in this forward as -(i + 1).
                    if previous < 0:
                        T.copy(kv[-previous - 1, 0], left[0, :])
                        T.copy(score[-previous - 1, 0], gate_left[0, :])
                    else:
                        T.copy(state_kv[previous, 0], left[0, :])
                        T.copy(state_score[previous, 0], gate_left[0, :])
                    T.copy(kv[row, 0], right[0, :])
                    T.copy(score[row, 0], gate_right[0, :])
                    _pool_norm(left, right, gate_left, gate_right, denom, weight_fp32, rounded, rms, eps)
                    T.copy(rounded[0, :], output[row, 0])

    return c2_decode_pool_norm
