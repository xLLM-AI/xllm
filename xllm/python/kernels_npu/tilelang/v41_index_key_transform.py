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

"""Fuse the V4.1 index key RMSNorm and interleaved RoPE."""

import tilelang.language as T

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

INDEX_KEY_DIM = 128
INDEX_ROPE_DIM = 64
MAX_INDEX_ROWS = 65536
INDEX_TRANSFORM_PASS_CONFIGS = {**DEFAULT_ASCEND_PASS_CONFIGS, "tl.disable_safe_memory_legalize": True}


def build_index_key_transform_kernel(*, vec_core_num: int):
    dim, rope_dim = INDEX_KEY_DIM, INDEX_ROPE_DIM
    max_rows = MAX_INDEX_ROWS
    vectors_per_block = 2

    @T.prim_func
    def index_key_transform(
        key: T.Tensor((max_rows, dim), "bfloat16"),
        weight: T.Tensor((dim,), "bfloat16"),
        cos_sin: T.Tensor((max_rows, rope_dim), "float32"),
        output: T.Tensor((max_rows, dim), "bfloat16"),
        num_rows: T.int32,
        eps: T.float32,
    ):
        with T.Kernel(vec_core_num // vectors_per_block, is_npu=True) as (cid, vid):
            worker = cid * vectors_per_block + vid
            with T.Scope("V"):
                key_bf16 = T.alloc_ub((1, dim), "bfloat16")
                x = T.alloc_ub((1, dim), "float32")
                square = T.alloc_ub((1, dim), "float32")
                weight_bf16 = T.alloc_ub((1, dim), "bfloat16")
                weight_fp32 = T.alloc_ub((1, dim), "float32")
                rms = T.alloc_ub((1,), "float32")
                tail_bf16 = T.alloc_ub((1, rope_dim), "bfloat16")
                tail = T.alloc_ub((1, rope_dim), "float32")
                swapped = T.alloc_ub((1, rope_dim), "float32")
                packed_rope = T.alloc_ub((1, rope_dim), "float32")
                cos_full = T.alloc_ub((1, rope_dim), "float32")
                sin_full = T.alloc_ub((1, rope_dim), "float32")
                cos_offsets = T.alloc_ub((1, rope_dim), "uint32")
                sin_offsets = T.alloc_ub((1, rope_dim), "uint32")
                swap_offsets = T.alloc_ub((1, rope_dim), "uint32")
                signs = T.alloc_ub((1, rope_dim), "float32")

                for lane in T.serial(rope_dim):
                    cos_offsets[0, lane] = T.Cast("uint32", (lane // 2) * 4)
                    sin_offsets[0, lane] = T.Cast("uint32", (rope_dim // 2 + lane // 2) * 4)
                    swap_offsets[0, lane] = T.Cast("uint32", (lane // 2 * 2 + 1 - lane % 2) * 4)
                    signs[0, lane] = 2.0 * T.Cast("float32", lane % 2) - 1.0
                T.copy(weight[0], weight_bf16[0, :])
                T.tile.cast(weight_fp32, weight_bf16, "CAST_NONE", dim)
                for batch in T.serial(T.ceildiv(num_rows, vec_core_num)):
                    row = batch * vec_core_num + worker
                    if row < num_rows:
                        T.copy(key[row, 0], key_bf16[0, :])
                        T.tile.cast(x, key_bf16, "CAST_NONE", dim)
                        T.tile.mul(square, x, x)
                        T.reduce_sum(square, rms, dim=-1)
                        T.tile.mul(rms, rms, 1.0 / dim)
                        T.tile.add(rms, rms, eps)
                        T.tile.rsqrt(rms, rms)
                        T.tile.broadcast(square, rms)
                        T.tile.mul(x, x, square)
                        T.tile.mul(x, x, weight_fp32)
                        T.tile.cast(key_bf16, x, "CAST_RINT", dim)

                        T.copy(key_bf16[0, dim - rope_dim : dim], tail_bf16[0, :])
                        T.tile.cast(tail, tail_bf16, "CAST_NONE", rope_dim)
                        T.copy(cos_sin[row, 0], packed_rope[0, :])
                        T.tile.gather(cos_full, packed_rope, cos_offsets, 0)
                        T.tile.gather(sin_full, packed_rope, sin_offsets, 0)
                        T.tile.gather(swapped, tail, swap_offsets, 0)
                        T.tile.mul(tail, tail, cos_full)
                        T.tile.mul(swapped, swapped, signs)
                        T.tile.mul(swapped, swapped, sin_full)
                        T.tile.add(tail, tail, swapped)
                        T.tile.cast(tail_bf16, tail, "CAST_RINT", rope_dim)
                        T.copy(tail_bf16[0, :], key_bf16[0, dim - rope_dim : dim])
                        T.copy(key_bf16[0, :], output[row, 0])

    return index_key_transform
