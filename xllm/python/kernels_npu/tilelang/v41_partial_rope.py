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

"""V4.1 interleaved partial RoPE with FP32 coefficients."""

from typing import Any

import tilelang.language as T

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

ROPE_WIDTH = 64
MAX_TOKENS = 4096
MAX_HEADS = 128
MAX_CACHE_ROWS = 1 << 21
PASS_CONFIGS = {**DEFAULT_ASCEND_PASS_CONFIGS, "tl.disable_safe_memory_legalize": True}


def build_v41_partial_rope_kernel(head_dim: int, tile_heads: int, vec_cores: int) -> Any:
    width = ROPE_WIDTH
    max_rows = MAX_TOKENS * MAX_HEADS
    tile_elements = tile_heads * width

    @T.prim_func
    def v41_partial_rope(
        value: T.Tensor((max_rows, head_dim), "bfloat16"),
        cos: T.Tensor((MAX_TOKENS, width), "float32"),
        signed_sin: T.Tensor((MAX_TOKENS, width), "float32"),
        swap_offsets: T.Tensor((8, width), "uint32"),
        tokens: T.int32,
        heads: T.int32,
        inverse: T.int32,
    ):
        with T.Kernel(vec_cores // 2, is_npu=True) as (cid, vid), T.Scope("V"):
            elements = T.alloc_ub((tile_heads, width), "bfloat16")
            rotated = T.alloc_ub((tile_heads, width), "float32")
            swapped = T.alloc_ub((tile_heads, width), "float32")
            coefficients = T.alloc_ub((1, width), "float32")
            broadcast = T.alloc_ub((tile_heads, width), "float32")
            offsets = T.alloc_ub((tile_heads, width), "uint32")
            T.copy(swap_offsets[:tile_heads, :], offsets)
            tiles = tokens * (heads // tile_heads)
            for iteration in T.serial(T.ceildiv(tiles, vec_cores)):
                tile = cid * 2 + vid + iteration * vec_cores
                if tile < tiles:
                    first = tile * tile_heads
                    token = first // heads
                    T.copy(value[first : first + tile_heads, head_dim - width :], elements)
                    T.tile.cast(rotated, elements, "CAST_NONE", tile_elements)
                    T.tile.gather(swapped, rotated, offsets, 0)
                    T.copy(cos[token : token + 1, :], coefficients)
                    if tile_heads == 1:
                        T.tile.mul(rotated, rotated, coefficients)
                    else:
                        T.tile.broadcast(broadcast, coefficients)
                        T.tile.mul(rotated, rotated, broadcast)
                    T.copy(signed_sin[token : token + 1, :], coefficients)
                    if inverse != 0:
                        T.tile.mul(coefficients, coefficients, -1.0)
                    if tile_heads == 1:
                        T.tile.mul(swapped, swapped, coefficients)
                    else:
                        T.tile.broadcast(broadcast, coefficients)
                        T.tile.mul(swapped, swapped, broadcast)
                    # Match the reference's separate FP32 products before BF16 rounding.
                    T.tile.add(rotated, rotated, swapped)
                    T.tile.cast(elements, rotated, "CAST_RINT", tile_elements)
                    T.copy(elements, value[first : first + tile_heads, head_dim - width :])

    return v41_partial_rope


def build_v41_swa_rope_cache_kernel(head_dim: int, vec_cores: int) -> Any:
    width = ROPE_WIDTH

    @T.prim_func
    def v41_swa_rope_cache(
        value: T.Tensor((MAX_TOKENS, head_dim), "bfloat16"),
        cos: T.Tensor((MAX_TOKENS, width), "float32"),
        signed_sin: T.Tensor((MAX_TOKENS, width), "float32"),
        swap_offsets: T.Tensor((8, width), "uint32"),
        slots: T.Tensor((MAX_TOKENS,), "int32"),
        cache: T.Tensor((MAX_CACHE_ROWS, head_dim), "bfloat16"),
        tokens: T.int32,
        cache_rows: T.int32,
    ):
        with T.Kernel(vec_cores // 2, is_npu=True) as (cid, vid), T.Scope("V"):
            kv = T.alloc_ub((1, head_dim), "bfloat16")
            tail = T.alloc_ub((1, width), "bfloat16")
            rotated = T.alloc_ub((1, width), "float32")
            swapped = T.alloc_ub((1, width), "float32")
            coefficients = T.alloc_ub((1, width), "float32")
            offsets = T.alloc_ub((1, width), "uint32")
            T.copy(swap_offsets[0:1, :], offsets)
            for iteration in T.serial(T.ceildiv(tokens, vec_cores)):
                row = cid * 2 + vid + iteration * vec_cores
                if row < tokens:
                    T.copy(value[row, :], kv[0, :])
                    T.copy(kv[0, head_dim - width :], tail[0, :])
                    T.tile.cast(rotated, tail, "CAST_NONE", width)
                    T.tile.gather(swapped, rotated, offsets, 0)
                    T.copy(cos[row, :], coefficients[0, :])
                    T.tile.mul(rotated, rotated, coefficients)
                    T.copy(signed_sin[row, :], coefficients[0, :])
                    T.tile.mul(swapped, swapped, coefficients)
                    T.tile.add(rotated, rotated, swapped)
                    T.tile.cast(tail, rotated, "CAST_RINT", width)
                    T.copy(tail[0, :], value[row, head_dim - width :])
                    slot = slots[row]
                    if slot >= 0 and slot < cache_rows:
                        T.copy(tail[0, :], kv[0, head_dim - width :])
                        T.copy(kv[0, :], cache[slot, :])

    return v41_swa_rope_cache
