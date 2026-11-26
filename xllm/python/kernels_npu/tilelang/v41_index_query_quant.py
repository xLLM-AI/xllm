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

"""Quantize V4.1 index queries with one FP16 scale per head."""

from typing import Any

import tilelang.language as T

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

INDEX_QUERY_DIM = 128
MAX_QUERY_HEADS = 4096 * 64
INDEX_QUERY_QUANT_PASS_CONFIGS = {**DEFAULT_ASCEND_PASS_CONFIGS, "tl.disable_safe_memory_legalize": True}


def build_index_query_quant_kernel(*, vec_core_num: int) -> Any:
    dim = INDEX_QUERY_DIM
    capacity = MAX_QUERY_HEADS
    tile_heads = 16
    size = tile_heads * dim

    @T.prim_func
    def index_query_quant(
        query: T.Tensor((capacity, dim), "bfloat16"),
        codes: T.Tensor((capacity, dim), "int8"),
        scales: T.Tensor((capacity,), "float16"),
        heads: T.int32,
    ) -> None:
        with T.Kernel(vec_core_num // 2, is_npu=True) as (cid, vid), T.Scope("V"):
            worker = cid * 2 + vid
            source = T.alloc_ub((tile_heads, dim), "bfloat16")
            values = T.alloc_ub((tile_heads, dim), "float32")
            scratch = T.alloc_ub((tile_heads, dim), "float32")
            scale = T.alloc_ub((tile_heads,), "float32")
            rounded_scale = T.alloc_ub((tile_heads,), "float16")
            rounded_codes = T.alloc_ub((tile_heads, dim), "int32")
            cast_codes = T.alloc_ub((tile_heads, dim), "float16")
            output = T.alloc_ub((tile_heads, dim), "int8")
            for batch in T.serial(T.ceildiv(heads // tile_heads, vec_core_num)):
                start = (batch * vec_core_num + worker) * tile_heads
                if start < heads:
                    T.copy(query[start, 0], source)
                    T.tile.cast(values, source, "CAST_NONE", size)
                    T.tile.abs(scratch, values)
                    T.reduce_max(scratch, scale, dim=-1)
                    T.tile.mul(scale, scale, 1.0 / 127.0)
                    T.tile.max(scale, scale, 2.0**-24)
                    T.tile.cast(rounded_scale, scale, "CAST_RINT", tile_heads)
                    T.copy(rounded_scale, scales[start])
                    # The indexer consumes FP16 scales. Quantize with the
                    # rounded value to match the existing Tensor path.
                    T.tile.cast(scale, rounded_scale, "CAST_NONE", tile_heads)
                    T.tile.broadcast(scratch, scale, axis=1)
                    T.tile.div(values, values, scratch)
                    T.tile.min(values, values, 127.0)
                    T.tile.max(values, values, -127.0)
                    T.tile.cast(rounded_codes, values, "CAST_RINT", size)
                    T.tile.cast(cast_codes, rounded_codes, "CAST_RINT", size)
                    T.tile.cast(output, cast_codes, "CAST_RINT", size)
                    T.copy(output, codes[start, 0])

    return index_query_quant
