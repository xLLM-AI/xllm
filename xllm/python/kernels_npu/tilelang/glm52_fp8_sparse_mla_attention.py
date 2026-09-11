#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import tilelang.language as T

if TYPE_CHECKING:
    from tvm.tir import PrimFunc

from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

LATENT_DIM = 512
ROPE_DIM = 64
TOPK = 2048
BLOCK_SIZE = 128
CORE_NUM = 24
HEAD_TILE = 16
KV_TILE = 64
VEC_NUM = 2
VEC_HEAD_TILE = HEAD_TILE // VEC_NUM
VEC_KV_TILE = KV_TILE // VEC_NUM
KV_COPY_ROWS = 32
MAX_NUM_QUERIES = 1024
MAX_CACHE_BLOCKS = 32768
MAX_BLOCK_TABLE_LEN = 32768
DEFAULT_DTYPE = "bf16"
SUPPORTED_NUM_HEADS = (4, 8, 16)


def build_glm52_fp8_sparse_mla_attention_kernel(num_heads: int) -> PrimFunc:
    if num_heads not in SUPPORTED_NUM_HEADS:
        raise ValueError(
            f"GLM-5.2 FP8 sparse MLA attention only supports num_heads in {SUPPORTED_NUM_HEADS}, got {num_heads}"
        )

    num_kv_tiles = TOPK // KV_TILE
    input_dtype = "bfloat16"
    accum_dtype = "float32"
    index_dtype = "int32"

    @T.prim_func
    def glm52_fp8_sparse_mla_attention_kernel(
        q_latent: T.Tensor((1, MAX_NUM_QUERIES * num_heads * LATENT_DIM), input_dtype),
        q_rope: T.Tensor((1, MAX_NUM_QUERIES * num_heads * ROPE_DIM), input_dtype),
        nope_cache: T.Tensor((MAX_CACHE_BLOCKS, BLOCK_SIZE, LATENT_DIM), "uint8"),
        rope_cache: T.Tensor((MAX_CACHE_BLOCKS, BLOCK_SIZE, ROPE_DIM), "uint8"),
        topk_indices: T.Tensor((MAX_NUM_QUERIES, TOPK), index_dtype),
        block_table: T.Tensor((1, MAX_NUM_QUERIES * MAX_BLOCK_TABLE_LEN), index_dtype),
        actual_seq_lengths_kv: T.Tensor((MAX_NUM_QUERIES,), index_dtype),
        e4m3_decode_table: T.Tensor((256,), accum_dtype),
        output: T.Tensor((MAX_NUM_QUERIES, num_heads, LATENT_DIM), input_dtype),
        workspace_k: T.Tensor((CORE_NUM, KV_TILE, LATENT_DIM), input_dtype),
        workspace_k_rope: T.Tensor((CORE_NUM, KV_TILE, ROPE_DIM), input_dtype),
        workspace_scores: T.Tensor((CORE_NUM, HEAD_TILE, KV_TILE), accum_dtype),
        workspace_probs: T.Tensor((CORE_NUM, HEAD_TILE, KV_TILE), input_dtype),
        workspace_output: T.Tensor((CORE_NUM, HEAD_TILE, LATENT_DIM), accum_dtype),
        workspace_q: T.Tensor((CORE_NUM, HEAD_TILE, LATENT_DIM), input_dtype),
        workspace_q_rope: T.Tensor((CORE_NUM, HEAD_TILE, ROPE_DIM), input_dtype),
        q_token_stride: T.int32,
        q_head_stride: T.int32,
        q_rope_token_stride: T.int32,
        q_rope_head_stride: T.int32,
        num_queries: T.int32,
        block_table_stride: T.int32,
        softmax_scale: T.float32,
    ):
        with T.Kernel(CORE_NUM, is_npu=True) as (cid, vid):
            q_l1 = T.alloc_L1((HEAD_TILE, LATENT_DIM), input_dtype)
            q_rope_l1 = T.alloc_L1((HEAD_TILE, ROPE_DIM), input_dtype)
            k_l1 = T.alloc_L1((KV_TILE, LATENT_DIM), input_dtype)
            k_rope_l1 = T.alloc_L1((KV_TILE, ROPE_DIM), input_dtype)
            probs_l1 = T.alloc_L1((HEAD_TILE, KV_TILE), input_dtype)
            scores_l0c = T.alloc_L0C((HEAD_TILE, KV_TILE), accum_dtype)
            output_l0c = T.alloc_L0C((HEAD_TILE, LATENT_DIM), accum_dtype)

            indices_ub = T.alloc_ub((KV_TILE,), index_dtype)
            indices_fp32_ub = T.alloc_ub((KV_TILE,), accum_dtype)
            valid_mask_ub = T.alloc_ub((32,), "uint8")
            nonnegative_mask_ub = T.alloc_ub((32,), "uint8")
            decode_table_ub = T.alloc_ub((256,), accum_dtype)
            raw_cache_ub = T.alloc_ub((LATENT_DIM,), "uint8")
            raw_cache_fp16_ub = T.alloc_ub((LATENT_DIM,), "float16")
            decode_offsets_i32_ub = T.alloc_ub((LATENT_DIM,), index_dtype)
            decode_offsets_u32_ub = T.alloc_ub((LATENT_DIM,), "uint32")
            decoded_fp32_ub = T.alloc_ub((LATENT_DIM,), accum_dtype)
            decoded_bf16_ub = T.alloc_ub((LATENT_DIM,), input_dtype)
            raw_rope_cache_ub = T.alloc_ub((ROPE_DIM,), "uint8")
            raw_rope_cache_fp16_ub = T.alloc_ub((ROPE_DIM,), "float16")
            rope_decode_offsets_i32_ub = T.alloc_ub((ROPE_DIM,), index_dtype)
            rope_decode_offsets_u32_ub = T.alloc_ub((ROPE_DIM,), "uint32")
            decoded_rope_fp32_ub = T.alloc_ub((ROPE_DIM,), accum_dtype)
            decoded_rope_bf16_ub = T.alloc_ub((ROPE_DIM,), input_dtype)
            k_gather_ub = T.alloc_ub((2, KV_COPY_ROWS, LATENT_DIM), input_dtype)
            k_rope_gather_ub = T.alloc_ub((2, KV_COPY_ROWS, ROPE_DIM), input_dtype)
            q_gather_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), input_dtype)
            q_rope_gather_ub = T.alloc_ub((VEC_HEAD_TILE, ROPE_DIM), input_dtype)

            score_max_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            previous_score_max_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            scores_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), accum_dtype)
            score_max_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), accum_dtype)
            output_scale_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            score_sum_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            normalizer_ub = T.alloc_ub((VEC_HEAD_TILE, 1), accum_dtype)
            probs_bf16_ub = T.alloc_ub((VEC_HEAD_TILE, KV_TILE), input_dtype)
            partial_output_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            accumulated_output_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            normalizer_broadcast_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), accum_dtype)
            output_bf16_ub = T.alloc_ub((VEC_HEAD_TILE, LATENT_DIM), input_dtype)

            queries_per_core = (num_queries + CORE_NUM - 1) // CORE_NUM
            query_start = cid * queries_per_core
            query_end = T.if_then_else(
                query_start + queries_per_core < num_queries,
                query_start + queries_per_core,
                num_queries,
            )

            if cid < num_queries:
                T.copy(e4m3_decode_table, decode_table_ub)
                T.set_flag("mte2", "v", 4)
                T.wait_flag("mte2", "v", 4)
                for query_idx in T.serial(query_start, query_end):
                    T.tile.fill(q_gather_ub, 0.0)
                    T.tile.fill(q_rope_gather_ub, 0.0)
                    T.set_flag("v", "mte2", 8)
                    T.wait_flag("v", "mte2", 8)
                    if num_heads <= VEC_HEAD_TILE:
                        if vid == 0:
                            for head_idx in range(num_heads):
                                T.copy(
                                    q_latent[
                                        0,
                                        query_idx * q_token_stride + head_idx * q_head_stride : query_idx
                                        * q_token_stride
                                        + head_idx * q_head_stride
                                        + LATENT_DIM,
                                    ],
                                    q_gather_ub[head_idx, :],
                                )
                                T.copy(
                                    q_rope[
                                        0,
                                        query_idx * q_rope_token_stride + head_idx * q_rope_head_stride : query_idx
                                        * q_rope_token_stride
                                        + head_idx * q_rope_head_stride
                                        + ROPE_DIM,
                                    ],
                                    q_rope_gather_ub[head_idx, :],
                                )
                    else:
                        for head_idx in range(VEC_HEAD_TILE):
                            global_head_idx = vid * VEC_HEAD_TILE + head_idx
                            T.copy(
                                q_latent[
                                    0,
                                    query_idx * q_token_stride + global_head_idx * q_head_stride : query_idx
                                    * q_token_stride
                                    + global_head_idx * q_head_stride
                                    + LATENT_DIM,
                                ],
                                q_gather_ub[head_idx, :],
                            )
                            T.copy(
                                q_rope[
                                    0,
                                    query_idx * q_rope_token_stride + global_head_idx * q_rope_head_stride : query_idx
                                    * q_rope_token_stride
                                    + global_head_idx * q_rope_head_stride
                                    + ROPE_DIM,
                                ],
                                q_rope_gather_ub[head_idx, :],
                            )
                    T.set_flag("mte2", "mte3", 6)
                    T.wait_flag("mte2", "mte3", 6)
                    T.copy(
                        q_gather_ub,
                        workspace_q[
                            cid,
                            vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                            :,
                        ],
                    )
                    T.copy(
                        q_rope_gather_ub,
                        workspace_q_rope[
                            cid,
                            vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                            :,
                        ],
                    )

                    T.copy(workspace_q[cid, :, :], q_l1)
                    T.copy(workspace_q_rope[cid, :, :], q_rope_l1)
                    T.set_flag("mte2", "mte1", 7)
                    T.wait_flag("mte2", "mte1", 7)

                    actual_kv_len = actual_seq_lengths_kv[query_idx]
                    T.tile.fill(accumulated_output_ub, 0.0)
                    T.tile.fill(normalizer_ub, 0.0)
                    T.tile.fill(score_max_ub, 2.0**30)

                    for tile_idx in T.serial(num_kv_tiles):
                        T.copy(
                            topk_indices[
                                query_idx,
                                tile_idx * KV_TILE : (tile_idx + 1) * KV_TILE,
                            ],
                            indices_ub,
                        )
                        T.set_flag("mte2", "v", 5)
                        T.wait_flag("mte2", "v", 5)
                        T.copy(indices_ub, indices_fp32_ub)
                        T.pipe_barrier("v")
                        T.tile.compare(
                            valid_mask_ub,
                            indices_fp32_ub,
                            T.float32(actual_kv_len - 1),
                            "LE",
                        )
                        T.tile.compare(
                            nonnegative_mask_ub,
                            indices_fp32_ub,
                            T.float32(0.0),
                            "GE",
                        )
                        T.tile.bitwise_and(
                            valid_mask_ub,
                            valid_mask_ub,
                            nonnegative_mask_ub,
                        )

                        for row_idx in range(VEC_KV_TILE):
                            copy_group = row_idx // KV_COPY_ROWS
                            copy_row = row_idx % KV_COPY_ROWS
                            global_copy_group = tile_idx * (VEC_KV_TILE // KV_COPY_ROWS) + copy_group
                            ping_pong = global_copy_group % 2
                            index_in_tile = row_idx + vid * VEC_KV_TILE
                            sparse_index = indices_ub[index_in_tile]
                            safe_sparse_index = T.if_then_else(
                                sparse_index >= 0,
                                T.if_then_else(
                                    sparse_index < actual_kv_len,
                                    sparse_index,
                                    0,
                                ),
                                0,
                            )
                            logical_block = safe_sparse_index // BLOCK_SIZE
                            physical_block = block_table[
                                0,
                                query_idx * block_table_stride + logical_block,
                            ]
                            physical_block = T.if_then_else(
                                physical_block >= 0,
                                physical_block,
                                0,
                            )
                            block_offset = safe_sparse_index % BLOCK_SIZE

                            T.copy(
                                nope_cache[physical_block, block_offset, :],
                                raw_cache_ub,
                            )
                            T.set_flag("mte2", "v", 6)
                            T.wait_flag("mte2", "v", 6)
                            T.tile.cast(
                                raw_cache_fp16_ub,
                                raw_cache_ub,
                                "CAST_NONE",
                                LATENT_DIM,
                            )
                            T.tile.cast(
                                decode_offsets_i32_ub,
                                raw_cache_fp16_ub,
                                "CAST_RINT",
                                LATENT_DIM,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                decode_offsets_i32_ub,
                                decode_offsets_i32_ub,
                                4,
                            )
                            T.pipe_barrier("v")
                            T.reinterpretcast(
                                decode_offsets_u32_ub,
                                decode_offsets_i32_ub,
                                "uint32_t",
                            )
                            # Gather derives its vector length from the source
                            # view. Decode 512 values as two bounded 256-value
                            # operations to avoid UB overrun.
                            T.tile.gather(
                                decoded_fp32_ub[:256],
                                decode_table_ub,
                                decode_offsets_u32_ub[:256],
                                0,
                            )
                            T.tile.gather(
                                decoded_fp32_ub[256:],
                                decode_table_ub,
                                decode_offsets_u32_ub[256:],
                                0,
                            )
                            T.pipe_barrier("v")
                            T.tile.cast(
                                decoded_bf16_ub,
                                decoded_fp32_ub,
                                "CAST_RINT",
                                LATENT_DIM,
                            )
                            T.copy(
                                decoded_bf16_ub,
                                k_gather_ub[ping_pong, copy_row, :],
                            )

                            T.copy(
                                rope_cache[physical_block, block_offset, :],
                                raw_rope_cache_ub,
                            )
                            T.set_flag("mte2", "v", 7)
                            T.wait_flag("mte2", "v", 7)
                            T.tile.cast(
                                raw_rope_cache_fp16_ub,
                                raw_rope_cache_ub,
                                "CAST_NONE",
                                ROPE_DIM,
                            )
                            T.tile.cast(
                                rope_decode_offsets_i32_ub,
                                raw_rope_cache_fp16_ub,
                                "CAST_RINT",
                                ROPE_DIM,
                            )
                            T.pipe_barrier("v")
                            T.tile.mul(
                                rope_decode_offsets_i32_ub,
                                rope_decode_offsets_i32_ub,
                                4,
                            )
                            T.pipe_barrier("v")
                            T.reinterpretcast(
                                rope_decode_offsets_u32_ub,
                                rope_decode_offsets_i32_ub,
                                "uint32_t",
                            )
                            # Limit the source view so Gather writes exactly 64
                            # RoPE values while byte offsets still address the
                            # complete 256-entry decode table allocation.
                            T.tile.gather(
                                decoded_rope_fp32_ub,
                                decode_table_ub[:ROPE_DIM],
                                rope_decode_offsets_u32_ub,
                                0,
                            )
                            T.pipe_barrier("v")
                            T.tile.cast(
                                decoded_rope_bf16_ub,
                                decoded_rope_fp32_ub,
                                "CAST_RINT",
                                ROPE_DIM,
                            )
                            T.copy(
                                decoded_rope_bf16_ub,
                                k_rope_gather_ub[ping_pong, copy_row, :],
                            )

                            if (row_idx + 1) % KV_COPY_ROWS == 0:
                                if global_copy_group > 1:
                                    T.wait_flag("mte3", "v", ping_pong)
                                T.set_flag("v", "mte3", ping_pong)
                                T.wait_flag("v", "mte3", ping_pong)
                                T.copy(
                                    k_gather_ub[ping_pong, :, :],
                                    workspace_k[
                                        cid,
                                        vid * VEC_KV_TILE + copy_group * KV_COPY_ROWS : vid * VEC_KV_TILE
                                        + (copy_group + 1) * KV_COPY_ROWS,
                                        :,
                                    ],
                                )
                                T.copy(
                                    k_rope_gather_ub[ping_pong, :, :],
                                    workspace_k_rope[
                                        cid,
                                        vid * VEC_KV_TILE + copy_group * KV_COPY_ROWS : vid * VEC_KV_TILE
                                        + (copy_group + 1) * KV_COPY_ROWS,
                                        :,
                                    ],
                                )
                                if global_copy_group < num_kv_tiles - 2:
                                    T.set_flag("mte3", "v", ping_pong)

                        T.copy(workspace_k[cid, :, :], k_l1)
                        T.copy(workspace_k_rope[cid, :, :], k_rope_l1)
                        T.set_flag("mte2", "mte1", 1)
                        T.wait_flag("mte2", "mte1", 1)
                        T.gemm_v0(
                            q_l1,
                            k_l1,
                            scores_l0c,
                            transpose_B=True,
                            init=True,
                        )
                        T.gemm_v0(
                            q_rope_l1,
                            k_rope_l1,
                            scores_l0c,
                            transpose_B=True,
                        )
                        T.set_flag("m", "fix", 2)
                        T.wait_flag("m", "fix", 2)
                        T.copy(scores_l0c, workspace_scores[cid, :, :])

                        T.copy(score_max_ub, previous_score_max_ub)
                        T.copy(
                            workspace_scores[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                            scores_ub,
                        )
                        T.set_flag("mte2", "v", 0)
                        T.wait_flag("mte2", "v", 0)
                        for head_idx in T.serial(VEC_HEAD_TILE):
                            T.tile.select(
                                scores_ub[head_idx, :],
                                valid_mask_ub,
                                scores_ub[head_idx, :],
                                -T.infinity(accum_dtype),
                                "VSEL_TENSOR_SCALAR_MODE",
                            )
                        T.pipe_barrier("v")
                        T.reduce_max(scores_ub, score_max_ub, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.mul(score_max_ub, score_max_ub, -softmax_scale)
                        T.pipe_barrier("v")
                        T.tile.min(
                            score_max_ub,
                            score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.broadcast(score_max_broadcast_ub, score_max_ub)
                        T.pipe_barrier("v")
                        T.tile.axpy(
                            score_max_broadcast_ub,
                            scores_ub,
                            softmax_scale,
                        )
                        T.pipe_barrier("v")
                        T.tile.exp(scores_ub, score_max_broadcast_ub)
                        T.pipe_barrier("v")
                        T.tile.sub(
                            previous_score_max_ub,
                            score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.exp(
                            previous_score_max_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.copy(scores_ub, probs_bf16_ub)
                        T.pipe_barrier("v")
                        T.set_flag("v", "mte3", 1)
                        T.wait_flag("v", "mte3", 1)
                        T.copy(
                            probs_bf16_ub,
                            workspace_probs[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                        )

                        T.copy(workspace_probs[cid, :, :], probs_l1)
                        T.set_flag("mte2", "mte1", 3)
                        T.wait_flag("mte2", "mte1", 3)
                        T.gemm_v0(
                            probs_l1,
                            k_l1,
                            output_l0c,
                            init=True,
                        )
                        T.set_flag("m", "fix", 4)
                        T.wait_flag("m", "fix", 4)
                        T.copy(output_l0c, workspace_output[cid, :, :])

                        T.copy(
                            workspace_output[
                                cid,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                            partial_output_ub,
                        )
                        T.set_flag("mte2", "v", 2)
                        T.wait_flag("mte2", "v", 2)
                        T.reduce_sum(scores_ub, score_sum_ub, dim=-1)
                        T.pipe_barrier("v")
                        T.tile.mul(
                            normalizer_ub,
                            normalizer_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(normalizer_ub, normalizer_ub, score_sum_ub)
                        T.pipe_barrier("v")
                        T.tile.broadcast(
                            output_scale_broadcast_ub,
                            previous_score_max_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.mul(
                            accumulated_output_ub,
                            accumulated_output_ub,
                            output_scale_broadcast_ub,
                        )
                        T.pipe_barrier("v")
                        T.tile.add(
                            accumulated_output_ub,
                            accumulated_output_ub,
                            partial_output_ub,
                        )

                    T.tile.broadcast(normalizer_broadcast_ub, normalizer_ub)
                    T.pipe_barrier("v")
                    T.tile.div(
                        accumulated_output_ub,
                        accumulated_output_ub,
                        normalizer_broadcast_ub,
                    )
                    T.pipe_barrier("v")
                    T.copy(accumulated_output_ub, output_bf16_ub)
                    T.set_flag("v", "mte3", 9)
                    T.wait_flag("v", "mte3", 9)
                    if num_heads <= VEC_HEAD_TILE:
                        if vid == 0:
                            T.copy(
                                output_bf16_ub[0:num_heads, :],
                                output[query_idx, 0:num_heads, :],
                            )
                    else:
                        T.copy(
                            output_bf16_ub,
                            output[
                                query_idx,
                                vid * VEC_HEAD_TILE : (vid + 1) * VEC_HEAD_TILE,
                                :,
                            ],
                        )

    return glm52_fp8_sparse_mla_attention_kernel
