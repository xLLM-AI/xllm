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

"""A-only ablation kernel for request-seeded sampling from row weights."""

import tilelang.language as T
from tilelang import tvm

from .sampling_common import (
    MAX_CHUNKS,
    MAX_ROWS,
    MAX_VOCAB,
    TILE_SIZE,
    select_categorical_chunk,
    select_categorical_tile,
)


def build_seeded_sample_kernel(task_count: int = 48) -> tvm.tir.PrimFunc:
    """Select a tile and then a token without a vocabulary-sized CDF buffer.

    Each vector core owns complete rows, so reduction order and random draws
    are independent of row order and batch size. The caller supplies uniform
    draws only for rows whose seed is -1. Weights must be finite, nonnegative,
    and contain at least one positive value in each sampled row.
    """
    if task_count <= 0 or task_count % 2:
        raise ValueError("task_count must be a positive even integer")

    @T.prim_func
    def seeded_sample(
        weights: T.Tensor((MAX_ROWS * MAX_VOCAB,), "float32"),
        seeds: T.Tensor((MAX_ROWS,), "int64"),
        offsets: T.Tensor((MAX_ROWS,), "int64"),
        uniforms: T.Tensor((MAX_ROWS,), "float32"),
        output: T.Tensor((MAX_ROWS,), "int64"),
        rows: T.int32,
        vocab: T.int32,
        row_stride: T.int32,
        rng_flags: T.int32,
    ) -> None:
        with T.Kernel(task_count // 2, is_npu=True) as (cid, vid):
            task = cid * 2 + vid
            with T.Scope("V"):
                tile = T.alloc_ub((TILE_SIZE,), "float32")
                chunks = T.alloc_ub((MAX_CHUNKS,), "float32")
                reduced = T.alloc_ub((1,), "float32")
                right = T.alloc_ub((1,), "float32")
                scratch = T.alloc_ub((TILE_SIZE // 2,), "float32")
                state = T.alloc_ub((8,), "float32")
                choice = T.alloc_ub((8,), "int32")
                result = T.alloc_ub((4,), "int64")
                for row_index in T.serial(T.ceildiv(rows, task_count)):
                    row = row_index * task_count + task
                    if row < rows:
                        n_chunks = T.ceildiv(vocab, TILE_SIZE)
                        state[0] = 0.0
                        for chunk in T.serial(n_chunks):
                            count = T.min(TILE_SIZE, vocab - chunk * TILE_SIZE)
                            T.tile.fill(tile, 0.0)
                            T.copy(
                                weights[
                                    row * row_stride + chunk * TILE_SIZE : row * row_stride + chunk * TILE_SIZE + count
                                ],
                                tile,
                            )
                            T.reduce_sum(tile, reduced, dim=0)
                            T.set_flag("v", "s", 7)
                            T.wait_flag("v", "s", 7)
                            chunks[chunk] = reduced[0]
                            state[0] = state[0] + reduced[0]

                        select_categorical_chunk(
                            seeds, offsets, uniforms, row, rng_flags, n_chunks, chunks, state, choice
                        )
                        count = T.min(TILE_SIZE, vocab - choice[0] * TILE_SIZE)
                        T.tile.fill(tile, 0.0)
                        T.copy(
                            weights[
                                row * row_stride + choice[0] * TILE_SIZE : row * row_stride
                                + choice[0] * TILE_SIZE
                                + count
                            ],
                            tile,
                        )
                        # Reduce a binary tree in UB. The final 32 values are
                        # scalar to avoid unsupported sub-block vector slices.
                        select_categorical_tile(tile, reduced, right, scratch, state, choice)
                        result[0] = T.Cast("int64", choice[0] * TILE_SIZE + choice[2])
                        T.copy(result[:1], output[row : row + 1])

    return seeded_sample
