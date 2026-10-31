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

"""Categorical sampling with on-chip softmax for Ascend vector cores."""

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


def build_sampling_logits_kernel(dtype: str, task_count: int = 48, normalize_only: bool = False) -> tvm.tir.PrimFunc:
    """Read logits twice without a vocabulary-sized FP32 probability buffer."""
    if dtype not in ("float32", "bfloat16", "float16"):
        raise ValueError(f"unsupported input dtype: {dtype}")
    if task_count <= 0 or task_count % 2:
        raise ValueError("task_count must be a positive even integer")

    output_dtype = "float32" if normalize_only else "int64"
    output_width = 2 if normalize_only else 1

    @T.macro
    def _load(
        inputs: tvm.tir.Buffer,
        native: tvm.tir.Buffer,
        tile: tvm.tir.Buffer,
        columns: tvm.tir.Buffer,
        mask: tvm.tir.Buffer,
        start: T.int32,
        count: T.int32,
    ) -> None:
        if dtype == "float32":
            T.copy(inputs[start : start + count], tile)
        else:
            T.copy(inputs[start : start + count], native)
            T.tile.cast(tile, native, "CAST_NONE", TILE_SIZE)
        T.tile.compare(mask, columns, T.Cast("float32", count), "LT")
        T.tile.select(tile, mask, tile, -T.infinity("float32"), "VSEL_TENSOR_SCALAR_MODE")

    @T.prim_func
    def sampling_logits(
        inputs: T.Tensor((MAX_ROWS * MAX_VOCAB,), dtype),
        seeds: T.Tensor((MAX_ROWS,), "int64"),
        offsets: T.Tensor((MAX_ROWS,), "int64"),
        uniforms: T.Tensor((MAX_ROWS,), "float32"),
        output: T.Tensor((MAX_ROWS * output_width,), output_dtype),
        rows: T.int32,
        vocab: T.int32,
        row_stride: T.int32,
        rng_flags: T.int32,
    ) -> None:
        with T.Kernel(task_count // 2, is_npu=True) as (cid, vid), T.Scope("V"):
            task = cid * 2 + vid
            native = T.alloc_ub((TILE_SIZE,), dtype)
            tile = T.alloc_ub((TILE_SIZE,), "float32")
            columns = T.alloc_ub((TILE_SIZE,), "float32")
            mask = T.alloc_ub((TILE_SIZE // 8,), "uint8")
            chunks = T.alloc_ub((MAX_CHUNKS,), "float32")
            maxima = T.alloc_ub((MAX_CHUNKS,), "float32")
            reduced = T.alloc_ub((1,), "float32")
            right = T.alloc_ub((1,), "float32")
            scratch = T.alloc_ub((TILE_SIZE // 2,), "float32")
            state = T.alloc_ub((8,), "float32")
            choice = T.alloc_ub((8,), "int32")
            result = T.alloc_ub((8,), output_dtype)
            T.tile.createvecindex(columns, 0)
            for row_index in T.serial(T.ceildiv(rows, task_count)):
                row = row_index * task_count + task
                if row < rows:
                    n_chunks = T.ceildiv(vocab, TILE_SIZE)
                    T.tile.fill(chunks, 0.0)
                    T.tile.fill(maxima, -T.infinity("float32"))
                    for chunk in T.serial(n_chunks):
                        count = T.min(TILE_SIZE, vocab - chunk * TILE_SIZE)
                        _load(inputs, native, tile, columns, mask, row * row_stride + chunk * TILE_SIZE, count)
                        T.reduce_max(tile, reduced, dim=0)
                        T.set_flag("v", "s", 7)
                        T.wait_flag("v", "s", 7)
                        state[4] = reduced[0]
                        maxima[chunk] = state[4]
                        if state[4] > -T.infinity("float32"):
                            T.tile.add(tile, tile, -state[4])
                            T.tile.exp(tile, tile)
                            T.reduce_sum(tile, reduced, dim=0)
                            T.set_flag("v", "s", 7)
                            T.wait_flag("v", "s", 7)
                            chunks[chunk] = reduced[0]
                    T.reduce_max(maxima, reduced, dim=0)
                    T.set_flag("v", "s", 7)
                    T.wait_flag("v", "s", 7)
                    state[4] = reduced[0]
                    T.tile.add(maxima, maxima, -state[4])
                    T.tile.exp(maxima, maxima)
                    T.tile.mul(chunks, chunks, maxima)
                    T.reduce_sum(chunks, reduced, dim=0)
                    T.set_flag("v", "s", 7)
                    T.wait_flag("v", "s", 7)
                    state[0] = reduced[0]

                    if normalize_only:
                        result[0] = state[4]
                        T.tile.ln(state[:1], state[:1])
                        T.set_flag("v", "s", 7)
                        T.wait_flag("v", "s", 7)
                        # Keep the shift separate: adding it to log(sumexp)
                        # would lose the latter for large finite logits.
                        result[1] = state[0]
                        T.copy(result[:2], output[row * 2 : row * 2 + 2])
                    else:
                        select_categorical_chunk(
                            seeds, offsets, uniforms, row, rng_flags, n_chunks, chunks, state, choice
                        )
                        count = T.min(TILE_SIZE, vocab - choice[0] * TILE_SIZE)
                        _load(inputs, native, tile, columns, mask, row * row_stride + choice[0] * TILE_SIZE, count)
                        T.reduce_max(tile, reduced, dim=0)
                        T.set_flag("v", "s", 7)
                        T.wait_flag("v", "s", 7)
                        state[4] = reduced[0]
                        T.tile.add(tile, tile, -state[4])
                        T.tile.exp(tile, tile)
                        T.reduce_sum(tile, reduced, dim=0)
                        T.set_flag("v", "s", 7)
                        T.wait_flag("v", "s", 7)
                        state[3] = T.min(state[3] / chunks[choice[0]] * reduced[0], reduced[0] * (1.0 - 2.0**-24))
                        select_categorical_tile(tile, reduced, right, scratch, state, choice)
                        result[0] = T.Cast("int64", choice[0] * TILE_SIZE + choice[2])
                        T.copy(result[:1], output[row : row + 1])

    return sampling_logits
