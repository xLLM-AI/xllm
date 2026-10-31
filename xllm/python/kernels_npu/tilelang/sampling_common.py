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

"""Shared random draws and categorical selection primitives for Ascend."""

import tilelang.language as T
from tilelang import tvm

from .utils import DEFAULT_ASCEND_PASS_CONFIGS

MAX_ROWS = 4096
MAX_VOCAB = 262144
TILE_SIZE = 4096
MAX_CHUNKS = MAX_VOCAB // TILE_SIZE
SAMPLING_PASS_CONFIGS = {
    **DEFAULT_ASCEND_PASS_CONFIGS,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


@T.macro
def _select_categorical_half(
    half: int,
    tile: tvm.tir.Buffer,
    reduced: tvm.tir.Buffer,
    right: tvm.tir.Buffer,
    scratch: tvm.tir.Buffer,
    state: tvm.tir.Buffer,
    choice: tvm.tir.Buffer,
) -> None:
    T.tile.fill(scratch, 0.0)
    T.copy(tile[choice[1] : choice[1] + half], scratch[:half])
    T.reduce_sum(scratch, reduced, dim=0)
    T.set_flag("v", "s", 7)
    T.wait_flag("v", "s", 7)
    T.tile.fill(scratch, 0.0)
    T.copy(tile[choice[1] + half : choice[1] + 2 * half], scratch[:half])
    T.reduce_sum(scratch, right, dim=0)
    T.set_flag("v", "s", 7)
    T.wait_flag("v", "s", 7)
    if right[0] > 0.0 and (state[3] >= reduced[0] or reduced[0] == 0.0):
        state[3] = T.min(T.max(state[3] - reduced[0], 0.0), right[0] * (1.0 - 2.0**-24))
        choice[1] = choice[1] + half
    else:
        state[3] = T.min(state[3], reduced[0] * (1.0 - 2.0**-24))


@T.macro
def select_categorical_chunk(
    seeds: tvm.tir.Buffer,
    offsets: tvm.tir.Buffer,
    uniforms: tvm.tir.Buffer,
    row: T.int32,
    rng_flags: T.int32,
    n_chunks: T.int32,
    chunks: tvm.tir.Buffer,
    state: tvm.tir.Buffer,
    choice: tvm.tir.Buffer,
) -> None:
    # state[0] holds total mass; choose a chunk and its local threshold.
    uniform = T.alloc_var("float32", 0.0)
    if (rng_flags & 1) == 0 and (rng_flags & 2) != 0:
        uniform = uniforms[row]
    if (rng_flags & 1) != 0:
        if seeds[row] == -1 and (rng_flags & 2) != 0:
            uniform = uniforms[row]
        if seeds[row] != -1:
            value = T.alloc_var(
                "uint64",
                T.Cast("uint64", seeds[row])
                + T.Cast("uint64", offsets[row]) * ((T.uint64(0x9E3779B9) << 32) | T.uint64(0x7F4A7C15)),
            )
            value = (value ^ (value >> 30)) * ((T.uint64(0xBF58476D) << 32) | T.uint64(0x1CE4E5B9))
            value = (value ^ (value >> 27)) * ((T.uint64(0x94D049BB) << 32) | T.uint64(0x133111EB))
            value = value ^ (value >> 31)
            uniform = T.Cast("float32", T.Cast("int32", value >> 40)) * (1.0 / (1 << 24))

    # Half-open intervals; clamp rounding at the last row
    # endpoint. Positive lower bounds give one owner even
    # when independently rounded interval ends disagree.
    state[1] = T.min(uniform * state[0], state[0] * (1.0 - 2.0**-24))
    state[2] = 0.0
    choice[0] = 0
    state[3] = 0.0
    for chunk in T.serial(n_chunks):
        if chunks[chunk] > 0.0 and state[2] <= state[1]:
            choice[0] = chunk
            state[3] = T.min(T.max(state[1] - state[2], 0.0), chunks[chunk] * (1.0 - 2.0**-24))
        state[2] = state[2] + chunks[chunk]


@T.macro
def select_categorical_tile(
    tile: tvm.tir.Buffer,
    reduced: tvm.tir.Buffer,
    right: tvm.tir.Buffer,
    scratch: tvm.tir.Buffer,
    state: tvm.tir.Buffer,
    choice: tvm.tir.Buffer,
) -> None:
    # state[3] is the threshold in this tile; choice[2] receives the token.
    choice[1] = 0
    _select_categorical_half(2048, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(1024, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(512, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(256, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(128, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(64, tile, reduced, right, scratch, state, choice)
    _select_categorical_half(32, tile, reduced, right, scratch, state, choice)
    choice[2] = choice[1]
    state[2] = 0.0
    for column in T.serial(32):
        weight = tile[choice[1] + column]
        if weight > 0.0 and state[2] <= state[3]:
            choice[2] = choice[1] + column
        state[2] = state[2] + weight
