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

import triton
import triton.language as tl


@triton.jit
def gather_rows(
    X: tl.tensor, M: tl.tensor, Y: tl.tensor, N: tl.constexpr, B: tl.constexpr, TABLE: tl.constexpr, TILE: tl.constexpr
) -> None:
    layer = tl.program_id(1)
    pos = tl.program_id(0) * TILE + tl.arange(0, TILE)
    logical = tl.load(X + pos, mask=pos < N, other=-1)
    table_pos = tl.arange(0, TABLE)
    table = tl.load(M + layer * B + table_pos, mask=table_pos < B, other=0)
    safe = tl.maximum(logical, 0)

    gathered = tl.gather(table.to(tl.float32, bitcast=True), safe, axis=0)
    physical = gathered.to(tl.int32, bitcast=True)
    result = tl.where(logical < 0, -1, physical)
    tl.store(Y + layer * N + pos, result, mask=pos < N)


@triton.jit
def prepare_indices(X: tl.tensor, INDICES: tl.tensor, INVALID: tl.tensor, N: tl.constexpr, TILE: tl.constexpr) -> None:
    pos = tl.program_id(0) * TILE + tl.arange(0, TILE)
    logical = tl.load(X + pos, mask=pos < N, other=-1)
    tl.store(INDICES + pos, tl.maximum(logical, 0).to(tl.int64), mask=pos < N)
    tl.store(INVALID + pos, logical < 0, mask=pos < N)
