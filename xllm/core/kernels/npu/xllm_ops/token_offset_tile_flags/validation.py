# Copyright 2026 The xLLM Authors.
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

"""One-time contract validation. This module does not compute output flags."""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
from typing import Any

I32_MAX = 2**31 - 1


def validate_attributes(attributes: dict[str, int]) -> tuple[int, int]:
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dict")
    if set(attributes) != {"tile_size", "max_tiles_per_seq"}:
        raise ValueError("missing or unknown attributes")
    if any(type(v) is not int or v < 1 for v in attributes.values()):
        raise ValueError("attributes must be positive integers, excluding bool")
    if attributes["tile_size"] > 4096:
        raise ValueError("tile_size must be in 1..4096")
    return attributes["tile_size"], attributes["max_tiles_per_seq"]


def validate_values(values: Sequence[Sequence[int]], tile: int, capacity: int) -> int:
    offsets, indptr, lengths, tileptr = values
    rows = len(lengths)
    if any(len(x) > I32_MAX for x in values) or rows * capacity > I32_MAX:
        raise ValueError("input/output exceeds int32 element-count limit")
    if len(indptr) != rows + 1 or len(tileptr) != rows + 1:
        raise ValueError("indptr and tileptr must have R+1 elements")
    if indptr[0] != 0 or indptr[-1] != len(offsets):
        raise ValueError("indptr must start at zero and end at N")
    if any(indptr[r] > indptr[r + 1] or lengths[r] < 0 or lengths[r] > indptr[r + 1] - indptr[r] for r in range(rows)):
        raise ValueError("invalid segment boundaries or lengths")
    if any(v != r * capacity for r, v in enumerate(tileptr)):
        raise ValueError("tileptr must be dense [0,K,...,R*K]")
    if any((n + tile - 1) // tile > capacity for n in lengths):
        raise ValueError("insufficient tile capacity")
    if any(v < 0 for v in offsets):
        raise ValueError("offsets must be nonnegative")
    return rows


def validate(inputs: Sequence[Any], attributes: dict[str, int], torch: ModuleType) -> tuple[int, int, int]:
    if not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence")
    if len(inputs) != 4:
        raise ValueError("exactly four input tensors are required")
    tile, capacity = validate_attributes(attributes)
    for x in inputs:
        if not isinstance(x, torch.Tensor) or x.dtype != torch.int32:
            raise TypeError("inputs must be int32 tensors")
        if x.device.type != "npu" or x.device != inputs[0].device:
            raise ValueError("inputs must be on the same NPU")
        if x.ndim != 1 or not x.is_contiguous():
            raise ValueError("inputs must be contiguous vectors")
        if x.numel() > I32_MAX:
            raise ValueError("input exceeds int32 element-count limit")
    # Only prepare performs readback. The hot path retains no CPU input values.
    rows = validate_values([x.detach().cpu().tolist() for x in inputs], tile, capacity)
    return rows, tile, capacity
