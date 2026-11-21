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

"""TokenOffsetTileFlags: prepare/run API for Ascend NPU."""

from __future__ import annotations

from typing import Any, Sequence

import torch
import torch_npu  # noqa: F401 -- registers the NPU device backend
import triton

if __package__:
    from .kernels import tile_flags, tile_flags_large, tile_flags_row
    from .validation import validate
else:
    from kernels import tile_flags, tile_flags_large, tile_flags_row
    from validation import validate


class Prepared:
    def __init__(self, inputs: Sequence[torch.Tensor], rows: int, tile: int, capacity: int) -> None:
        self._inputs = tuple(inputs)
        self._device = inputs[0].device
        self._total = rows * capacity
        self._n = inputs[0].numel()
        self._tile = tile
        self._capacity = capacity
        self._lanes = triton.next_power_of_2(tile)
        self._group = min(32, max(1, 1024 // self._lanes))
        self._grid = (triton.cdiv(self._total, self._group),)
        self._kernel = tile_flags_large if self._group == 1 else tile_flags
        if tile == 128:
            self._group = 32
            self._grid = (rows * triton.cdiv(capacity, self._group),)
            self._kernel = tile_flags_row
        self._launch = None
        if self._n and self._total:
            with torch.npu.device(self._device):
                template = torch.empty((self._total,), dtype=torch.uint8, device=self._device)
                compiled = self._kernel.warmup(
                    self._inputs[0],
                    self._inputs[1],
                    self._inputs[2],
                    template,
                    N=self._n,
                    TOTAL=self._total,
                    T=self._tile,
                    K=self._capacity,
                    LANES=self._lanes,
                    GROUP=self._group,
                    grid=self._grid,
                )
                # CompiledKernel's runner queries the current stream on each
                # invocation. The temporary allocation is never an output cache.
                self._launch = compiled[(self._grid[0], 1, 1)]

    def run(self) -> torch.Tensor:
        # Resolve the caller's current stream on this device at every launch.
        # No stream, input values, or output buffers are cached by prepare.
        with torch.npu.device(self._device):
            if self._n == 0:
                return torch.zeros((self._total,), dtype=torch.uint8, device=self._device)
            out = torch.empty((self._total,), dtype=torch.uint8, device=self._device)
            if self._total:
                self._launch(self._inputs[0], self._inputs[1], self._inputs[2], out)
            return out


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    """Validate once; shape/dtype/device/attribute changes require a new prepare.

    In-place value changes must remain legal and be synchronized by the caller
    when crossing streams. Legal storage offsets are supported without copying.
    """
    rows, tile, capacity = validate(inputs, attributes, torch)
    return Prepared(inputs, rows, tile, capacity)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    return prepare(inputs, attributes).run()
