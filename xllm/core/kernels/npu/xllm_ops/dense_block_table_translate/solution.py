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

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch_npu
import triton

if __package__:
    from .dense_contract import validate
    from .dense_kernels import gather_rows, prepare_indices
    from .native_loader import load_native
else:
    from dense_contract import validate
    from dense_kernels import gather_rows, prepare_indices
    from native_loader import load_native


class Prepared:
    def __init__(self, inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> None:
        self.info = validate(inputs, attributes)
        self.inputs = tuple(inputs)
        self.attributes = dict(attributes)
        x, _ = self.inputs
        self.n = x.numel()
        self.shape = tuple(x.shape)
        if not self.info["single"]:
            self.shape = (self.info["layers"], *self.shape)
        self.backend = "native" if self.info["blocks"] > 4096 else "triton"
        self._native = load_native() if self.backend == "native" and self.n else None
        self._launcher = None
        if self.n:
            with torch.npu.device(x.device):
                if self.backend == "native":
                    indices = torch.empty((self.n,), dtype=torch.int64, device=x.device)
                    invalid = torch.empty((self.n,), dtype=torch.bool, device=x.device)
                    grid = (triton.cdiv(self.n, 512), 1, 1)
                    kernel = prepare_indices[grid](x, indices, invalid, self.n, TILE=512)
                else:
                    scratch = torch.empty(self.shape, dtype=torch.int32, device=x.device)
                    tile = 256 if self.n * self.info["layers"] < 32768 else 512
                    grid = (triton.cdiv(self.n, tile), self.info["layers"], 1)
                    kernel = gather_rows[grid](
                        x,
                        self.inputs[1],
                        scratch,
                        self.n,
                        self.info["blocks"],
                        TABLE=triton.next_power_of_2(self.info["blocks"]),
                        TILE=tile,
                    )
                self._launcher = kernel[grid]

    def run(self) -> torch.Tensor:
        x, mapping = self.inputs
        with torch.npu.device(x.device):
            if self.n == 0:
                return torch.empty(self.shape, dtype=torch.int32, device=x.device)
            if self.backend == "native":
                indices = torch.empty((self.n,), dtype=torch.int64, device=x.device)
                invalid = torch.empty((self.n,), dtype=torch.bool, device=x.device)
                self._launcher(x, indices, invalid)
                return self._native.gather(mapping, indices, invalid, self.shape)
            output = torch.empty(self.shape, dtype=torch.int32, device=x.device)
            self._launcher(x, mapping, output)
            return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> Prepared:
    return Prepared(inputs, attributes)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    return prepare(inputs, attributes).run()
