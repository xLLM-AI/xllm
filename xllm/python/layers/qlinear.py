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

"""Transparent linear wrapper: fp or W8A8, resolved at load time.

Mirrors libtorch #1426's resolved_weight_quant_method_ pattern in pure Python.
The model graph constructs ``QLinear`` everywhere; whether it actually quantizes
is decided at ``load_weights`` by probing the checkpoint for quant tensors
(``deq_scale`` / ``weight_scale``). Forward is identical either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        kind: str = "static",
        row_parallel: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        self.row_parallel = row_parallel
        self.use_w8a8: bool | None = None
        # fp weight (used iff use_w8a8 is False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self._w8a8: nn.Module | None = None  # built in Task 2

    def resolve_quant(self, has_quant_tensors: bool) -> None:
        self.use_w8a8 = bool(has_quant_tensors)
        if self.use_w8a8:
            self._build_w8a8()

    def _build_w8a8(self) -> None:
        from xllm.python.models.deepseek_v32 import (
            W8A8DynamicLinear,
            W8A8StaticLinear,
        )

        device = self.weight.device
        cls = W8A8DynamicLinear if self.kind == "dynamic" else W8A8StaticLinear
        if cls is W8A8StaticLinear:
            self._w8a8 = cls(self.in_features, self.out_features, device, row_parallel=self.row_parallel)
        else:
            self._w8a8 = cls(self.in_features, self.out_features, device)

    def process_weights_after_loading(self) -> None:
        if self.use_w8a8:
            self._w8a8.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_w8a8 is None:
            raise RuntimeError("QLinear.forward before resolve_quant")
        if self.use_w8a8:
            return self._w8a8(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)
