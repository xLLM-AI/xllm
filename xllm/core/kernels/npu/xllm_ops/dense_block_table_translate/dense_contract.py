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

I32_MAX = 2147483647


def validate(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> dict[str, int | bool]:
    if not isinstance(inputs, Sequence) or isinstance(inputs, (str, bytes)):
        raise TypeError("inputs must be a sequence of tensors")
    if len(inputs) != 2:
        raise ValueError("exactly two inputs are required")
    if not isinstance(attributes, dict):
        raise TypeError("attributes must be a dict")
    if attributes:
        raise ValueError("attributes must be empty")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("inputs must be int32 tensors")
        if tensor.device.type != "npu":
            raise ValueError("inputs must be on NPU")
        if tensor.device != inputs[0].device:
            raise ValueError("inputs must be on the same NPU")
        if not tensor.is_contiguous():
            raise ValueError("inputs must be contiguous")
        if tensor.numel() > I32_MAX:
            raise ValueError("input element count exceeds int32")
    x, mapping = inputs
    if x.ndim != 2:
        raise ValueError("X must have shape [R,C]")
    if mapping.ndim not in (1, 2) or mapping.shape[-1] < 1:
        raise ValueError("M must have shape [B] or [L,B], B >= 1")
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    if layers < 1:
        raise ValueError("L must be at least one")
    if layers * x.numel() > I32_MAX:
        raise ValueError("output element count exceeds int32")
    x_cpu = x.detach().cpu()
    m_cpu = mapping.detach().cpu()
    if bool((m_cpu < 0).any()):
        raise ValueError("mapping values must be nonnegative")
    if bool((x_cpu >= mapping.shape[-1]).any()):
        raise ValueError("logical block address is outside M")
    return {"layers": layers, "blocks": mapping.shape[-1], "single": mapping.ndim == 1}
