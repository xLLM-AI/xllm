# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Intermediate residual-stream capture for Python target models."""

from __future__ import annotations

import torch


class AuxHiddenCapture:
    """Captures selected layer inputs and concatenates them in config order."""

    def __init__(self, layers_to_capture: tuple[int, ...]) -> None:
        self._layers_to_capture = layers_to_capture
        self._capture_set = frozenset(layers_to_capture)

    @property
    def enabled(self) -> bool:
        return bool(self._layers_to_capture)

    def capture_layer(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        captured: dict[int, torch.Tensor],
    ) -> None:
        if layer_id not in self._capture_set:
            return
        captured[layer_id] = hidden.clone() if residual is None else hidden + residual

    def finalize(
        self,
        hidden: torch.Tensor,
        captured: dict[int, torch.Tensor],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return hidden
        aux_hidden = torch.cat([captured[layer_id] for layer_id in self._layers_to_capture], dim=-1)
        return hidden, aux_hidden
