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
    """Captures selected layer outputs into a buffer in config order."""

    def __init__(self, layers_to_capture: tuple[int, ...]) -> None:
        self._layers_to_capture = layers_to_capture
        self._capture_slots = {layer_id: slot_id for slot_id, layer_id in enumerate(layers_to_capture)}

    @property
    def enabled(self) -> bool:
        return bool(self._layers_to_capture)

    def create_buffer(self, hidden: torch.Tensor) -> torch.Tensor | None:
        if not self.enabled:
            return None
        return hidden.new_empty((*hidden.shape[:-1], hidden.shape[-1] * len(self._layers_to_capture)))

    def capture_layer(
        self,
        layer_id: int,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        buffer: torch.Tensor | None,
    ) -> None:
        slot_id = self._capture_slots.get(layer_id)
        if slot_id is None:
            return
        assert buffer is not None
        hidden_size = hidden.shape[-1]
        slot = buffer.narrow(-1, slot_id * hidden_size, hidden_size)
        if residual is None:
            slot.copy_(hidden)
        else:
            torch.add(hidden, residual, out=slot)

    def finalize(
        self,
        hidden: torch.Tensor,
        buffer: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if buffer is None:
            return hidden
        return hidden, buffer
