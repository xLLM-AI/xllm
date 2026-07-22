# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from xllm.python.attention.backend import AttentionBackend, AttentionMetadata


class BaseRunner(ABC):
    def __init__(
        self, model: nn.Module, attention_backend: AttentionBackend, device: torch.device,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.device = device

    @abstractmethod
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        pass
