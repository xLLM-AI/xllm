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

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner


class EagerRunner(BaseRunner):
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
    ) -> torch.Tensor:
        self.attention_backend.prepare(metadata)
        with forward_context(ForwardContext(self.attention_backend, self.device)):
            return self.model(input_ids, positions)
