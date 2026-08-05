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

"""NPU rotary-embedding implementations."""

from __future__ import annotations

import torch
import torch_npu


def interleaved_rotary_embedding(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_heads, head_dim = value.shape
    output = torch_npu.npu_interleave_rope(
        value.view(num_tokens, num_heads, 1, head_dim), cosine, sine
    )
    return output.view(num_tokens, num_heads, head_dim)
