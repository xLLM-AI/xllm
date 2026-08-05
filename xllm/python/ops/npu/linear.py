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

"""NPU weight preparation for linear operators."""

from __future__ import annotations

import torch
import torch_npu

_FRACTAL_NZ_FORMAT = 29


def prepare_row_parallel_weight(weight: torch.Tensor) -> torch.Tensor:
    transposed = weight.transpose(0, 1).contiguous()
    return torch_npu.npu_format_cast(transposed, _FRACTAL_NZ_FORMAT)
