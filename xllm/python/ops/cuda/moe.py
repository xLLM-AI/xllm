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

"""CUDA capability checks for public MoE operators."""

from __future__ import annotations

import torch

_CUTLASS_CAPABILITIES = frozenset((9, 10, 12))


def supports_cutlass(device: torch.device) -> bool:
    if not torch.cuda.is_available():
        return False
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    major, _ = torch.cuda.get_device_capability(device_index)
    return major in _CUTLASS_CAPABILITIES
