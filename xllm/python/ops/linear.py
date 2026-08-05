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

"""Platform-neutral weight preparation for linear layers."""

from __future__ import annotations

import torch

from xllm.python import platform


def prepare_row_parallel_weight(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, bool]:
    """Prepare a row-parallel weight and report whether it became ``[K, N]``."""
    if weight.device.type == "cpu" or not platform.is_npu():
        return weight, False
    from xllm.python.ops.npu.linear import prepare_row_parallel_weight as prepare

    return prepare(weight), True


__all__ = ["prepare_row_parallel_weight"]
