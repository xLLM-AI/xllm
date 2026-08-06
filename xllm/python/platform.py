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

"""Active platform state for the Python model executor."""

from __future__ import annotations

from functools import cache

import torch


@cache
def current_platform() -> str:
    """Detect the active runtime using the same priority as ``setup.py``."""
    if torch.cuda.is_available():
        return "cuda"
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return "cpu"
    if torch.npu.is_available():
        return "npu"
    return "cpu"


def is_npu() -> bool:
    """Return whether the active executor platform is NPU."""
    return current_platform() == "npu"


def is_gpu() -> bool:
    """Return whether the active executor platform is CUDA GPU."""
    return current_platform() == "cuda"
