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

"""CANN Triton helpers shared by NPU kernels."""

from __future__ import annotations

import torch
import triton
import triton.language.extra.cann.extension as _cann_ext

insert_slice = _cann_ext.insert_slice
extract_slice = _cann_ext.extract_slice
get_element = _cann_ext.get_element

_NUM_VECTORCORE = -1


def get_vectorcore_num() -> int:
    global _NUM_VECTORCORE
    if _NUM_VECTORCORE == -1:
        props = triton.runtime.driver.active.utils.get_device_properties(torch.npu.current_device())
        _NUM_VECTORCORE = props.get("num_vectorcore", -1)
    return _NUM_VECTORCORE
