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

import tilelang
from compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel

from xllm.python.kernels_npu.tilelang import sampling_common
from xllm.python.kernels_npu.tilelang import sampling_logits as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.sampling_logits import (
    build_sampling_logits_kernel,
)

DEPENDENCY_MODULES = (kernel_impl, sampling_common, tilelang_utils)


@register_kernel
class SamplingLogitsKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("task_count", "int32"),
        DispatchField("dtype", "dtype"),
        DispatchField("normalize_only", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": ("normalization_" if normalize_only else "") + dtype,
            "task_count": 48,
            "dtype": dtype,
            "normalize_only": normalize_only,
        }
        for normalize_only in (0, 1)
        for dtype in ("float32", "bfloat16", "float16")
    ]

    @staticmethod
    def generate_source(task_count: int, dtype: str, normalize_only: int) -> str:
        tilelang.disable_cache()
        kernel = build_sampling_logits_kernel(dtype, task_count, bool(normalize_only))
        with tilelang.tvm.transform.PassContext(opt_level=3, config=sampling_common.SAMPLING_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
