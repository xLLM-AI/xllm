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
from xllm.python.kernels_npu.tilelang import seeded_sample as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.seeded_sample import (
    build_seeded_sample_kernel,
)

DEPENDENCY_MODULES = (kernel_impl, sampling_common, tilelang_utils)


@register_kernel
class SeededSampleKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("task_count", "int32")]
    SPECIALIZATIONS = [{"variant_key": "tasks_48", "task_count": 48}]

    @staticmethod
    def generate_source(task_count: int) -> str:
        tilelang.disable_cache()
        kernel = build_seeded_sample_kernel(task_count)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=sampling_common.SAMPLING_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
