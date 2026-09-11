# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

from xllm.python.kernels_npu.tilelang import fp8_cache_write as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.fp8_cache_write import (
    SUPPORTED_HEAD_DIMS,
    build_fp8_cache_write_kernel,
)
from xllm.python.kernels_npu.tilelang.utils import DEFAULT_ASCEND_PASS_CONFIGS

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class Fp8CacheWriteKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32")]
    SPECIALIZATIONS = [{"variant_key": f"d{head_dim}", "head_dim": head_dim} for head_dim in SUPPORTED_HEAD_DIMS]

    @staticmethod
    def generate_source(head_dim: int) -> str:
        tilelang.disable_cache()
        kernel = build_fp8_cache_write_kernel(head_dim)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
