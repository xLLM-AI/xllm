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

from xllm.python.kernels_npu.tilelang import (
    apply_token_bitmask as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.apply_token_bitmask import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    SUPPORTED_DTYPES,
    build_apply_token_bitmask_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class ApplyTokenBitmaskKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("dtype", "dtype")]
    SPECIALIZATIONS = [
        {
            "variant_key": dtype,
            "dtype": dtype,
        }
        for dtype in SUPPORTED_DTYPES
    ]

    @staticmethod
    def generate_source(dtype: str) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_apply_token_bitmask_kernel(
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
