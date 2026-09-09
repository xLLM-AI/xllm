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
    spec_verify_attention_tiling_update as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.spec_verify_attention_tiling_update import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    SUPPORTED_SPEC_VERIFY_WIDTHS,
    build_spec_verify_attention_tiling_update_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class SpecVerifyAttentionTilingUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("spec_width", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"attn_tiling_w{spec_width}",
            "spec_width": spec_width,
        }
        for spec_width in SUPPORTED_SPEC_VERIFY_WIDTHS
    ]

    @staticmethod
    def generate_source(spec_width: int) -> str:
        tilelang.disable_cache()
        kernel = build_spec_verify_attention_tiling_update_kernel(spec_width)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
