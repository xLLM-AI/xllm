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
    sfa_dcp_remap as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.sfa_dcp_remap import (
    DEFAULT_REMAP_TOPK,
    REMAP_PASS_CONFIGS,
    build_remap_topk_kernel,
)
from xllm.python.kernels_npu.tilelang.utils import detect_vec_core_num

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class SfaDcpRemapKernel(TilelangKernel):
    KERNEL_NAME = "sfa_dcp_remap"
    DISPATCH_SCHEMA = [
        DispatchField("topk", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"tk{DEFAULT_REMAP_TOPK}",
            "topk": DEFAULT_REMAP_TOPK,
        }
    ]

    @staticmethod
    def generate_source(topk: int) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_remap_topk_kernel(
            vec_core_num=detect_vec_core_num(),
            topk=topk,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=REMAP_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
