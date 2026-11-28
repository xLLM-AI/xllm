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

from xllm.python.kernels_npu.tilelang import utils, v41_c2_pool_norm

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (v41_c2_pool_norm, utils)


@register_kernel
class V41C2PoolNormKernel(TilelangKernel):
    KERNEL_NAME = "v41_c2_pool_norm"
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32")]
    SPECIALIZATIONS = [{"variant_key": "d512", "head_dim": v41_c2_pool_norm.C2_HEAD_DIM}]

    @staticmethod
    def generate_source(head_dim: int) -> str:
        if head_dim != v41_c2_pool_norm.C2_HEAD_DIM:
            raise ValueError(f"Unsupported V4.1 C2 head dimension {head_dim}")
        tilelang.disable_cache()
        func = v41_c2_pool_norm.build_c2_pool_norm_kernel(vec_core_num=utils.detect_vec_core_num())
        with tilelang.tvm.transform.PassContext(opt_level=3, config=v41_c2_pool_norm.C2_PASS_CONFIGS):
            return tilelang.engine.lower(func).kernel_source
