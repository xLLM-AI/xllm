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

from xllm.python.kernels_npu.tilelang import utils, v41_index_query_quant

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (v41_index_query_quant, utils)


@register_kernel
class V41IndexQueryQuantKernel(TilelangKernel):
    KERNEL_NAME = "v41_index_query_quant"
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32")]
    SPECIALIZATIONS = [{"variant_key": "d128", "head_dim": v41_index_query_quant.INDEX_QUERY_DIM}]

    @staticmethod
    def generate_source(head_dim: int) -> str:
        if head_dim != v41_index_query_quant.INDEX_QUERY_DIM:
            raise ValueError(f"Unsupported V4.1 index query head dimension {head_dim}")
        tilelang.disable_cache()
        func = v41_index_query_quant.build_index_query_quant_kernel(vec_core_num=utils.detect_vec_core_num())
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=v41_index_query_quant.INDEX_QUERY_QUANT_PASS_CONFIGS
        ):
            return tilelang.engine.lower(func).kernel_source
