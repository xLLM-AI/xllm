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

from xllm.python.kernels_npu.tilelang import shared_full_indices as kernel_impl

DEPENDENCY_MODULES = (kernel_impl,)


@register_kernel
class SharedFullIndicesKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("tile_size", "int32")]
    SPECIALIZATIONS = [
        {
            "variant_key": "tile128",
            "tile_size": kernel_impl.SHARED_FULL_INDICES_TILE,
        }
    ]

    @staticmethod
    def generate_source(tile_size: int) -> str:
        tilelang.disable_cache()
        kernel = kernel_impl.build_shared_full_indices_kernel(tile_size)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=kernel_impl.SHARED_FULL_INDICES_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
