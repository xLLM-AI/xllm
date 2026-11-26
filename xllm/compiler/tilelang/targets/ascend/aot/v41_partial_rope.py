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

from xllm.python.kernels_npu.tilelang import utils, v41_partial_rope

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (utils, v41_partial_rope)


@register_kernel
class V41PartialRopeKernel(TilelangKernel):
    KERNEL_NAME = "v41_partial_rope"
    DISPATCH_SCHEMA = [DispatchField("head_dim", "int32"), DispatchField("tile_heads", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"d{head_dim}_h{tile_heads}", "head_dim": head_dim, "tile_heads": tile_heads}
        for head_dim in (128, 512)
        for tile_heads in (1, 8)
    ]

    @staticmethod
    def generate_source(head_dim: int, tile_heads: int) -> str:
        tilelang.disable_cache()
        program = v41_partial_rope.build_v41_partial_rope_kernel(head_dim, tile_heads, utils.detect_vec_core_num())
        with tilelang.tvm.transform.PassContext(opt_level=3, config=v41_partial_rope.PASS_CONFIGS):
            return tilelang.engine.lower(program).kernel_source
