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
    causal_conv1d as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.causal_conv1d import (
    build_causal_conv1d_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class CausalConv1dKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("batch_size", "int32"),
        DispatchField("dim", "int32"),
        DispatchField("width", "int32"),
        DispatchField("has_silu", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "bs1_d2048_w4_silu0_f16",
            "batch_size": 1,
            "dim": 2048,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d4096_w4_silu0_f16",
            "batch_size": 1,
            "dim": 4096,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d5120_w4_silu0_f16",
            "batch_size": 1,
            "dim": 5120,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d6144_w4_silu0_f16",
            "batch_size": 1,
            "dim": 6144,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        },
        {
            "variant_key": "bs1_d8192_w4_silu0_f16",
            "batch_size": 1,
            "dim": 8192,
            "width": 4,
            "has_silu": 0,
            "dtype": "float16",
        },
    ]

    @staticmethod
    def generate_source(
        batch_size: int,
        dim: int,
        width: int,
        has_silu: int,
        dtype: str,
    ) -> str:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"CausalConv1D TileLang kernel only supports dtype=float16/bfloat16, got {dtype}")
        vec_core_num = detect_vec_core_num()
        block_dim = (dim + vec_core_num - 1) // vec_core_num
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_kernel(
            width=width,
            block_dim=block_dim,
            vec_core_num=vec_core_num,
            dtype_str=dtype,
            has_silu=bool(has_silu),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config={
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_auto_sync": True,
                "tl.ascend_memory_planning": True,
            },
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
