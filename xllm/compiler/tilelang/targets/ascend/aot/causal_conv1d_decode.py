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
    causal_conv1d_decode as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.causal_conv1d_decode import (
    DIM_PER_CORE,
    build_causal_conv1d_decode_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class CausalConv1dDecodeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("dim", "int32"),
        DispatchField("width", "int32"),
        DispatchField("has_silu", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"d{d}_w4_silu{s}_bf16",
            "dim": d,
            "width": 4,
            "has_silu": s,
            "dtype": "bfloat16",
        }
        for d in sorted(
            {dim // tp for dim in [2048, 4096, 5120, 6144, 8192, 10240] for tp in [1, 2, 4, 8] if dim % tp == 0}
        )
        for s in [0, 1]
    ]

    @staticmethod
    def generate_source(
        dim: int,
        width: int,
        has_silu: int,
        dtype: str,
    ) -> str:
        if dtype not in ("float16", "bfloat16"):
            raise ValueError(f"CausalConv1D Decode TileLang kernel only supports dtype=float16/bfloat16, got {dtype}")
        dim_chunks = (dim + DIM_PER_CORE - 1) // DIM_PER_CORE
        tilelang.disable_cache()
        tilelang_kernel = build_causal_conv1d_decode_kernel(
            width=width,
            dim_chunks=dim_chunks,
            dim_per_core=DIM_PER_CORE,
            dtype_str=dtype,
            has_silu=bool(has_silu),
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config={
                "tl.ascend_auto_cv_combine": True,
                "tl.ascend_auto_sync": False,
                "tl.ascend_memory_planning": True,
            },
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
