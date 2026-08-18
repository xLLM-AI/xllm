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
    chunk_gated_delta_rule_fwd_h as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.chunk_gated_delta_rule_fwd_h import (
    COMPILE_BT,
    DEFAULT_DTYPE,
    _build_chunk_gated_delta_rule_fwd_h_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

_AOT_PASS_CONFIGS = {
    "tl.ascend_auto_sync": False,
    "tl.ascend_auto_cv_combine": False,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_memory_planning": False,
}

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class ChunkGatedDeltaRuleFwdHKernel(TilelangKernel):
    KERNEL_NAME = "chunk_gated_delta_rule_fwd_h"
    DISPATCH_SCHEMA = [
        DispatchField("H", "int32"),
        DispatchField("Hg", "int32"),
        DispatchField("K", "int32"),
        DispatchField("V", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"h{hv}_hg{hg}_k{k}_v{v}_bf16",
            "H": hv,
            "Hg": hg,
            "K": k,
            "V": v,
            "dtype": DEFAULT_DTYPE,
        }
        for hv, hg, k, v in sorted(
            {
                (h // tp, hg // tp, k, v)
                for h, hg, k, v in [
                    (16, 16, 128, 128),
                    (32, 16, 128, 128),
                    (48, 16, 128, 128),
                    (64, 16, 128, 128),
                ]
                for tp in [1, 2, 4, 8]
                if h % tp == 0 and hg % tp == 0 and h // tp >= hg // tp
            }
        )
    ]

    @staticmethod
    def generate_source(H: int, Hg: int, K: int, V: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"chunk_gated_delta_rule_fwd_h only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        tilelang.disable_cache()
        tilelang_kernel = _build_chunk_gated_delta_rule_fwd_h_kernel(
            H=H,
            Hg=Hg,
            K=K,
            V=V,
            dtype=dtype,
            bt=COMPILE_BT,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=_AOT_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
