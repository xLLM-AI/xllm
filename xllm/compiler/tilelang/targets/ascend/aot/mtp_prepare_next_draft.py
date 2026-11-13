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

from xllm.python.kernels_npu.tilelang import mtp_prepare_next_draft as kernel_impl
from xllm.python.kernels_npu.tilelang.mtp_prepare_next_draft import (
    MTP_PREPARE_NEXT_DRAFT_PASS_CONFIGS,
    build_mtp_prepare_next_draft_kernel,
    prepare_vector_only_source,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl,)


@register_kernel
class MtpPrepareNextDraftKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("preparation_mode", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": name, "preparation_mode": mode}
        for mode, name in ((0, "full_hidden"), (1, "compact_hidden"), (2, "verifier_prefix"))
    ]

    @staticmethod
    def generate_source(preparation_mode: int) -> str:
        tilelang.disable_cache()
        kernel = build_mtp_prepare_next_draft_kernel(preparation_mode)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=MTP_PREPARE_NEXT_DRAFT_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return prepare_vector_only_source(lowered.kernel_source)
