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

from xllm.compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel
from xllm.python.kernels_npu.tilelang import dspark_pack_verify_inputs as kernel_impl

DEPENDENCY_MODULES = (kernel_impl,)


@register_kernel
class DSparkPackVerifyInputsKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("max_batch_size", "int32"),
        DispatchField("device_control", "int32"),
        DispatchField("draft_width", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"batch_{batch}_control_{control}_draft_{width}",
            "max_batch_size": batch,
            "device_control": control,
            "draft_width": width,
        }
        for batch in (1, 2, 4, 8, 16, 32, 64, 128)
        for control in (0, 1)
        for width in (5, 7)
    ]

    @staticmethod
    def generate_source(max_batch_size: int, device_control: int, draft_width: int) -> str:
        tilelang.disable_cache()
        kernel = kernel_impl.build_dspark_pack_verify_inputs_kernel(max_batch_size, bool(device_control), draft_width)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=kernel_impl.DSPARK_PACK_VERIFY_INPUTS_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
