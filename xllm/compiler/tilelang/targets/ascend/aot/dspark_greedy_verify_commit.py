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
from xllm.python.kernels_npu.tilelang import dspark_greedy_verify_commit as kernel_impl

DEPENDENCY_MODULES = (kernel_impl,)


@register_kernel
class DSparkGreedyVerifyCommitKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("max_batch_size", "int32"), DispatchField("draft_width", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"batch_{batch}_draft_{width}", "max_batch_size": batch, "draft_width": width}
        for batch in (1, 2, 4, 8, 16, 32, 64, 128)
        for width in (5, 7)
    ]

    @staticmethod
    def generate_source(max_batch_size: int, draft_width: int) -> str:
        tilelang.disable_cache()
        kernel = kernel_impl.build_dspark_greedy_verify_commit_kernel(max_batch_size, draft_width)
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=kernel_impl.DSPARK_GREEDY_VERIFY_COMMIT_PASS_CONFIGS
        ):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
