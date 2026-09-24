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

from xllm.python.kernels_npu.tilelang import greedy_prefix_verify as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.greedy_prefix_verify import (
    DEFAULT_TASK_COUNT,
    GREEDY_PREFIX_VERIFY_PASS_CONFIGS,
    build_greedy_prefix_verify_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class GreedyPrefixVerifyKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("task_count", "int32"),
        DispatchField("draft_bits", "int32"),
        DispatchField("target_bits", "int32"),
        DispatchField("bonus_bits", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"tasks_{task_count}_d{draft_bits}_t{target_bits}_b{bonus_bits}",
            "task_count": task_count,
            "draft_bits": draft_bits,
            "target_bits": target_bits,
            "bonus_bits": bonus_bits,
        }
        for task_count in sorted({2, DEFAULT_TASK_COUNT})
        for draft_bits in (32, 64)
        for target_bits in (32, 64)
        for bonus_bits in (32, 64)
    ]

    @staticmethod
    def generate_source(task_count: int, draft_bits: int, target_bits: int, bonus_bits: int) -> str:
        tilelang.disable_cache()
        kernel = build_greedy_prefix_verify_kernel(task_count, draft_bits, target_bits, bonus_bits)
        with tilelang.tvm.transform.PassContext(opt_level=3, config=GREEDY_PREFIX_VERIFY_PASS_CONFIGS):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
