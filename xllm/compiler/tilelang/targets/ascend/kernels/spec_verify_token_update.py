#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tilelang
import tilelang.language as T

from .utils import DEFAULT_ASCEND_PASS_CONFIGS, SUPPORTED_SPEC_VERIFY_WIDTHS
from ....common.spec import DispatchField, TilelangKernel, register_kernel

SYMBOL_BUFFER_CAPACITY = T.symbolic("buffer_capacity")


def build_spec_verify_token_update_kernel(spec_width: int):
    if spec_width not in SUPPORTED_SPEC_VERIFY_WIDTHS:
        raise ValueError(f"unsupported speculative verify width: {spec_width}")

    @T.prim_func
    def spec_verify_token_update(
        base_tokens: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int32"),
        draft_token_0: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int64"),
        draft_token_1: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int64"),
        draft_token_2: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int64"),
        draft_token_3: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int64"),
        draft_token_4: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int64"),
        persistent_tokens: T.Tensor((SYMBOL_BUFFER_CAPACITY,), "int32"),
        batch_size: T.int32,
    ):
        with T.Kernel(1, is_npu=True) as (_, vid):
            if batch_size == 1:
                if vid == 0:
                    with T.Scope("V"):
                        # Use one 256-bit store for the latency-critical batch-1
                        # path. The wrapper contract requires eight writable
                        # int32 elements; clear the unused tail so replay never
                        # observes stale token values.
                        packed_ub = T.alloc_ub((8,), "int32")
                        packed_ub[0] = base_tokens[0]
                        packed_ub[1] = T.Cast("int32", draft_token_0[0])
                        packed_ub[2] = T.Cast("int32", draft_token_1[0])
                        packed_ub[3] = T.Cast("int32", draft_token_2[0])
                        if spec_width >= 5:
                            packed_ub[4] = T.Cast("int32", draft_token_3[0])
                        else:
                            packed_ub[4] = 0
                        if spec_width >= 6:
                            packed_ub[5] = T.Cast("int32", draft_token_4[0])
                        else:
                            packed_ub[5] = 0
                        packed_ub[6] = 0
                        packed_ub[7] = 0
                        T.copy(packed_ub, persistent_tokens[0])
            else:
                for batch_idx in T.serial(batch_size):
                    output_offset = batch_idx * spec_width
                    persistent_tokens[output_offset] = base_tokens[output_offset]
                    persistent_tokens[output_offset + 1] = T.Cast(
                        "int32", draft_token_0[batch_idx]
                    )
                    persistent_tokens[output_offset + 2] = T.Cast(
                        "int32", draft_token_1[batch_idx]
                    )
                    persistent_tokens[output_offset + 3] = T.Cast(
                        "int32", draft_token_2[batch_idx]
                    )
                    if spec_width >= 5:
                        persistent_tokens[output_offset + 4] = T.Cast(
                            "int32", draft_token_3[batch_idx]
                        )
                    if spec_width >= 6:
                        persistent_tokens[output_offset + 5] = T.Cast(
                            "int32", draft_token_4[batch_idx]
                        )

    return spec_verify_token_update


@register_kernel
class SpecVerifyTokenUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("spec_width", "int32")]
    SPECIALIZATIONS = [
        {"variant_key": f"w{spec_width}", "spec_width": spec_width}
        for spec_width in SUPPORTED_SPEC_VERIFY_WIDTHS
    ]

    @staticmethod
    def generate_source(spec_width: int) -> str:
        tilelang.disable_cache()
        kernel = build_spec_verify_token_update_kernel(spec_width)
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
