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

SYMBOL_NUM_ROWS = T.symbolic("num_rows")
SYMBOL_TILING_WORDS = T.symbolic("tiling_words")


def build_spec_verify_attention_tiling_update_kernel(spec_width: int):
    if spec_width not in SUPPORTED_SPEC_VERIFY_WIDTHS:
        raise ValueError(f"unsupported MTP tiling width: {spec_width}")

    @T.prim_func
    def spec_verify_attention_tiling_update(
        src_kv_seq_lens: T.Tensor((SYMBOL_NUM_ROWS,), "int32"),
        tiling_data: T.Tensor((SYMBOL_TILING_WORDS,), "int32"),
        max_kv: T.int32,
        kv_split_length: T.int32,
        max_kv_offset: T.int32,
        kv_split_length_offset: T.int32,
        row_kv_offset: T.int32,
        row_stride: T.int32,
    ):
        with T.Kernel(1, is_npu=True):
            tiling_data[max_kv_offset] = max_kv
            tiling_data[kv_split_length_offset] = kv_split_length
            for i in T.serial(SYMBOL_NUM_ROWS):
                kv_len = src_kv_seq_lens[i]
                tiling_data[row_kv_offset + i * row_stride] = kv_len

    return spec_verify_attention_tiling_update


@register_kernel
class SpecVerifyAttentionTilingUpdateKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("spec_width", "int32"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"w{spec_width}",
            "spec_width": spec_width,
        }
        for spec_width in SUPPORTED_SPEC_VERIFY_WIDTHS
    ]

    @staticmethod
    def generate_source(spec_width: int) -> str:
        tilelang.disable_cache()
        kernel = build_spec_verify_attention_tiling_update_kernel(spec_width)
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            lowered = tilelang.engine.lower(kernel)
        return lowered.kernel_source
