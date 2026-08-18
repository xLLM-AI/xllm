#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tilelang
import tilelang.language as T

from ....common.spec import DispatchField, TilelangKernel, register_kernel
from .utils import DEFAULT_ASCEND_PASS_CONFIGS, detect_vec_core_num

MAX_NUM_ROWS = 4096
MAX_VOCAB_SIZE = 262144
MAX_MASK_WORDS = (MAX_VOCAB_SIZE + 31) // 32
MAX_LOGIT_ELEMENTS = MAX_NUM_ROWS * MAX_VOCAB_SIZE
MAX_MASK_ELEMENTS = MAX_NUM_ROWS * MAX_MASK_WORDS
SUPPORTED_DTYPES = ("bf16", "float16", "float32")
TENSOR_DTYPES = {
    "bf16": "bfloat16",
    "float16": "float16",
    "float32": "float32",
}
VEC_NUM = 2
DISALLOWED_TOKEN_MASK = -1.0e9


def build_apply_token_bitmask_kernel(*, dtype: str, vec_core_num: int):
    if dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"apply_token_bitmask only supports {SUPPORTED_DTYPES}, got {dtype}")
    if vec_core_num <= 0 or vec_core_num % VEC_NUM != 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be positive and divisible by {VEC_NUM}")

    tensor_dtype = TENSOR_DTYPES[dtype]
    block_num = vec_core_num // VEC_NUM
    task_num = vec_core_num

    @T.prim_func
    def apply_token_bitmask_kernel(
        logits: T.Tensor((MAX_LOGIT_ELEMENTS,), tensor_dtype),
        bitmask: T.Tensor((MAX_MASK_ELEMENTS,), "int32"),
        num_rows: T.int32,
        vocab_size: T.int32,
        num_words: T.int32,
    ):
        with T.Kernel(block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total_words = num_rows * num_words
            words_per_task = (total_words + task_num - 1) // task_num
            word_start = task_id * words_per_task
            words_left = T.if_then_else(total_words > word_start, total_words - word_start, 0)
            word_count = T.if_then_else(words_left < words_per_task, words_left, words_per_task)

            with T.Scope("V"):
                logits_input_ub = T.alloc_ub((128,), tensor_dtype)
                logits_fp32_ub = T.alloc_ub((64,), "float32")
                additive_fp32_ub = T.alloc_ub((64,), "float32")
                zero_fp32_ub = T.alloc_ub((64,), "float32")
                disallowed_fp32_ub = T.alloc_ub((64,), "float32")
                bit_values_ub = T.alloc_ub((64,), "int32")
                packed_word_ub = T.alloc_ub((64,), "int32")
                allowed_bits_ub = T.alloc_ub((64,), "int32")
                zero_int32_ub = T.alloc_ub((64,), "int32")
                allowed_mask_ub = T.alloc_ub((8,), "uint8")

                T.tile.fill(zero_fp32_ub, 0.0)
                T.tile.fill(disallowed_fp32_ub, DISALLOWED_TOKEN_MASK)
                T.tile.fill(bit_values_ub, 0)
                T.tile.fill(zero_int32_ub, 0)
                for bit_index in T.serial(32):
                    if bit_index == 31:
                        bit_values_ub[bit_index] = -(1 << 31)
                    else:
                        bit_values_ub[bit_index] = 1 << bit_index

                for local_word in T.serial(word_count):
                    flat_word = word_start + local_word
                    row = flat_word // num_words
                    word_index = flat_word - row * num_words
                    packed_word = bitmask[flat_word]
                    token_base = word_index * 32
                    logit_base = row * vocab_size + token_base

                    # All-one words are common for unconstrained rows and do
                    # not need to touch logits.
                    with T.If(packed_word != -1), T.Then():
                        T.copy(logits[logit_base], logits_input_ub[0:32])
                        if dtype == "float32":
                            T.copy(
                                logits_input_ub[0:32],
                                logits_fp32_ub[0:32],
                            )
                        else:
                            T.tile.cast(
                                logits_fp32_ub,
                                logits_input_ub,
                                "CAST_NONE",
                                32,
                            )
                        T.tile.fill(packed_word_ub, packed_word)
                        T.tile.bitwise_and(
                            allowed_bits_ub,
                            packed_word_ub,
                            bit_values_ub,
                        )
                        T.tile.compare(
                            allowed_mask_ub,
                            allowed_bits_ub,
                            zero_int32_ub,
                            "NE",
                        )
                        T.tile.select(
                            additive_fp32_ub,
                            allowed_mask_ub,
                            disallowed_fp32_ub,
                            zero_fp32_ub,
                            "VSEL_TENSOR_TENSOR_MODE",
                        )
                        T.tile.add(
                            logits_fp32_ub,
                            logits_fp32_ub,
                            additive_fp32_ub,
                        )
                        if dtype == "float32":
                            T.copy(
                                logits_fp32_ub[0:32],
                                logits_input_ub[0:32],
                            )
                        else:
                            T.tile.cast(
                                logits_input_ub,
                                logits_fp32_ub,
                                "CAST_RINT",
                                32,
                            )
                        T.copy(logits_input_ub[0:32], logits[logit_base])

    return apply_token_bitmask_kernel


@register_kernel
class ApplyTokenBitmaskKernel(TilelangKernel):
    DISPATCH_SCHEMA = [DispatchField("dtype", "dtype")]
    SPECIALIZATIONS = [
        {
            "variant_key": dtype,
            "dtype": dtype,
        }
        for dtype in SUPPORTED_DTYPES
    ]

    @staticmethod
    def generate_source(dtype: str) -> str:
        tilelang.disable_cache()
        tilelang_kernel = build_apply_token_bitmask_kernel(
            dtype=dtype,
            vec_core_num=detect_vec_core_num(),
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source
