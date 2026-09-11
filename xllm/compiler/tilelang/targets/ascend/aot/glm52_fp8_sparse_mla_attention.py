#!/usr/bin/env python3

# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from pathlib import Path

import tilelang
from compiler.tilelang.common.spec import DispatchField, TilelangKernel, register_kernel

from xllm.python.kernels_npu.tilelang import glm52_fp8_sparse_mla_attention as kernel_impl
from xllm.python.kernels_npu.tilelang import utils as tilelang_utils
from xllm.python.kernels_npu.tilelang.glm52_fp8_sparse_mla_attention import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    DEFAULT_DTYPE,
    SUPPORTED_NUM_HEADS,
    build_glm52_fp8_sparse_mla_attention_kernel,
)

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class Glm52Fp8SparseMlaAttentionKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("num_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"h{num_heads}_bf16",
            "num_heads": num_heads,
            "dtype": DEFAULT_DTYPE,
        }
        for num_heads in SUPPORTED_NUM_HEADS
    ]

    @staticmethod
    def generate_source(num_heads: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"GLM-5.2 FP8 sparse MLA attention only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        tilelang.disable_cache()
        tilelang_kernel = build_glm52_fp8_sparse_mla_attention_kernel(
            num_heads=num_heads,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3,
            config=DEFAULT_ASCEND_PASS_CONFIGS,
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the GLM-5.2 FP8 sparse MLA TileLang Ascend-C source.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        Glm52Fp8SparseMlaAttentionKernel.generate_source(
            num_heads=args.num_heads,
            dtype=args.dtype,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
