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

import argparse
from pathlib import Path

import tilelang

from xllm.python.kernels_npu.tilelang import (
    rope as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.rope import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    DEFAULT_DTYPE,
    DEFAULT_HEAD_DIM,
    DEFAULT_ROPE_DIM,
    FIXED_UB_BUFFER_BYTES,
    REF_CHECK_NUM_TOKENS,
    SECONDARY_HEAD_DIM,
    SECONDARY_ROPE_DIM,
    _run_ref_check,
    build_rope_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class RopeKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("head_dim", "int32"),
        DispatchField("rope_dim", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": "hd128_rd128_bf16",
            "head_dim": SECONDARY_HEAD_DIM,
            "rope_dim": SECONDARY_ROPE_DIM,
            "dtype": DEFAULT_DTYPE,
        },
        {
            "variant_key": "hd576_rd64_bf16",
            "head_dim": DEFAULT_HEAD_DIM,
            "rope_dim": DEFAULT_ROPE_DIM,
            "dtype": DEFAULT_DTYPE,
        },
    ]

    @staticmethod
    def generate_source(head_dim: int, rope_dim: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"RoPE TileLang kernel only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        tilelang.disable_cache()
        vec_core_num = detect_vec_core_num()
        ub_buffer_bytes = FIXED_UB_BUFFER_BYTES
        tilelang_kernel = build_rope_kernel(
            head_dim=head_dim,
            rope_dim=rope_dim,
            vec_core_num=vec_core_num,
            ub_buffer_bytes=ub_buffer_bytes,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AscendC source for RoPE AOT kernel.")
    parser.add_argument("--output", required=True, help="Output AscendC .cpp file")
    parser.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument("--rope-dim", type=int, default=DEFAULT_ROPE_DIM)
    parser.add_argument("--dtype", default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        RopeKernel.generate_source(
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            dtype=args.dtype,
        ),
        encoding="utf-8",
    )

    if not args.skip_ref_check:
        _run_ref_check(
            num_tokens=REF_CHECK_NUM_TOKENS,
            head_dim=args.head_dim,
            rope_dim=args.rope_dim,
            vec_core_num=detect_vec_core_num(),
            ub_buffer_bytes=FIXED_UB_BUFFER_BYTES,
        )


if __name__ == "__main__":
    main()
