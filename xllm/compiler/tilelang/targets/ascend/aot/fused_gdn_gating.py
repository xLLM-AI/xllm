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
    fused_gdn_gating as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.fused_gdn_gating import (
    BATCH_SIZE_SPECIALIZATIONS,
    DEFAULT_ASCEND_PASS_CONFIGS,
    DEFAULT_DTYPE,
    DEFAULT_MAX_BATCH,
    DEFAULT_NUM_HEADS,
    REF_CHECK_NUM_BATCHES,
    REF_CHECK_NUM_HEADS,
    SUPPORTED_NUM_HEADS,
    _run_ref_suite,
    build_fused_gdn_gating_kernel,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class FusedGdnGatingKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("batch_size", "int32"),
        DispatchField("num_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": f"bs{batch_size}_nh{num_heads}_bf16",
            "batch_size": batch_size,
            "num_heads": num_heads,
            "dtype": DEFAULT_DTYPE,
        }
        for num_heads in SUPPORTED_NUM_HEADS
        for batch_size in BATCH_SIZE_SPECIALIZATIONS
    ]

    @staticmethod
    def generate_source(batch_size: int, num_heads: int, dtype: str) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"fused_gdn_gating only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        if num_heads not in SUPPORTED_NUM_HEADS:
            raise ValueError(f"fused_gdn_gating only supports num_heads in {SUPPORTED_NUM_HEADS}, got {num_heads}")
        if batch_size not in BATCH_SIZE_SPECIALIZATIONS:
            raise ValueError(
                f"fused_gdn_gating only supports batch_size in {BATCH_SIZE_SPECIALIZATIONS}, got {batch_size}"
            )
        tilelang.disable_cache()
        tilelang_kernel = build_fused_gdn_gating_kernel(
            batch_size=batch_size,
            compile_max_batch=DEFAULT_MAX_BATCH,
            num_heads=num_heads,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TileLang AscendC source for fused_gdn_gating AOT kernel.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=max(BATCH_SIZE_SPECIALIZATIONS),
        help=(f"Batch-size specialization used for source generation. Supported values: {BATCH_SIZE_SPECIALIZATIONS}"),
    )
    parser.add_argument("--num-heads", type=int, default=DEFAULT_NUM_HEADS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-batches",
        type=int,
        default=REF_CHECK_NUM_BATCHES,
        help="Batch size used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--softplus-beta",
        type=float,
        default=1.0,
        help="Softplus beta used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--softplus-threshold",
        type=float,
        default=20.0,
        help="Softplus threshold used by the optional torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-heads-list",
        type=int,
        nargs="+",
        default=list(REF_CHECK_NUM_HEADS),
        help="Head counts covered by the optional bf16 torch-reference test suite.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = FusedGdnGatingKernel.generate_source(
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_suite(
            num_batches=args.ref_num_batches,
            compile_max_batch=DEFAULT_MAX_BATCH,
            softplus_beta=args.softplus_beta,
            softplus_threshold=args.softplus_threshold,
            ref_num_heads_list=args.ref_num_heads_list,
        )


if __name__ == "__main__":
    main()
