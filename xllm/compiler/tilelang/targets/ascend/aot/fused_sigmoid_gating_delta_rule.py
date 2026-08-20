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
    fused_sigmoid_gating_delta_rule as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.fused_sigmoid_gating_delta_rule import (
    _SIGMOID_PASS_CONFIGS,
    DEFAULT_ACCUM_DTYPE,
    DEFAULT_DK,
    DEFAULT_DTYPE,
    DEFAULT_DV,
    DEFAULT_NK,
    DEFAULT_NUM_CORES,
    DEFAULT_NV,
    DEFAULT_SOFTPLUS_BETA,
    DEFAULT_USE_QK_L2NORM,
    NUM_SEQS_SPECIALIZATIONS,
    REF_CHECK_NUM_SEQS,
    _auto_block_v,
    build_fused_sigmoid_gating_delta_rule_kernel,
)
from xllm.python.kernels_npu.tilelang.fused_sigmoid_gating_delta_rule import (
    main as run_reference,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class FusedSigmoidGatingDeltaRuleKernel(TilelangKernel):
    DISPATCH_SCHEMA = [
        DispatchField("max_num_seqs", "int32"),
        DispatchField("nk", "int32"),
        DispatchField("nv", "int32"),
        DispatchField("dk", "int32"),
        DispatchField("dv", "int32"),
        DispatchField("block_v", "int32"),
        DispatchField("use_qk_l2norm", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": (f"ns{num_seqs}_nk{nk}_nv{nv}_dk{dk}_dv{dv}_bv{block_v}_l2{int(use_qk_l2norm)}_bf16"),
            "max_num_seqs": num_seqs,
            "nk": nk,
            "nv": nv,
            "dk": dk,
            "dv": dv,
            "block_v": block_v,
            "use_qk_l2norm": int(use_qk_l2norm),
            "dtype": DEFAULT_DTYPE,
        }
        for num_seqs in NUM_SEQS_SPECIALIZATIONS
        for nk, nv, dk, dv, use_qk_l2norm in sorted(
            {
                (nk // tp, nv // tp, dk, dv, use_qk_l2norm)
                for nk, nv, dk, dv, use_qk_l2norm in [
                    (16, 16, 128, 128, True),
                    (16, 32, 128, 128, True),
                    (16, 48, 128, 128, True),
                    (16, 64, 128, 128, True),
                ]
                for tp in [1, 2, 4, 8]
                if nk % tp == 0 and nv % tp == 0
            }
        )
        for block_v in [_auto_block_v(dv)]
    ]

    @staticmethod
    def generate_source(
        max_num_seqs: int,
        nk: int,
        nv: int,
        dk: int,
        dv: int,
        block_v: int,
        use_qk_l2norm: int,
        dtype: str,
    ) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"fused_sigmoid_gating_delta_rule only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        tilelang.disable_cache()
        tilelang_kernel = build_fused_sigmoid_gating_delta_rule_kernel(
            nk=nk,
            nv=nv,
            dk=dk,
            dv=dv,
            block_v=block_v,
            max_num_seqs=max_num_seqs,
            use_qk_l2norm=bool(use_qk_l2norm),
            softplus_beta=DEFAULT_SOFTPLUS_BETA,
            dtype=dtype,
            accum_dtype=DEFAULT_ACCUM_DTYPE,
            num_cores=DEFAULT_NUM_CORES,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=_SIGMOID_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for fused_sigmoid_gating_delta_rule AOT kernel."
    )
    parser.add_argument("--output", type=Path, required=True, help="Output AscendC .cpp file")
    parser.add_argument("--nk", type=int, default=DEFAULT_NK)
    parser.add_argument("--nv", type=int, default=DEFAULT_NV)
    parser.add_argument("--dk", type=int, default=DEFAULT_DK)
    parser.add_argument("--dv", type=int, default=DEFAULT_DV)
    parser.add_argument("--block-v", type=int, default=None)
    parser.add_argument("--max-num-seqs", type=int, default=NUM_SEQS_SPECIALIZATIONS[-1])
    parser.add_argument("--use-qk-l2norm", type=int, default=DEFAULT_USE_QK_L2NORM)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    block_v = args.block_v if args.block_v is not None else _auto_block_v(args.dv)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        FusedSigmoidGatingDeltaRuleKernel.generate_source(
            max_num_seqs=args.max_num_seqs,
            nk=args.nk,
            nv=args.nv,
            dk=args.dk,
            dv=args.dv,
            block_v=block_v,
            use_qk_l2norm=args.use_qk_l2norm,
            dtype=args.dtype,
        ),
        encoding="utf-8",
    )

    if not args.skip_ref_check:
        run_reference(
            seqlens=[4, 8] * (REF_CHECK_NUM_SEQS // 2),
            nk=args.nk,
            nv=args.nv,
            dk=args.dk,
            dv=args.dv,
            block_v=block_v,
            use_qk_l2norm=bool(args.use_qk_l2norm),
            softplus_beta=DEFAULT_SOFTPLUS_BETA,
        )


if __name__ == "__main__":
    main_cli()
