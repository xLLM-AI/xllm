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
    split_qkv_rmsnorm_mrope as kernel_impl,
)
from xllm.python.kernels_npu.tilelang import (
    utils as tilelang_utils,
)
from xllm.python.kernels_npu.tilelang.split_qkv_rmsnorm_mrope import (
    ALL_HEAD_CONFIGS,
    DEFAULT_DTYPE,
    DEFAULT_MROPE_SECTION,
    DEFAULT_NUM_KV_HEADS,
    DEFAULT_NUM_Q_HEADS,
    KERNEL_PASS_CONFIGS,
    NUM_TOKEN_SPECIALIZATIONS,
    REF_CHECK_EPS,
    REF_CHECK_HEAD_CONFIGS,
    REF_CHECK_NUM_TOKENS,
    SUPPORTED_HEAD_SPECS,
    _run_ref_suite,
    _validate_head_spec,
    _validate_mrope_section,
    _validate_specialization_num_tokens,
    build_split_qkv_rmsnorm_mrope_kernel,
    detect_vec_core_num,
)

from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEPENDENCY_MODULES = (kernel_impl, tilelang_utils)


@register_kernel
class SplitQkvRmsnormMropeKernel(TilelangKernel):
    KERNEL_NAME = "split_qkv_rmsnorm_mrope"
    DISPATCH_SCHEMA = [
        DispatchField("head_size", "int32"),
        DispatchField("rope_dim", "int32"),
        DispatchField("num_tokens", "int32"),
        DispatchField("num_q_heads", "int32"),
        DispatchField("num_kv_heads", "int32"),
        DispatchField("dtype", "dtype"),
    ]
    SPECIALIZATIONS = [
        {
            "variant_key": (f"hs{head_size}_rd{rope_dim}_nt{num_tokens}_qh{num_q_heads}_kvh{num_kv_heads}_bf16"),
            "head_size": head_size,
            "rope_dim": rope_dim,
            "num_tokens": num_tokens,
            "num_q_heads": num_q_heads,
            "num_kv_heads": num_kv_heads,
            "dtype": DEFAULT_DTYPE,
        }
        for head_size, rope_dim in SUPPORTED_HEAD_SPECS
        for num_tokens in NUM_TOKEN_SPECIALIZATIONS
        for num_q_heads, num_kv_heads in ALL_HEAD_CONFIGS
    ]

    @staticmethod
    def generate_source(
        head_size: int,
        rope_dim: int,
        num_tokens: int,
        num_q_heads: int,
        num_kv_heads: int,
        dtype: str,
    ) -> str:
        if dtype != DEFAULT_DTYPE:
            raise ValueError(f"split_qkv_rmsnorm_mrope only supports dtype={DEFAULT_DTYPE}, got {dtype}")
        _validate_head_spec(head_size=head_size, rope_dim=rope_dim)
        _validate_specialization_num_tokens(num_tokens)
        tilelang.disable_cache()
        vec_core_num = detect_vec_core_num()
        tilelang_kernel = build_split_qkv_rmsnorm_mrope_kernel(
            head_size=head_size,
            rope_dim=rope_dim,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            vec_core_num=vec_core_num,
            max_num_tokens=num_tokens,
        )
        with tilelang.tvm.transform.PassContext(opt_level=3, config=KERNEL_PASS_CONFIGS):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _parse_head_config(text: str) -> tuple[int, int]:
    q_heads_text, sep, kv_heads_text = text.partition(":")
    if sep != ":":
        raise argparse.ArgumentTypeError(f"Invalid head config '{text}'. Expected format <num_q_heads>:<num_kv_heads>.")
    try:
        return int(q_heads_text), int(kv_heads_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid head config '{text}'. Expected integer pair.") from exc


def parse_args() -> argparse.Namespace:
    default_head_size, default_rope_dim = SUPPORTED_HEAD_SPECS[0]
    parser = argparse.ArgumentParser(
        description=("Generate TileLang AscendC source for split_qkv_rmsnorm_mrope AOT kernel.")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-size", type=int, default=default_head_size)
    parser.add_argument("--rope-dim", type=int, default=default_rope_dim)
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=NUM_TOKEN_SPECIALIZATIONS[-1],
        help=(
            "Launch token specialization bucket used for source generation. "
            f"Supported values: {NUM_TOKEN_SPECIALIZATIONS}"
        ),
    )
    parser.add_argument("--num-q-heads", type=int, default=DEFAULT_NUM_Q_HEADS)
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULT_NUM_KV_HEADS)
    parser.add_argument("--eps", type=float, default=REF_CHECK_EPS)
    parser.add_argument("--dtype", type=str, default=DEFAULT_DTYPE)
    parser.add_argument(
        "--skip-ref-check",
        action="store_true",
        help="Skip runtime torch-reference check.",
    )
    parser.add_argument(
        "--ref-num-tokens",
        type=int,
        default=REF_CHECK_NUM_TOKENS,
        help="Token count used by the optional torch-reference test suite.",
    )
    parser.add_argument(
        "--ref-head-configs",
        type=_parse_head_config,
        nargs="+",
        default=list(REF_CHECK_HEAD_CONFIGS),
        help=(
            "Head configurations covered by the optional runtime check. "
            "Each item must use <num_q_heads>:<num_kv_heads>."
        ),
    )
    parser.add_argument(
        "--mrope-section",
        type=int,
        nargs=3,
        default=list(DEFAULT_MROPE_SECTION),
        metavar=("T", "H", "W"),
        help="MRoPE section [t h w] used by the optional test.",
    )
    parser.add_argument(
        "--is-interleaved",
        action="store_true",
        default=True,
        help="Use interleaved MRoPE layout (default: True for Qwen3.5).",
    )
    parser.add_argument(
        "--no-interleaved",
        dest="is_interleaved",
        action="store_false",
        help="Use non-interleaved MRoPE layout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_mrope_section(
        rope_dim=args.rope_dim,
        mrope_section=tuple(args.mrope_section),
    )
    source = SplitQkvRmsnormMropeKernel.generate_source(
        head_size=args.head_size,
        rope_dim=args.rope_dim,
        num_tokens=args.num_tokens,
        num_q_heads=args.num_q_heads,
        num_kv_heads=args.num_kv_heads,
        dtype=args.dtype,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(source, encoding="utf-8")

    if not args.skip_ref_check:
        _run_ref_suite(
            num_tokens=args.ref_num_tokens,
            head_configs=args.ref_head_configs,
            head_size=args.head_size,
            rope_dim=args.rope_dim,
            eps=args.eps,
            mrope_section=tuple(args.mrope_section),
            is_interleaved=args.is_interleaved,
        )


if __name__ == "__main__":
    main()
