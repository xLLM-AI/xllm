#!/usr/bin/env python3

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T

from scripts.logger import logger

from .utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
)
from ....common.spec import DispatchField, TilelangKernel, register_kernel

DEFAULT_HEAD_DIM = 576
DEFAULT_ROPE_DIM = 64
DEFAULT_DTYPE = "bf16"
SECONDARY_HEAD_DIM = 128
SECONDARY_ROPE_DIM = 128
VEC_NUM = 2
FIXED_UB_BUFFER_BYTES = 64 * 1024
REF_CHECK_NUM_TOKENS = 16
# AOT kernel tensor signatures still require static first-dim bounds.
# Keep a sufficiently large compile-time upper bound so runtime rows
# (`num_tokens * num_heads`) used by wrapper tests stay in-range.
MIN_COMPILE_NUM_TOKENS = 65536

# Per-row bytes in UB for this kernel:
# x_half(2) + x(4) + sin_half(2) + sin(4) + cos_half(2) + cos(4)
# + x_rotate(4) + out(4) + mask(4) = 30 bytes per rope element.
UB_BYTES_PER_ROW_PER_ROPE_ELEM = 30


def _derive_max_rows_num_in_ub(rope_dim: int, ub_buffer_bytes: int) -> int:
    if ub_buffer_bytes <= 0:
        raise ValueError(f"ub_buffer_bytes({ub_buffer_bytes}) must be > 0")
    if rope_dim <= 0:
        raise ValueError(f"rope_dim({rope_dim}) must be > 0")

    bytes_per_row = UB_BYTES_PER_ROW_PER_ROPE_ELEM * rope_dim
    max_rows = ub_buffer_bytes // bytes_per_row
    if max_rows <= 0:
        raise ValueError(
            "UB budget is too small for current rope_dim: "
            f"ub_buffer_bytes={ub_buffer_bytes}, rope_dim={rope_dim}"
        )
    return max_rows


def build_rope_kernel(
    head_dim: int,
    rope_dim: int,
    vec_core_num: int,
    ub_buffer_bytes: int,
):
    if rope_dim % 2 != 0:
        raise ValueError(f"rope_dim({rope_dim}) must be even")
    if rope_dim > head_dim:
        raise ValueError(f"rope_dim({rope_dim}) must be <= head_dim({head_dim})")
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    if vec_core_num % VEC_NUM != 0:
        raise ValueError(
            f"vec_core_num({vec_core_num}) must be divisible by VEC_NUM({VEC_NUM})"
        )

    task_num = vec_core_num
    m_num = vec_core_num // VEC_NUM
    max_rows_num_in_ub = _derive_max_rows_num_in_ub(
        rope_dim=rope_dim,
        ub_buffer_bytes=ub_buffer_bytes,
    )
    # Current AOT path fixes launch block_num at compile time, so runtime input
    # shape only changes per-task workload splitting. The tensor signature still
    # needs a static upper bound for the first dimension.
    compile_num_tokens = max(task_num * max_rows_num_in_ub, MIN_COMPILE_NUM_TOKENS)
    compile_flatten_width = compile_num_tokens * head_dim
    acc_dtype = "float32"
    mask_dtype = "uint32"

    @T.prim_func
    def rope_in_place_kernel(
        x_in: T.Tensor((1, compile_flatten_width), "bfloat16"),
        sin: T.Tensor((compile_num_tokens, rope_dim), "bfloat16"),
        cos: T.Tensor((compile_num_tokens, rope_dim), "bfloat16"),
        x_out: T.Tensor((1, compile_flatten_width), "bfloat16"),
        num_tokens: T.int32,
        x_stride: T.int32,
    ):
        with T.Kernel(m_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            block_m = (num_tokens + task_num - 1) // task_num
            row_start = task_id * block_m
            rows_left = T.if_then_else(
                num_tokens > row_start, num_tokens - row_start, 0
            )
            num_rows_per_vec = T.if_then_else(
                rows_left < block_m,
                rows_left,
                block_m,
            )

            with T.Scope("V"):
                mask_ub = T.alloc_ub([1, rope_dim], mask_dtype)
                for j in T.serial(rope_dim // 2):
                    mask_ub[0, 2 * j] = 4 * (2 * j + 1)
                    mask_ub[0, 2 * j + 1] = 4 * (2 * j)

                sin_mask_ub = T.alloc_ub((rope_dim,), acc_dtype)
                T.tile.fill(sin_mask_ub, 1.0)
                for i in T.serial(rope_dim):
                    if i % 2 == 0:
                        sin_mask_ub[i] = -1.0
                x_half_ub = T.alloc_shared([1, rope_dim], "bfloat16")
                x_ub = T.alloc_shared([1, rope_dim], acc_dtype)
                sin_half_ub = T.alloc_shared([1, rope_dim], "bfloat16")
                sin_ub = T.alloc_shared([1, rope_dim], acc_dtype)
                cos_half_ub = T.alloc_shared([1, rope_dim], "bfloat16")
                cos_ub = T.alloc_shared([1, rope_dim], acc_dtype)
                x_rotate_ub = T.alloc_shared([1, rope_dim], acc_dtype)
                out_ub = T.alloc_shared([1, rope_dim], acc_dtype)

                for row_local in T.serial(num_rows_per_vec):
                    row = row_start + row_local
                    row_offset = row * x_stride
                    T.copy(x_in[0, row_offset], x_half_ub[0, :])
                    T.copy(sin[row, :], sin_half_ub[0, :])
                    T.copy(cos[row, :], cos_half_ub[0, :])

                    T.tile.cast(x_ub, x_half_ub, "CAST_NONE", rope_dim)
                    T.tile.cast(sin_ub, sin_half_ub, "CAST_NONE", rope_dim)
                    T.tile.cast(cos_ub, cos_half_ub, "CAST_NONE", rope_dim)
                    T.tile.mul(sin_ub[0, :], sin_ub[0, :], sin_mask_ub)

                    T.tile.gather(x_rotate_ub, x_ub, mask_ub, 0)
                    T.tile.mul(x_ub, x_ub, cos_ub)
                    T.tile.mul(x_rotate_ub, x_rotate_ub, sin_ub)
                    T.tile.add(out_ub, x_ub, x_rotate_ub)
                    T.tile.cast(x_half_ub, out_ub, "CAST_RINT", rope_dim)
                    T.copy(x_half_ub[0, :], x_out[0, row_offset])

    return rope_in_place_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def rope_in_place_kernel_jit(
    head_dim: int,
    rope_dim: int,
    vec_core_num: int,
    ub_buffer_bytes: int,
):
    return build_rope_kernel(
        head_dim=head_dim,
        rope_dim=rope_dim,
        vec_core_num=vec_core_num,
        ub_buffer_bytes=ub_buffer_bytes,
    )


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
            raise ValueError(
                f"RoPE TileLang kernel only supports dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        tilelang.disable_cache()
        vec_core_num = detect_vec_core_num()
        ub_buffer_bytes = FIXED_UB_BUFFER_BYTES
        tilelang_kernel = build_rope_kernel(
            head_dim=head_dim,
            rope_dim=rope_dim,
            vec_core_num=vec_core_num,
            ub_buffer_bytes=ub_buffer_bytes,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _torch_rope_ref_rows(
    x: "torch.Tensor",
    sin: "torch.Tensor",
    cos: "torch.Tensor",
    dim_start: int,
) -> "torch.Tensor":
    import torch

    x_fp32 = x.to(torch.float32)
    sin_fp32 = sin.to(torch.float32)
    cos_fp32 = cos.to(torch.float32)
    rope_dim = sin_fp32.shape[1]
    x_part = x_fp32[:, dim_start : dim_start + rope_dim]
    x_reshape = x_part.reshape(x_part.shape[0], -1, 2)
    x0 = x_reshape[:, :, 0]
    x1 = x_reshape[:, :, 1]
    x_rot = torch.stack([-x1, x0], dim=-1).reshape_as(x_part)

    out = x.clone()
    out[:, dim_start : dim_start + rope_dim] = (
        x_part * cos_fp32 + x_rot * sin_fp32
    ).to(torch.bfloat16)
    return out


def _run_ref_check(
    num_tokens: int,
    head_dim: int,
    rope_dim: int,
    vec_core_num: int,
    ub_buffer_bytes: int,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning("Skip RoPE reference check: NPU is not available")
        return

    torch.manual_seed(42)
    device = torch.device("npu")
    x_in = torch.randn((num_tokens, head_dim), device=device, dtype=torch.bfloat16)
    sin = torch.randn((num_tokens, rope_dim), device=device, dtype=torch.bfloat16)
    cos = torch.randn((num_tokens, rope_dim), device=device, dtype=torch.bfloat16)
    x_out = x_in.clone()
    x_in_flat = x_in.view(1, -1)
    x_out_flat = x_out.view(1, -1)
    kernel = rope_in_place_kernel_jit(
        head_dim=head_dim,
        rope_dim=rope_dim,
        vec_core_num=vec_core_num,
        ub_buffer_bytes=ub_buffer_bytes,
    )
    kernel(x_in_flat, sin, cos, x_out_flat, num_tokens, head_dim)
    torch.npu.synchronize()

    x_ref = _torch_rope_ref_rows(x_in, sin, cos, 0)
    torch.testing.assert_close(x_out, x_ref, rtol=1e-3, atol=1e-3)
    logger.info("RoPE output matches torch reference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for RoPE AOT kernel."
    )
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
