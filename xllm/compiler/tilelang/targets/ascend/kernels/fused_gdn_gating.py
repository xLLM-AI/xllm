#!/usr/bin/env python3

import argparse
from pathlib import Path

import tilelang
import tilelang.language as T

from scripts.logger import logger

from compiler.tilelang.targets.ascend.kernels.utils import (
    DEFAULT_ASCEND_PASS_CONFIGS,
    detect_vec_core_num,
    mte2_wait_mte3,
    mte3_notify_mte2,
)
from compiler.tilelang.common.spec import (
    DispatchField,
    TilelangKernel,
    register_kernel,
)

DEFAULT_NUM_HEADS = 32
DEFAULT_DTYPE = "bf16"
DEFAULT_MAX_BATCH = 262144
DEFAULT_MAX_HEADS = 128
REF_CHECK_NUM_BATCHES = 16
REF_CHECK_NUM_HEADS = (1, 16, 32, 48, 64, 128)
VEC_NUM = 2
VECTOR_BYTES_PER_ITER = 256
SUPPORTED_NUM_HEADS = (4, 6, 8, 12, 16, 24, 32, 48, 64, 128)
MAX_VEC_CORE_NUM = detect_vec_core_num()
BATCH_SIZE_SPECIALIZATIONS = tuple(range(2, 49, 2))
# Dedicated MTE3->MTE2 event for chunk-to-chunk UB reuse. The numeric
# id has no semantic meaning; it only needs to be paired and not collide
# with other MTE3_MTE2 syncs in this kernel.
CHUNK_OUTPUT_STORED_EVENT = 7


def select_launch_block_num(*, num_batches: int, vec_core_num: int) -> int:
    """Pick launch block_num by current batch size."""
    if num_batches <= 0:
        raise ValueError(f"num_batches({num_batches}) must be > 0")
    if vec_core_num <= 0:
        raise ValueError(f"vec_core_num({vec_core_num}) must be > 0")
    return min(num_batches, vec_core_num)


def _dtype_size_in_bytes(dtype: str) -> int:
    sizes = {
        "float16": 2,
        "bfloat16": 2,
        "float32": 4,
    }
    if dtype not in sizes:
        raise ValueError(f"Unsupported dtype for vector alignment: {dtype}")
    return sizes[dtype]


def _align_count_to_vector_bytes(count: int, dtype: str) -> int:
    elem_bytes = _dtype_size_in_bytes(dtype)
    elems_per_iter = VECTOR_BYTES_PER_ITER // elem_bytes
    return ((count + elems_per_iter - 1) // elems_per_iter) * elems_per_iter


DATACOPY_MIN_BYTES = 32


def _can_use_bulk_dma(num_heads: int, rows_per_iter: int) -> bool:
    """Check if multi-row DMA is viable (GM and UB strides match)."""
    f32_vec_elems = VECTOR_BYTES_PER_ITER // 4
    if (rows_per_iter * num_heads) % f32_vec_elems != 0:
        return False
    bf16_block_elems = DATACOPY_MIN_BYTES // 2
    if num_heads % bf16_block_elems != 0:
        return False
    return True


UB_BUDGET_BYTES = 64 * 1024


def _compute_rows_per_iter(num_heads: int, ub_dim: int | None = None) -> int:
    """Compute how many rows to process per loop iteration based on UB budget."""
    if ub_dim is None:
        ub_dim = _align_count_to_vector_bytes(num_heads, "float32")
    cmp_mask_bytes = (ub_dim + 7) // 8
    shared_bytes = 1 * ub_dim * 4
    per_row_bytes = (
        2 * ub_dim * 4
        + 2 * ub_dim * 2
        + 5 * ub_dim * 4
        + 1 * ub_dim * 1
        + 1 * cmp_mask_bytes
    )
    max_rows = (UB_BUDGET_BYTES - shared_bytes) // per_row_bytes
    if max_rows >= 32:
        return 32
    if max_rows >= 16:
        return 16
    if max_rows >= 8:
        return 8
    if max_rows >= 4:
        return 4
    if max_rows >= 2:
        return 2
    return 1


def build_fused_gdn_gating_kernel(
    *,
    batch_size: int,
    compile_max_batch: int,
    num_heads: int,
):
    if num_heads not in SUPPORTED_NUM_HEADS:
        raise ValueError(
            "fused_gdn_gating only supports num_heads in "
            f"{SUPPORTED_NUM_HEADS}, got {num_heads}"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size({batch_size}) must be > 0")
    if compile_max_batch <= 0:
        raise ValueError(
            f"compile_max_batch({compile_max_batch}) must be > 0"
        )
    if batch_size > compile_max_batch:
        raise ValueError(
            f"batch_size({batch_size}) must be <= compile_max_batch({compile_max_batch})"
        )

    vec_core_num = MAX_VEC_CORE_NUM
    block_num = select_launch_block_num(
        num_batches=batch_size, vec_core_num=vec_core_num
    )
    cubecore_block_num = block_num
    task_num = block_num * VEC_NUM
    acc_dtype = "float32"
    input_dtype = "bfloat16"
    mask_dtype = "uint8"
    rows_conservative = _compute_rows_per_iter(num_heads)
    use_bulk_dma = _can_use_bulk_dma(num_heads, rows_conservative)
    if use_bulk_dma:
        ub_tensor_dim = num_heads
        rows_per_iter = _compute_rows_per_iter(num_heads, ub_dim=num_heads)
    else:
        ub_tensor_dim = _align_count_to_vector_bytes(num_heads, acc_dtype)
        rows_per_iter = rows_conservative

    if batch_size <= rows_per_iter:
        rows_per_iter = max(1, batch_size // block_num)
        use_bulk_dma = _can_use_bulk_dma(num_heads, rows_per_iter)
        if use_bulk_dma:
            ub_tensor_dim = num_heads
        else:
            ub_tensor_dim = _align_count_to_vector_bytes(num_heads, acc_dtype)
    compare_select_mask_bytes = (ub_tensor_dim + 7) // 8
    multi_count = rows_per_iter * ub_tensor_dim
    multi_cmp_mask = rows_per_iter * compare_select_mask_bytes

    @T.prim_func
    def fused_gdn_gating_kernel(
        A_log: T.Tensor((num_heads,), acc_dtype),
        a: T.Tensor((compile_max_batch, num_heads), input_dtype),
        b: T.Tensor((compile_max_batch, num_heads), input_dtype),
        dt_bias: T.Tensor((num_heads,), acc_dtype),
        g_out: T.Tensor((compile_max_batch, num_heads), acc_dtype),
        beta_out: T.Tensor((compile_max_batch, num_heads), input_dtype),
        num_batches: T.int32,
        softplus_beta: T.float32,
        softplus_threshold: T.float32,
    ):
        with T.Kernel(cubecore_block_num, is_npu=True) as (cid, vid):
            task_id = cid * VEC_NUM + vid
            total_chunks = (num_batches + rows_per_iter - 1) // rows_per_iter
            chunks_per_task = (total_chunks + task_num - 1) // task_num
            chunk_start = task_id * chunks_per_task
            chunks_left = T.if_then_else(
                total_chunks > chunk_start,
                total_chunks - chunk_start,
                0,
            )
            num_chunks = T.if_then_else(
                chunks_left < chunks_per_task,
                chunks_left,
                chunks_per_task,
            )

            with T.Scope("V"):
                A_log_ub = T.alloc_shared((1, ub_tensor_dim), acc_dtype)
                neg_exp_A_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                dt_bias_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                a_half_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), input_dtype
                )
                b_half_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), input_dtype
                )
                x_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                beta_x_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                softplus_abs_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                softplus_tmp_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                beta_fp32_ub = T.alloc_shared(
                    (rows_per_iter, ub_tensor_dim), acc_dtype
                )
                softplus_cmp_mask_ub = T.alloc_ub(
                    (rows_per_iter, compare_select_mask_bytes), mask_dtype
                )

                # Preamble: load constants and replicate to R rows.
                T.copy(A_log[0], A_log_ub[0, :num_heads])
                T.tile.exp(A_log_ub, A_log_ub)
                T.tile.mul(A_log_ub, A_log_ub, -1.0)

                dt_bias_base_ub = T.alloc_shared(
                    (1, ub_tensor_dim), acc_dtype
                )
                T.copy(dt_bias[0], dt_bias_base_ub[0, :num_heads])
                for r in T.serial(rows_per_iter):
                    T.copy(
                        A_log_ub[0, :ub_tensor_dim],
                        neg_exp_A_ub[r, :ub_tensor_dim],
                    )
                    T.copy(
                        dt_bias_base_ub[0, :ub_tensor_dim],
                        dt_bias_ub[r, :ub_tensor_dim],
                    )

                if use_bulk_dma:
                    for chunk_idx in T.serial(num_chunks):
                        with T.If(chunk_idx > 0):
                            with T.Then():
                                mte2_wait_mte3(CHUNK_OUTPUT_STORED_EVENT)
                        base_row = (
                            (chunk_start + chunk_idx) * rows_per_iter
                        )
                        remaining = T.if_then_else(
                            num_batches > base_row,
                            num_batches - base_row,
                            0,
                        )
                        is_full_chunk = remaining >= rows_per_iter

                        with T.If(is_full_chunk):
                            with T.Then():
                                T.copy(a[base_row, 0], a_half_ub)
                                T.copy(b[base_row, 0], b_half_ub)
                            with T.Else():
                                for r in T.serial(remaining):
                                    T.copy(
                                        a[base_row + r, 0],
                                        a_half_ub[r, :num_heads],
                                    )
                                    T.copy(
                                        b[base_row + r, 0],
                                        b_half_ub[r, :num_heads],
                                    )

                        T.tile.cast(
                            x_ub, a_half_ub, "CAST_NONE", multi_count
                        )
                        T.tile.axpy(x_ub, dt_bias_ub, 1.0)
                        T.tile.mul(beta_x_ub, x_ub, softplus_beta)
                        T.tile.abs(softplus_abs_ub, beta_x_ub)
                        T.tile.mul(
                            softplus_tmp_ub, softplus_abs_ub, -1.0
                        )
                        T.tile.exp(beta_fp32_ub, softplus_tmp_ub)
                        T.tile.add(beta_fp32_ub, beta_fp32_ub, 1.0)
                        T.tile.ln(softplus_tmp_ub, beta_fp32_ub)
                        T.tile.compare(
                            softplus_cmp_mask_ub,
                            beta_x_ub,
                            softplus_threshold,
                            "GT",
                        )
                        T.tile.add(
                            beta_x_ub, beta_x_ub, softplus_abs_ub
                        )
                        T.tile.mul(
                            beta_x_ub, beta_x_ub, 0.5 / softplus_beta
                        )
                        T.tile.axpy(
                            beta_x_ub,
                            softplus_tmp_ub,
                            1.0 / softplus_beta,
                        )
                        T.tile.select(
                            beta_x_ub,
                            softplus_cmp_mask_ub,
                            x_ub,
                            beta_x_ub,
                            "VSEL_TENSOR_TENSOR_MODE",
                        )
                        T.tile.cast(
                            x_ub, b_half_ub, "CAST_NONE", multi_count
                        )
                        T.tile.sigmoid(
                            beta_fp32_ub, x_ub
                        )
                        T.tile.mul(x_ub, neg_exp_A_ub, beta_x_ub)
                        T.tile.cast(
                            b_half_ub,
                            beta_fp32_ub,
                            "CAST_RINT",
                            multi_count,
                        )

                        with T.If(is_full_chunk):
                            with T.Then():
                                T.copy(x_ub, g_out[base_row, 0])
                                T.copy(b_half_ub, beta_out[base_row, 0])
                            with T.Else():
                                for r in T.serial(remaining):
                                    T.copy(
                                        x_ub[r, :num_heads],
                                        g_out[base_row + r, :],
                                    )
                                    T.copy(
                                        b_half_ub[r, :num_heads],
                                        beta_out[base_row + r, :],
                                    )
                        with T.If(chunk_idx < num_chunks - 1):
                            with T.Then():
                                mte3_notify_mte2(CHUNK_OUTPUT_STORED_EVENT)

                else:
                    for chunk_idx in T.serial(num_chunks):
                        with T.If(chunk_idx > 0):
                            with T.Then():
                                mte2_wait_mte3(CHUNK_OUTPUT_STORED_EVENT)
                        base_row = (
                            (chunk_start + chunk_idx) * rows_per_iter
                        )
                        remaining = T.if_then_else(
                            num_batches > base_row,
                            num_batches - base_row,
                            0,
                        )
                        valid_rows = T.if_then_else(
                            remaining >= rows_per_iter,
                            rows_per_iter,
                            remaining,
                        )

                        for r in T.serial(valid_rows):
                            T.copy(
                                a[base_row + r, 0],
                                a_half_ub[r, :num_heads],
                            )
                            T.copy(
                                b[base_row + r, 0],
                                b_half_ub[r, :num_heads],
                            )

                        T.tile.cast(
                            x_ub, a_half_ub, "CAST_NONE", multi_count
                        )
                        T.tile.axpy(x_ub, dt_bias_ub, 1.0)
                        T.tile.mul(beta_x_ub, x_ub, softplus_beta)
                        T.tile.abs(softplus_abs_ub, beta_x_ub)
                        T.tile.mul(
                            softplus_tmp_ub, softplus_abs_ub, -1.0
                        )
                        T.tile.exp(beta_fp32_ub, softplus_tmp_ub)
                        T.tile.add(beta_fp32_ub, beta_fp32_ub, 1.0)
                        T.tile.ln(softplus_tmp_ub, beta_fp32_ub)
                        T.tile.compare(
                            softplus_cmp_mask_ub,
                            beta_x_ub,
                            softplus_threshold,
                            "GT",
                        )
                        T.tile.add(
                            beta_x_ub, beta_x_ub, softplus_abs_ub
                        )
                        T.tile.mul(
                            beta_x_ub, beta_x_ub, 0.5 / softplus_beta
                        )
                        T.tile.axpy(
                            beta_x_ub,
                            softplus_tmp_ub,
                            1.0 / softplus_beta,
                        )
                        T.tile.select(
                            beta_x_ub,
                            softplus_cmp_mask_ub,
                            x_ub,
                            beta_x_ub,
                            "VSEL_TENSOR_TENSOR_MODE",
                        )
                        T.tile.cast(
                            x_ub, b_half_ub, "CAST_NONE", multi_count
                        )
                        T.tile.sigmoid(
                            beta_fp32_ub, x_ub
                        )
                        T.tile.mul(x_ub, neg_exp_A_ub, beta_x_ub)
                        T.tile.cast(
                            b_half_ub,
                            beta_fp32_ub,
                            "CAST_RINT",
                            multi_count,
                        )

                        for r in T.serial(valid_rows):
                            T.copy(
                                x_ub[r, :num_heads],
                                g_out[base_row + r, :],
                            )
                            T.copy(
                                b_half_ub[r, :num_heads],
                                beta_out[base_row + r, :],
                            )
                        with T.If(chunk_idx < num_chunks - 1):
                            with T.Then():
                                mte3_notify_mte2(CHUNK_OUTPUT_STORED_EVENT)

    return fused_gdn_gating_kernel


@tilelang.jit(pass_configs=DEFAULT_ASCEND_PASS_CONFIGS)
def fused_gdn_gating_kernel_jit(
    num_batches: int,
    compile_max_batch: int,
    num_heads: int,
):
    return build_fused_gdn_gating_kernel(
        batch_size=num_batches,
        compile_max_batch=compile_max_batch,
        num_heads=num_heads,
    )


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
            raise ValueError(
                f"fused_gdn_gating only supports dtype={DEFAULT_DTYPE}, got {dtype}"
            )
        if num_heads not in SUPPORTED_NUM_HEADS:
            raise ValueError(
                "fused_gdn_gating only supports num_heads in "
                f"{SUPPORTED_NUM_HEADS}, got {num_heads}"
            )
        if batch_size not in BATCH_SIZE_SPECIALIZATIONS:
            raise ValueError(
                "fused_gdn_gating only supports batch_size in "
                f"{BATCH_SIZE_SPECIALIZATIONS}, got {batch_size}"
            )
        tilelang.disable_cache()
        tilelang_kernel = build_fused_gdn_gating_kernel(
            batch_size=batch_size,
            compile_max_batch=DEFAULT_MAX_BATCH,
            num_heads=num_heads,
        )
        with tilelang.tvm.transform.PassContext(
            opt_level=3, config=DEFAULT_ASCEND_PASS_CONFIGS
        ):
            kernel = tilelang.engine.lower(tilelang_kernel)
        return kernel.kernel_source


def _torch_fused_gdn_gating(
    A_log: "torch.Tensor",
    a: "torch.Tensor",
    b: "torch.Tensor",
    dt_bias: "torch.Tensor",
    softplus_beta: float,
    softplus_threshold: float,
) -> tuple["torch.Tensor", "torch.Tensor"]:
    import torch
    import torch.nn.functional as F

    softplus_out = F.softplus(
        a.to(torch.float32) + dt_bias,
        beta=softplus_beta,
        threshold=softplus_threshold,
    )
    g_ref = -A_log.exp() * softplus_out
    beta_ref = torch.sigmoid(b.to(torch.float32)).to(torch.bfloat16)
    return g_ref, beta_ref


def _run_ref_check(
    *,
    num_batches: int,
    num_heads: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
) -> None:
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        logger.warning("Skip fused_gdn_gating reference check: NPU is not available")
        return

    if num_batches <= 0:
        raise ValueError(f"num_batches({num_batches}) must be > 0")
    if num_batches > compile_max_batch:
        raise ValueError(
            f"num_batches({num_batches}) must be <= compile_max_batch({compile_max_batch})"
        )

    torch.manual_seed(42)
    device = torch.device("npu")

    A_log = torch.randn((num_heads,), device=device, dtype=torch.float32)
    a = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    b = torch.randn((num_batches, num_heads), device=device, dtype=torch.bfloat16)
    dt_bias = torch.randn((num_heads,), device=device, dtype=torch.float32)
    g_out = torch.empty((num_batches, num_heads), device=device, dtype=torch.float32)
    beta_out = torch.empty(
        (num_batches, num_heads), device=device, dtype=torch.bfloat16
    )

    kernel = fused_gdn_gating_kernel_jit(
        num_batches=num_batches,
        compile_max_batch=num_batches,
        num_heads=num_heads,
    )
    kernel(
        A_log,
        a,
        b,
        dt_bias,
        g_out,
        beta_out,
        num_batches,
        softplus_beta,
        softplus_threshold,
    )
    torch.npu.synchronize()

    g_ref, beta_ref = _torch_fused_gdn_gating(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
    )
    torch.testing.assert_close(g_out, g_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(
        beta_out.to(torch.float32),
        beta_ref.to(torch.float32),
        rtol=1e-2,
        atol=1e-2,
    )
    logger.info(f"fused_gdn_gating output matches torch reference for num_heads={num_heads}")


def _run_ref_suite(
    *,
    num_batches: int,
    compile_max_batch: int,
    softplus_beta: float,
    softplus_threshold: float,
    ref_num_heads_list: list[int],
) -> None:
    for num_heads in ref_num_heads_list:
        _run_ref_check(
            num_batches=num_batches,
            num_heads=num_heads,
            compile_max_batch=compile_max_batch,
            softplus_beta=softplus_beta,
            softplus_threshold=softplus_threshold,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate TileLang AscendC source for fused_gdn_gating AOT kernel."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=max(BATCH_SIZE_SPECIALIZATIONS),
        help=(
            "Batch-size specialization used for source generation. "
            f"Supported values: {BATCH_SIZE_SPECIALIZATIONS}"
        ),
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
