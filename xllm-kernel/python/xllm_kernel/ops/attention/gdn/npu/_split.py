# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Private Triton Ascend DSL; imported only by the NPU solution."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "mixed_qkv",
        "z",
        "b",
        "a",
        "mixed_qkvz",
        "mixed_ba",
        "num_heads_qk",
        "num_heads_v",
        "total_rows",
        "rows_per_vec",
        "qkvz_row_stride",
        "ba_row_stride",
        "qkv_row_stride",
        "z_row_stride",
        "ba_out_row_stride",
    ]
)
def _fused_qkvzba_split_reshape_cat_gqa_r1_kernel(
    mixed_qkv: tl.tensor,
    z: tl.tensor,
    b: tl.tensor,
    a: tl.tensor,
    mixed_qkvz: tl.tensor,
    mixed_ba: tl.tensor,
    num_heads_qk: tl.tensor,
    num_heads_v: tl.tensor,
    total_rows: tl.tensor,
    rows_per_vec: tl.tensor,
    qkvz_row_stride: tl.tensor,
    ba_row_stride: tl.tensor,
    qkv_row_stride: tl.tensor,
    z_row_stride: tl.tensor,
    ba_out_row_stride: tl.tensor,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
) -> None:
    vec_id = tl.program_id(0)
    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)
    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end
        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 2)
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])
            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = (
                row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            )
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])
            v_range = tl.arange(0, HEAD_V)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK * 2 + head_id * HEAD_V + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])
            z_range = tl.arange(0, HEAD_V)
            z_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V + z_range[None, :]
            z_dst = row_indices[:, None] * z_row_stride + head_id * HEAD_V + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])
            ba_head_offset = head_id * 2
            b_range = tl.arange(0, 1)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])
            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 1 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])
        row_offset += ROWS_PER_ITER


@triton.jit(
    do_not_specialize=[
        "mixed_qkv",
        "z",
        "b",
        "a",
        "mixed_qkvz",
        "mixed_ba",
        "num_heads_qk",
        "num_heads_v",
        "total_rows",
        "rows_per_vec",
        "qkvz_row_stride",
        "ba_row_stride",
        "qkv_row_stride",
        "z_row_stride",
        "ba_out_row_stride",
    ]
)
def _fused_qkvzba_split_reshape_cat_gqa_r2_kernel(
    mixed_qkv: tl.tensor,
    z: tl.tensor,
    b: tl.tensor,
    a: tl.tensor,
    mixed_qkvz: tl.tensor,
    mixed_ba: tl.tensor,
    num_heads_qk: tl.tensor,
    num_heads_v: tl.tensor,
    total_rows: tl.tensor,
    rows_per_vec: tl.tensor,
    qkvz_row_stride: tl.tensor,
    ba_row_stride: tl.tensor,
    qkv_row_stride: tl.tensor,
    z_row_stride: tl.tensor,
    ba_out_row_stride: tl.tensor,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
) -> None:
    vec_id = tl.program_id(0)
    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)
    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end
        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 4)
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])
            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = (
                row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            )
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])
            v_range = tl.arange(0, HEAD_V * 2)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * qkv_row_stride
                + num_heads_qk * HEAD_QK * 2
                + head_id * (HEAD_V * 2)
                + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])
            z_range = tl.arange(0, HEAD_V * 2)
            z_src = (
                row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 2 + z_range[None, :]
            )
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 2) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])
            ba_head_offset = head_id * 4
            b_range = tl.arange(0, 2)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 2 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])
            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 2 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])
        row_offset += ROWS_PER_ITER


@triton.jit(
    do_not_specialize=[
        "mixed_qkv",
        "z",
        "b",
        "a",
        "mixed_qkvz",
        "mixed_ba",
        "num_heads_qk",
        "num_heads_v",
        "total_rows",
        "rows_per_vec",
        "qkvz_row_stride",
        "ba_row_stride",
        "qkv_row_stride",
        "z_row_stride",
        "ba_out_row_stride",
    ]
)
def _fused_qkvzba_split_reshape_cat_gqa_r3_kernel(
    mixed_qkv: tl.tensor,
    z: tl.tensor,
    b: tl.tensor,
    a: tl.tensor,
    mixed_qkvz: tl.tensor,
    mixed_ba: tl.tensor,
    num_heads_qk: tl.tensor,
    num_heads_v: tl.tensor,
    total_rows: tl.tensor,
    rows_per_vec: tl.tensor,
    qkvz_row_stride: tl.tensor,
    ba_row_stride: tl.tensor,
    qkv_row_stride: tl.tensor,
    z_row_stride: tl.tensor,
    ba_out_row_stride: tl.tensor,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
) -> None:
    vec_id = tl.program_id(0)
    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)
    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end
        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 6)
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])
            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = (
                row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            )
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])
            v_range = tl.arange(0, HEAD_V * 3)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * qkv_row_stride
                + num_heads_qk * HEAD_QK * 2
                + head_id * (HEAD_V * 3)
                + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])
            z_range = tl.arange(0, HEAD_V * 3)
            z_src = (
                row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 3 + z_range[None, :]
            )
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 3) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])
            ba_head_offset = head_id * 6
            b_range = tl.arange(0, 3)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 3 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])
            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 3 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])
        row_offset += ROWS_PER_ITER


@triton.jit(
    do_not_specialize=[
        "mixed_qkv",
        "z",
        "b",
        "a",
        "mixed_qkvz",
        "mixed_ba",
        "num_heads_qk",
        "num_heads_v",
        "total_rows",
        "rows_per_vec",
        "qkvz_row_stride",
        "ba_row_stride",
        "qkv_row_stride",
        "z_row_stride",
        "ba_out_row_stride",
    ]
)
def _fused_qkvzba_split_reshape_cat_gqa_r4_kernel(
    mixed_qkv: tl.tensor,
    z: tl.tensor,
    b: tl.tensor,
    a: tl.tensor,
    mixed_qkvz: tl.tensor,
    mixed_ba: tl.tensor,
    num_heads_qk: tl.tensor,
    num_heads_v: tl.tensor,
    total_rows: tl.tensor,
    rows_per_vec: tl.tensor,
    qkvz_row_stride: tl.tensor,
    ba_row_stride: tl.tensor,
    qkv_row_stride: tl.tensor,
    z_row_stride: tl.tensor,
    ba_out_row_stride: tl.tensor,
    HEAD_QK: tl.constexpr,
    HEAD_V: tl.constexpr,
    ROWS_PER_ITER: tl.constexpr,
) -> None:
    vec_id = tl.program_id(0)
    row_start = vec_id * rows_per_vec
    row_end = tl.minimum(row_start + rows_per_vec, total_rows)
    row_offset = row_start
    iter_count = (row_end - row_start + ROWS_PER_ITER - 1) // ROWS_PER_ITER
    for _ in tl.range(iter_count):
        row_indices = tl.arange(0, ROWS_PER_ITER) + row_offset
        row_mask = row_indices < row_end
        for head_id in tl.range(num_heads_qk):
            src_head_offset = head_id * (HEAD_QK * 2 + HEAD_V * 8)
            q_range = tl.arange(0, HEAD_QK)
            q_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + q_range[None, :]
            q_dst = row_indices[:, None] * qkv_row_stride + head_id * HEAD_QK + q_range[None, :]
            q_data = tl.load(mixed_qkvz + q_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + q_dst, q_data, mask=row_mask[:, None])
            k_range = tl.arange(0, HEAD_QK)
            k_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK + k_range[None, :]
            k_dst = (
                row_indices[:, None] * qkv_row_stride + num_heads_qk * HEAD_QK + head_id * HEAD_QK + k_range[None, :]
            )
            k_data = tl.load(mixed_qkvz + k_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + k_dst, k_data, mask=row_mask[:, None])
            v_range = tl.arange(0, HEAD_V * 4)
            v_src = row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + v_range[None, :]
            v_dst = (
                row_indices[:, None] * qkv_row_stride
                + num_heads_qk * HEAD_QK * 2
                + head_id * (HEAD_V * 4)
                + v_range[None, :]
            )
            v_data = tl.load(mixed_qkvz + v_src, mask=row_mask[:, None])
            tl.store(mixed_qkv + v_dst, v_data, mask=row_mask[:, None])
            z_range = tl.arange(0, HEAD_V * 4)
            z_src = (
                row_indices[:, None] * qkvz_row_stride + src_head_offset + HEAD_QK * 2 + HEAD_V * 4 + z_range[None, :]
            )
            z_dst = row_indices[:, None] * z_row_stride + head_id * (HEAD_V * 4) + z_range[None, :]
            z_data = tl.load(mixed_qkvz + z_src, mask=row_mask[:, None])
            tl.store(z + z_dst, z_data, mask=row_mask[:, None])
            ba_head_offset = head_id * 8
            b_range = tl.arange(0, 4)
            b_src = row_indices[:, None] * ba_row_stride + ba_head_offset + b_range[None, :]
            b_dst = row_indices[:, None] * ba_out_row_stride + head_id * 4 + b_range[None, :]
            b_data = tl.load(mixed_ba + b_src, mask=row_mask[:, None])
            tl.store(b + b_dst, b_data, mask=row_mask[:, None])
            a_src = row_indices[:, None] * ba_row_stride + ba_head_offset + 4 + b_range[None, :]
            a_data = tl.load(mixed_ba + a_src, mask=row_mask[:, None])
            tl.store(a + b_dst, a_data, mask=row_mask[:, None])
        row_offset += ROWS_PER_ITER
