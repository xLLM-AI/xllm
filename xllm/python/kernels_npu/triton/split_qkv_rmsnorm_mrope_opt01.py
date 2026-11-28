# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.
"""Ordinary packed QKV fusion with a once-per-forward mRoPE table."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from xllm.python.kernels_npu.triton.utils import extract_slice, get_vectorcore_num, insert_slice


@triton.jit
def _prepare_kernel(CACHE, POS, AXIS, OUT, N, D: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    col = tl.arange(0, D)
    t = tl.load(POS + row, row < N, other=0).to(tl.int32)
    h = tl.load(POS + N + row, row < N, other=0).to(tl.int32)
    w = tl.load(POS + 2 * N + row, row < N, other=0).to(tl.int32)
    tc = tl.load(CACHE + t[:, None] * D + col[None, :], (row < N)[:, None], other=0)
    hc = tl.load(CACHE + h[:, None] * D + col[None, :], (row < N)[:, None], other=0)
    wc = tl.load(CACHE + w[:, None] * D + col[None, :], (row < N)[:, None], other=0)
    axis = tl.load(AXIS + col % (D // 2)).to(tl.float32)
    value = tl.where((axis == 1)[None, :], hc, tc)
    value = tl.where((axis == 2)[None, :], wc, value)
    tl.store(OUT + row[:, None] * D + col[None, :], value, (row < N)[:, None])


@torch.library.custom_op("xllm_python::opt01_prepare_mrope", mutates_args=())
def prepare_mrope(cache: torch.Tensor, positions: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Gather shared T/H/W cos/sin while retaining live positions in graphs."""
    n, d = positions.shape[1], cache.shape[1]
    out = torch.empty((n, d), device=cache.device, dtype=cache.dtype)
    _prepare_kernel[(triton.cdiv(n, 4),)](cache, positions, axis, out, n, d, 4)
    return out


@prepare_mrope.register_fake
def _prepare_mrope_fake(cache: torch.Tensor, positions: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    return cache.new_empty((positions.shape[1], cache.shape[1]))


@triton.jit
def _split_qkv_rmsnorm_mrope_kernel(
    input_gm_ptr,
    q_gm_ptr,
    k_gm_ptr,
    v_gm_ptr,
    q_weight_ptr,
    k_weight_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    num_vectorcore: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    qk_head_nums_per_iter_per_vec: tl.constexpr,
    q_head_num: tl.constexpr,
    kv_head_num: tl.constexpr,
    qk_head_num_sum: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    cos_sin_cache_gm_ptr,
):
    row_pid = tl.program_id(0)

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    batch_size_per_vec = tl.cdiv(batch_size, num_vectorcore)
    iter_num_per_vec = tl.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)
    v_iter_num_per_vec = tl.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)
    input_batch_offset = row_pid * batch_size_per_vec
    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, q_hidden_size + kv_hidden_size)
    nmask = nblk_idx < total_hidden_size

    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    output_q_nblk_idx = tl.arange(0, q_hidden_size)
    output_q_nmask = output_q_nblk_idx < q_hidden_size
    output_kv_nblk_idx = tl.arange(0, kv_hidden_size)
    output_kv_nmask = output_kv_nblk_idx < kv_hidden_size
    sin_cos_range = tl.arange(0, ROPE_DIM)

    for iter in tl.range(iter_num_per_vec):
        pos_offset = iter * batch_size_per_iter_per_vec
        mmask = (mblk_idx + pos_offset) < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = (mblk_idx + pos_offset)[:, None] * total_hidden_size + nblk_idx[None, :]
        values_tmp1 = tl.load(input_gm_ptr + idx, mask=mask).reshape(qk_head_nums_per_iter_per_vec, HEAD_DIM)

        table_offsets = (mblk_idx + pos_offset)[:, None] * ROPE_DIM + sin_cos_range[None, :]
        values_tmp3 = tl.load(cos_sin_cache_gm_ptr + table_offsets, mask=mmask[:, None], other=0)
        values_tmp3 = values_tmp3.reshape(batch_size_per_iter_per_vec, 1, ROPE_DIM)
        cos = extract_slice(
            values_tmp3,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        sin = extract_slice(
            values_tmp3,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, 1, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )

        normalized_values = values_tmp1.to(tl.float32)
        normalized_values = normalized_values * normalized_values
        normalized_values = tl.sum(normalized_values, axis=1) / HEAD_DIM
        normalized_values = 1 / tl.sqrt(normalized_values + eps).reshape(qk_head_nums_per_iter_per_vec, 1)
        normalized_values = values_tmp1 * normalized_values

        # Q: norm * weight + rope
        normalized_values_tmp = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        normalized_values_tmp = (normalized_values_tmp * q_weight_values).to(tl.bfloat16)

        values_tmp = tl.zeros((batch_size_per_iter_per_vec, q_head_num, ROPE_DIM), dtype=tl.bfloat16)
        x1 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp = insert_slice(
            values_tmp,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, q_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        q_output_idx = output_q_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * q_hidden_size
        q_mask = (mmask[:, None]) & (output_q_nmask[None, :])
        tl.store(
            q_gm_ptr + q_output_idx,
            values_tmp.reshape(batch_size_per_iter_per_vec, q_hidden_size),
            mask=q_mask,
        )

        # K: norm * weight + rope
        normalized_values_tmp1 = extract_slice(
            normalized_values.reshape(batch_size_per_iter_per_vec, qk_head_num_sum, HEAD_DIM),
            offsets=(0, q_head_num, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HEAD_DIM),
            strides=(1, 1, 1),
        )
        normalized_values_tmp1 = (normalized_values_tmp1 * k_weight_values).to(tl.bfloat16)

        values_tmp2 = tl.zeros((batch_size_per_iter_per_vec, kv_head_num, ROPE_DIM), dtype=tl.bfloat16)
        x1 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        x2 = extract_slice(
            normalized_values_tmp1,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x1 * cos - x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        values_tmp2 = insert_slice(
            values_tmp2,
            x2 * cos + x1 * sin,
            offsets=(0, 0, HALF_ROPE_DIM),
            sizes=(batch_size_per_iter_per_vec, kv_head_num, HALF_ROPE_DIM),
            strides=(1, 1, 1),
        )
        kv_output_idx = output_kv_nblk_idx[None, :] + (mblk_idx + pos_offset)[:, None] * kv_hidden_size
        kv_mask = (mmask[:, None]) & (output_kv_nmask[None, :])
        tl.store(
            k_gm_ptr + kv_output_idx,
            values_tmp2.reshape(batch_size_per_iter_per_vec, kv_hidden_size),
            mask=kv_mask,
        )

    # V: just copy
    mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
    nmask = nblk_idx < total_hidden_size
    out_nblk_idx = tl.arange(0, kv_hidden_size)
    out_nmask = out_nblk_idx < kv_hidden_size

    for _ in tl.range(v_iter_num_per_vec):
        mmask = mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nmask[None, :])
        idx = mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        values = tl.load(input_gm_ptr + idx, mask=mask)
        out_idx = mblk_idx[:, None] * kv_hidden_size + out_nblk_idx[None, :]
        out_mask = (mmask[:, None]) & (out_nmask[None, :])
        tl.store(v_gm_ptr + out_idx, values, mask=out_mask)
        mblk_idx += v_batch_size_per_iter_per_vec


@torch.library.custom_op("xllm_python::opt01_split_qkv_norm_mrope", mutates_args=())
def split_qkv_norm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return contiguous Q/K/V without exposing aliases of the packed input."""
    n = qkv.shape[0]
    q = qkv.new_empty((n, num_q_heads * head_dim))
    k = qkv.new_empty((n, num_kv_heads * head_dim))
    v = torch.empty_like(k)
    cores = get_vectorcore_num()
    q_size, k_size = num_q_heads * head_dim, num_kv_heads * head_dim
    factor = 5 * q_size + 3 * k_size + 2 * head_dim + num_q_heads * head_dim // 2
    tokens_per_iter = max(1, (87040 // qkv.element_size()) // factor)
    v_tokens_per_iter = 87040 // qkv.element_size() // (k_size + 1)
    heads_sum = num_q_heads + num_kv_heads
    _split_qkv_rmsnorm_mrope_kernel[(cores, 1, 1)](
        qkv,
        q,
        k,
        v,
        q_weight,
        k_weight,
        n,
        q_size,
        k_size,
        q_size + 2 * k_size,
        eps,
        head_dim,
        head_dim,
        head_dim // 2,
        cores,
        tokens_per_iter,
        tokens_per_iter * heads_sum,
        num_q_heads,
        num_kv_heads,
        heads_sum,
        v_tokens_per_iter,
        cos_sin,
        enable_fp_fusion=False,
    )
    return q, k, v


@split_qkv_norm_mrope.register_fake
def _split_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = qkv.shape[0]
    return (
        qkv.new_empty((n, num_q_heads * head_dim)),
        qkv.new_empty((n, num_kv_heads * head_dim)),
        qkv.new_empty((n, num_kv_heads * head_dim)),
    )


def build_axis_pattern(sections: list[int], interleaved: bool, device: torch.device) -> torch.Tensor:
    """Build the persistent table at initialization, without runtime gathers."""
    if len(sections) != 3 or sum(sections) != 64 or min(sections) < 0:
        raise ValueError("OPT-01 requires three nonnegative sections summing to 64")
    if interleaved:
        axis = [0] * 64
        for dim in (1, 2):
            for index in range(dim, sections[dim] * 3, 3):
                axis[index] = dim
    else:
        axis = [dim for dim, size in enumerate(sections) for _ in range(size)]
    return torch.tensor(axis, dtype=torch.int32, device=device)
