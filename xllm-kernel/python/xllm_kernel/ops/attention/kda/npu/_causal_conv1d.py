# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""KDA V2 convolution DSL, migrated from torch_npu_ops Triton Ascend."""

import triton
import triton.language as tl


@triton.jit(
    do_not_specialize=[
        "x_ptr",
        "w_ptr",
        "bias_ptr",
        "conv_state_ptr",
        "conv_state_indices_ptr",
        "num_accepted_tokens_ptr",
        "query_start_loc_ptr",
        "block_idx_last_scheduled_token_ptr",
        "initial_state_idx_ptr",
        "o_ptr",
        "batch",
        "dim",
        "seqlen",
        "state_len",
        "num_cache_lines",
        "stride_x_token",
        "stride_w_width",
        "stride_conv_state_seq",
        "stride_conv_state_tok",
        "stride_state_indices",
        "stride_o_token",
        "kernel_width",
        "is_spec_decoding",
        "pad_slot_id",
    ]
)
def _causal_conv1d_update_kernel_npu_tiled_v2(
    x_ptr: tl.tensor,
    w_ptr: tl.tensor,
    bias_ptr: tl.tensor,
    conv_state_ptr: tl.tensor,
    conv_state_indices_ptr: tl.tensor,
    num_accepted_tokens_ptr: tl.tensor,
    query_start_loc_ptr: tl.tensor,
    block_idx_last_scheduled_token_ptr: tl.tensor,
    initial_state_idx_ptr: tl.tensor,
    o_ptr: tl.tensor,
    batch: tl.int32,
    dim: tl.tensor,
    seqlen: tl.tensor,
    state_len: tl.tensor,
    num_cache_lines: tl.tensor,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.tensor,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.tensor,
    stride_conv_state_seq: tl.tensor,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.tensor,
    stride_state_indices: tl.tensor,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.tensor,
    kernel_width: tl.tensor,
    is_spec_decoding: tl.tensor,
    pad_slot_id: tl.tensor,
    BLOCK_N: tl.constexpr,
    B_TILE: tl.constexpr,
    T_CHUNK: tl.constexpr,
) -> None:
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    idx_feats = pid_c * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_w = idx_feats < dim

    kw = kernel_width.to(tl.int32)
    spec_enabled = is_spec_decoding.to(tl.int32) != 0

    w_base = w_ptr + idx_feats * stride_w_dim
    w_col0 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col1 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col2 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col3 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col4 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col5 = tl.zeros((BLOCK_N,), dtype=tl.float32)
    w_col0 = tl.load(w_base + 0 * stride_w_width, mask=mask_w & (kw > 0), other=0.0).to(tl.float32)
    w_col1 = tl.load(w_base + 1 * stride_w_width, mask=mask_w & (kw > 1), other=0.0).to(tl.float32)
    w_col2 = tl.load(w_base + 2 * stride_w_width, mask=mask_w & (kw > 2), other=0.0).to(tl.float32)
    w_col3 = tl.load(w_base + 3 * stride_w_width, mask=mask_w & (kw > 3), other=0.0).to(tl.float32)
    w_col4 = tl.load(w_base + 4 * stride_w_width, mask=mask_w & (kw > 4), other=0.0).to(tl.float32)
    w_col5 = tl.load(w_base + 5 * stride_w_width, mask=mask_w & (kw > 5), other=0.0).to(tl.float32)

    acc_bias = tl.load(bias_ptr + idx_feats, mask=mask_w, other=0.0).to(tl.float32)

    tok_vec = tl.arange(0, T_CHUNK)

    for bi in tl.static_range(0, B_TILE):
        b = pid_b * B_TILE + bi
        lane_active = b < batch

        conv_state_init = tl.load(initial_state_idx_ptr + b, mask=lane_active, other=0).to(tl.int32)
        current_last_index = tl.load(block_idx_last_scheduled_token_ptr + b, mask=lane_active, other=0).to(tl.int32)

        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + conv_state_init,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        lane_active = lane_active & (conv_states_input_coord != pad_slot_id)

        qs = tl.load(query_start_loc_ptr + b, mask=lane_active, other=0).to(tl.int64)
        qe = tl.load(query_start_loc_ptr + (b + 1), mask=lane_active, other=0).to(tl.int64)
        seqlen_run = (qe - qs).to(tl.int32)
        # Ordinary varlen keeps the full history, independent of max_query_len.
        state_len_run = (kw - 1 + tl.where(spec_enabled, seqlen_run - 1, 0)).to(tl.int32)
        x_offset = (qs * stride_x_token).to(tl.int64)
        o_offset = (qs * stride_o_token).to(tl.int64)

        lane_active = lane_active & (seqlen_run > 0)

        accepted_tokens = tl.load(num_accepted_tokens_ptr + b, mask=lane_active, other=1).to(tl.int64)
        conv_state_token_offset = tl.where(spec_enabled, accepted_tokens - 1, 0).to(tl.int64)
        shift = tl.where(spec_enabled, 1, seqlen_run).to(tl.int32)

        conv_states_base = (
            conv_state_ptr + conv_states_input_coord * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        )
        prior_tokens = conv_states_base + conv_state_token_offset * stride_conv_state_tok

        col0 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col1 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col2 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col3 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col4 = tl.zeros((BLOCK_N,), dtype=tl.float16)
        col0 = tl.load(
            prior_tokens + 0 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 1),
            other=0.0,
        ).to(tl.float16)
        col1 = tl.load(
            prior_tokens + 1 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 2),
            other=0.0,
        ).to(tl.float16)
        col2 = tl.load(
            prior_tokens + 2 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 3),
            other=0.0,
        ).to(tl.float16)
        col3 = tl.load(
            prior_tokens + 3 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 4),
            other=0.0,
        ).to(tl.float16)
        col4 = tl.load(
            prior_tokens + 4 * stride_conv_state_tok,
            mask=lane_active & mask_w & (kw > 5),
            other=0.0,
        ).to(tl.float16)

        conv_states_offset = tl.load(
            conv_state_indices_ptr + b * stride_state_indices + current_last_index,
            mask=lane_active,
            other=0,
        ).to(tl.int64)

        use_shift = seqlen_run < state_len_run
        use_tail = seqlen_run >= state_len_run
        keep_shift = tl.where(use_shift, (state_len_run - seqlen_run), 0).to(tl.int32)
        tail_start = tl.where(use_tail, (seqlen_run - state_len_run), 0).to(tl.int32)

        state_src_base = (
            conv_state_ptr
            + conv_states_input_coord * stride_conv_state_seq
            + conv_state_token_offset * stride_conv_state_tok
            + idx_feats * stride_conv_state_dim
        )
        state_dst_base = conv_state_ptr + conv_states_offset * stride_conv_state_seq + idx_feats * stride_conv_state_dim
        x_base = x_ptr + x_offset + idx_feats * stride_x_dim

        for t0 in tl.range(0, state_len_run, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)
            src_tok = (dst_tok + shift).to(tl.int32)
            m_tok = use_shift & (dst_tok < keep_shift) & (src_tok < state_len_run) & (dst_tok < state_len_run)
            m = (
                (lane_active & m_tok)[:, None]
                & mask_w[None, :]
                & (conv_states_input_coord < num_cache_lines)
                & (conv_states_offset < num_cache_lines)
            )
            src_ptrs = state_src_base[None, :] + src_tok[:, None] * stride_conv_state_tok
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            vals = tl.load(src_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, vals, mask=m)

        for t0 in tl.range(0, seqlen_run, T_CHUNK):
            x_tok = (t0 + tok_vec).to(tl.int32)
            dst_tok = (keep_shift + x_tok).to(tl.int32)
            m_tok = use_shift & (x_tok < seqlen_run) & (dst_tok < state_len_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        for t0 in tl.range(0, state_len_run, T_CHUNK):
            dst_tok = (t0 + tok_vec).to(tl.int32)
            x_tok = (tail_start + dst_tok).to(tl.int32)
            m_tok = use_tail & (dst_tok < state_len_run) & (x_tok < seqlen_run)
            m = (lane_active & m_tok)[:, None] & mask_w[None, :] & (conv_states_offset < num_cache_lines)
            x_ptrs = x_base[None, :] + x_tok[:, None] * stride_x_token
            dst_ptrs = state_dst_base[None, :] + dst_tok[:, None] * stride_conv_state_tok
            x_vals = tl.load(x_ptrs, mask=m, other=0.0)
            tl.store(dst_ptrs, x_vals, mask=m)

        x_base_1d = x_base
        o_base_1d = o_ptr + o_offset + idx_feats * stride_o_dim

        for idx_token in tl.range(seqlen_run):
            acc = acc_bias
            x_ptrs_1d = x_base_1d + idx_token * stride_x_token
            current_x = tl.load(x_ptrs_1d, mask=lane_active & mask_w, other=0.0).to(tl.float16)

            acc += tl.where(kw > 1, col0.to(tl.float32) * w_col0, 0.0)
            acc += tl.where(kw > 2, col1.to(tl.float32) * w_col1, 0.0)
            acc += tl.where(kw > 3, col2.to(tl.float32) * w_col2, 0.0)
            acc += tl.where(kw > 4, col3.to(tl.float32) * w_col3, 0.0)
            acc += tl.where(kw > 5, col4.to(tl.float32) * w_col4, 0.0)

            current_w = (
                tl.where(kw == 1, w_col0, 0.0)
                + tl.where(kw == 2, w_col1, 0.0)
                + tl.where(kw == 3, w_col2, 0.0)
                + tl.where(kw == 4, w_col3, 0.0)
                + tl.where(kw == 5, w_col4, 0.0)
                + tl.where(kw == 6, w_col5, 0.0)
            )
            acc += current_x.to(tl.float32) * current_w

            old0 = col0
            old1 = col1
            old2 = col2
            old3 = col3
            old4 = col4
            col0 = tl.where(kw > 1, tl.where(kw == 2, current_x, old1), col0)
            col1 = tl.where(kw > 2, tl.where(kw == 3, current_x, old2), col1)
            col2 = tl.where(kw > 3, tl.where(kw == 4, current_x, old3), col2)
            col3 = tl.where(kw > 4, tl.where(kw == 5, current_x, old4), col3)
            col4 = tl.where(kw > 5, current_x, col4)

            o_ptrs = o_base_1d + idx_token * stride_o_token
            tl.store(o_ptrs, acc, mask=lane_active & mask_w)
