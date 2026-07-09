# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This kernel is adapted from vLLM's FLA Triton ops:
# https://github.com/vllm-project/vllm/tree/v0.18.0/vllm/model_executor/layers/fla/ops
# Upstream license: Apache License, Version 2.0.
# Modified for xLLM MLU TMO integration.

import triton
import triton.language as tl


@triton.jit
def tmo_causal_conv1d_update_decode_kernel(
    # Pointers to tensors
    x_ptr,  # (batch, dim, seqlen) or (dim, cu_seqlen)
    weight_ptr,  # (dim, width)
    bias_ptr,  # (dim,) or nullptr
    conv_state_ptr,  # (num_cache_lines, dim, state_len)
    conv_state_indices_ptr,  # (batch,) int32
    num_accepted_tokens_ptr,  # (batch,) int32 or nullptr
    query_start_loc_ptr,  # (batch+1,) int32 or nullptr
    block_idx_last_scheduled_token_ptr,  # (batch,) int32 or nullptr
    initial_state_idx_ptr,  # (batch,) int32 or nullptr
    out_ptr,  # (batch, dim, seqlen) or (dim, cu_seqlen)
    # Dimensions (runtime)
    batch,
    num_cache_lines,
    # Dimensions (constexpr for AOT specialization)
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    # Strides (runtime)
    stride_x_seq,
    stride_x_dim,
    stride_x_token,
    stride_w_dim,
    stride_w_width,
    stride_istate_seq,
    stride_istate_dim,
    stride_istate_tok,
    stride_state_indices,
    stride_o_seq,
    stride_o_dim,
    stride_o_token,
    # Others (runtime)
    pad_slot_id,
    # Meta-parameters (constexpr for AOT specialization)
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BD: tl.constexpr,
    BW: tl.constexpr,
):
    # ruff: noqa: E501
    # Grid layout: program_id(0) = feature block, program_id(1) = sequence
    # This follows the torch_mlu_ops convention for better MLU parallelism.
    i_d = tl.program_id(0) * BD + tl.arange(0, BD)  # [BD] feature indices
    i_n = tl.program_id(1)  # sequence index

    if i_n >= batch:
        return

    # APC: load initial_state_idx and block_idx_last_scheduled_token
    if IS_APC_ENABLED:
        conv_state_init = tl.load(initial_state_idx_ptr + i_n)
        current_last_index = tl.load(block_idx_last_scheduled_token_ptr + i_n)
    else:
        conv_state_init = 0
        current_last_index = 0

    # Load conv_state_index for this sequence
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + i_n * stride_state_indices + conv_state_init
    ).to(tl.int64)

    # Skip padded entries
    if USE_PAD_SLOT:
        if conv_states_input_coord == pad_slot_id:
            return

    # Varlen: compute sequence start/end from query_start_loc
    if IS_VARLEN:
        query_start_index = tl.load(query_start_loc_ptr + i_n).to(tl.int64)
        query_end_index = tl.load(query_start_loc_ptr + (i_n + 1)).to(tl.int64)
        # revise state_len and seqlen for this specific sequence
        cur_state_len = state_len - (seqlen - (query_end_index - query_start_index))
        cur_seqlen = query_end_index - query_start_index
        x_offset = query_start_index * stride_x_token
        o_offset = query_start_index * stride_o_token
    else:
        query_start_index = i_n * seqlen
        query_end_index = query_start_index + seqlen
        cur_state_len = state_len
        cur_seqlen = seqlen
        x_offset = i_n * stride_x_seq
        o_offset = i_n * stride_o_seq

    if query_start_index == query_end_index:
        return

    # Speculative decoding: compute conv_state token offset
    if IS_SPEC_DECODING:
        conv_state_token_offset = (
            tl.load(num_accepted_tokens_ptr + i_n).to(tl.int64) - 1
        )
    else:
        conv_state_token_offset = 0

    # STEP 1: READ init_state prior tokens from conv_state
    # conv_states_base points to the feature slice for this sequence
    conv_states_base = (
        conv_state_ptr
        + (conv_states_input_coord * stride_istate_seq)
        + (i_d * stride_istate_dim)
    )
    mask_feat = i_d < dim

    prior_tokens = conv_states_base + conv_state_token_offset * stride_istate_tok
    if KERNEL_WIDTH >= 2:
        col0 = tl.load(prior_tokens + 0 * stride_istate_tok, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        col1 = tl.load(prior_tokens + 1 * stride_istate_tok, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        col2 = tl.load(prior_tokens + 2 * stride_istate_tok, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        col3 = tl.load(prior_tokens + 3 * stride_istate_tok, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        col4 = tl.load(prior_tokens + 4 * stride_istate_tok, mask_feat, 0.0).to(tl.float32)

    # STEP 2: Build new_conv_state (shift + append) and write back
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [NP2_STATELEN]

    # Load existing conv_state for shift
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_istate_seq)
        + conv_state_token_offset * stride_istate_tok
        + (i_d * stride_istate_dim)[None, :]
        + ((idx_tokens + (1 if IS_SPEC_DECODING else cur_seqlen)) * stride_istate_tok)[
            :, None
        ]
    )  # [NP2_STATELEN, BD]
    mask_shift = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + cur_seqlen) < cur_state_len)[:, None]
        & (i_d < dim)[None, :]
    )
    conv_state_shifted = tl.load(conv_state_ptrs_source, mask_shift, other=0.0)

    # Load x values to fill the new positions
    VAL = cur_state_len - cur_seqlen
    x_base = x_ptr + x_offset + (i_d * stride_x_dim)  # [BD]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [NP2_STATELEN, BD]
    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < cur_seqlen)[:, None]
        & (i_d < dim)[None, :]
    )
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    # Merge: keep shifted state where mask_shift is True, use x otherwise
    new_conv_state = tl.where(mask_shift, conv_state_shifted, loaded_x)

    # Determine target cache slot (for APC, may be different from source)
    conv_states_offset = tl.load(
        conv_state_indices_ptr + i_n * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_istate_seq)
        + (i_d * stride_istate_dim)
    )[None, :] + (
        idx_tokens * stride_istate_tok
    )[:, None]
    mask_target = (idx_tokens < cur_state_len)[:, None] & (i_d < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask_target)

    # STEP 3: Initialize accumulator with bias (or zero)
    if HAS_BIAS:
        acc_preload = tl.load(bias_ptr + i_d, mask=mask_feat, other=0.0).to(tl.float32)
    else:
        acc_preload = tl.zeros((BD,), dtype=tl.float32)

    # STEP 4: Pre-load weight columns
    w_base = weight_ptr + (i_d * stride_w_dim)  # [BD]
    if KERNEL_WIDTH >= 2:
        w_col0 = tl.load(w_base + 0 * stride_w_width, mask_feat, 0.0).to(tl.float32)
        w_col1 = tl.load(w_base + 1 * stride_w_width, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 3:
        w_col2 = tl.load(w_base + 2 * stride_w_width, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 4:
        w_col3 = tl.load(w_base + 3 * stride_w_width, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 5:
        w_col4 = tl.load(w_base + 4 * stride_w_width, mask_feat, 0.0).to(tl.float32)
    if KERNEL_WIDTH >= 6:
        w_col5 = tl.load(w_base + 5 * stride_w_width, mask_feat, 0.0).to(tl.float32)

    # STEP 5: Compute convolution output for each token in the sequence
    for idx_token in tl.range(seqlen):
        acc = acc_preload

        # Compute depthwise conv1d: sum of (prior_state[i] * weight[i]) for i in KERNEL_WIDTH
        # The last column always uses the current x token
        for j in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if j == 0:
                    matrix_w = w_col0
                    matrix_x = col0
                elif j == 1:
                    matrix_w = w_col1
                    matrix_x = tl.load(
                        x_base + idx_token * stride_x_token, mask_feat, 0.0
                    ).to(tl.float32)
            elif KERNEL_WIDTH == 3:
                if j == 0:
                    matrix_w = w_col0
                    matrix_x = col0
                elif j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = tl.load(
                        x_base + idx_token * stride_x_token, mask_feat, 0.0
                    ).to(tl.float32)
            elif KERNEL_WIDTH == 4:
                if j == 0:
                    matrix_w = w_col0
                    matrix_x = col0
                elif j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = tl.load(
                        x_base + idx_token * stride_x_token, mask_feat, 0.0
                    ).to(tl.float32)
            elif KERNEL_WIDTH == 5:
                if j == 0:
                    matrix_w = w_col0
                    matrix_x = col0
                elif j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = tl.load(
                        x_base + idx_token * stride_x_token, mask_feat, 0.0
                    ).to(tl.float32)
            elif KERNEL_WIDTH == 6:
                if j == 0:
                    matrix_w = w_col0
                    matrix_x = col0
                elif j == 1:
                    matrix_w = w_col1
                    matrix_x = col1
                elif j == 2:
                    matrix_w = w_col2
                    matrix_x = col2
                elif j == 3:
                    matrix_w = w_col3
                    matrix_x = col3
                elif j == 4:
                    matrix_w = w_col4
                    matrix_x = col4
                elif j == 5:
                    matrix_w = w_col5
                    matrix_x = tl.load(
                        x_base + idx_token * stride_x_token, mask_feat, 0.0
                    ).to(tl.float32)

            acc += matrix_x * matrix_w  # [BD] float32 accumulation

        # Update history columns (shift window)
        if KERNEL_WIDTH == 2:
            col0 = matrix_x
        elif KERNEL_WIDTH == 3:
            col0 = col1
            col1 = matrix_x
        elif KERNEL_WIDTH == 4:
            col0 = col1
            col1 = col2
            col2 = matrix_x
        elif KERNEL_WIDTH == 5:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = matrix_x
        elif KERNEL_WIDTH == 6:
            col0 = col1
            col1 = col2
            col2 = col3
            col3 = col4
            col4 = matrix_x

        # Apply SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))

        # Store output
        mask_1d = i_d < dim
        o_ptrs = (
            out_ptr + o_offset + idx_token * stride_o_token + (i_d * stride_o_dim)
        )
        tl.store(o_ptrs, acc, mask=mask_1d)
