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


@triton.jit()
def tmo_causal_conv1d_update_decode_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    num_accepted_tokens_ptr,
    query_start_loc_ptr,  # (batch + 1)
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.int64,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.int64,
    # others
    null_block_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    IS_SPEC_DECODING: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    HAS_NULL_BLOCK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_B: tl.constexpr = 1024,
):
    pid = tl.program_id(0)
    num_jobs = tl.num_programs(0)
    num_n = (dim + BLOCK_N - 1) // BLOCK_N

    NEED_MASK_N = dim % BLOCK_N != 0
    total_blocks = batch* ((dim + BLOCK_N - 1) // BLOCK_N)
    
    arange_batch = tl.arange(0, BLOCK_B)
    mask_batch = arange_batch < batch
    # Pre-load weights outside the persistent loop
    if IS_APC_ENABLED:
        conv_state_inits = tl.load(initial_state_idx + arange_batch, mask = mask_batch)
        current_last_indexs = tl.load(block_idx_last_scheduled_token + arange_batch, mask = mask_batch)
    else:
        conv_state_inits = 0
        current_last_indexs = 0
    
    # conv_state_init = 0
    # current_last_index = 0
    if IS_APC_ENABLED:
        pass
    else:
        conv_states_input_coords = tl.load(
                conv_state_indices_ptr + arange_batch * stride_state_indices, 
                mask = mask_batch,
                cache_modifier=".cg",
            ).to(tl.int64)
    w_base = w_ptr + (tl.arange(0, ((dim + BLOCK_N - 1) // BLOCK_N)*BLOCK_N) * stride_w_dim)[:, None] + tl.arange(0, KERNEL_WIDTH)[None, :]  # [BLOCK_N, KERNEL_WIDTH]
    mask_wraw = (tl.arange(0, ((dim + BLOCK_N - 1) // BLOCK_N)*BLOCK_N) < dim)[:, None]
    w_raw = tl.load(w_base, mask_wraw, other=0.0, cache_modifier=".cg")
    ws = tl.trans(w_raw)  # [KERNEL_WIDTH, BLOCK_N*num_n]
    # mask_w = (tl.arange(0, BLOCK_N) < dim)[:, None]
    for flat_pid in range(pid, total_blocks, num_jobs):
        idx_seq = flat_pid // num_n
        pid_n = flat_pid % num_n

        # [BLOCK_N,] elements along the feature-dimension (channel)
        idx_feats = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_bias = idx_feats < dim

        # cache_idx
        if IS_APC_ENABLED:
            conv_state_init = conv_state_inits[idx_seq]
            current_last_index = current_last_indexs[idx_seq]
            conv_states_input_coord = tl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
            ).to(tl.int64)
            conv_states_offset = tl.load(
                conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
            ).to(tl.int64)
        else:
            conv_state_init = 0
            current_last_index = 0
            conv_states_input_coord = conv_states_input_coords[idx_seq]
            conv_states_offset = conv_states_input_coords[idx_seq]

        should_process = True
        if HAS_NULL_BLOCK:
            if conv_states_input_coord == null_block_id:
                should_process = False

        if should_process:
            if IS_VARLEN:
                query_start_index = tl.load(query_start_loc_ptr + idx_seq).to(tl.int64)
                query_end_index = tl.load(query_start_loc_ptr + (idx_seq + 1)).to(tl.int64)
                actual_seqlen = query_end_index - query_start_index
                actual_state_len = state_len - (seqlen - actual_seqlen)
                x_offset = query_start_index * stride_x_token
                o_offset = query_start_index * stride_o_token
            else:
                query_start_index = idx_seq * seqlen
                query_end_index = query_start_index + seqlen
                actual_seqlen = seqlen
                actual_state_len = state_len
                x_offset = idx_seq * stride_x_seq
                o_offset = idx_seq * stride_o_seq

            if query_start_index != query_end_index:
                if IS_SPEC_DECODING:
                    conv_state_token_offset = (
                        tl.load(num_accepted_tokens_ptr + idx_seq).to(tl.int64) - 1
                    )
                else:
                    conv_state_token_offset = 0

                # STEP 1: READ init_state data
                col_offsets = tl.arange(0, KERNEL_WIDTH - 1) * stride_conv_state_tok  # [KERNEL_WIDTH-1]
                conv_states_base = (
                    conv_state_ptr
                    + (conv_states_input_coord * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)[:, None]
                    + col_offsets[None, :]
                )

                prior_tokens = conv_states_base + (conv_state_token_offset * stride_conv_state_tok)[None, :]
                oldState_raw = tl.load(prior_tokens, mask_bias[:, None], 0.0, cache_modifier=".cg")
                oldState = tl.trans(oldState_raw)

                # STEP 2: assume state_len > seqlen
                x_ptrs = x_ptr + x_offset + (idx_feats * stride_x_dim)[None, :] + (tl.arange(0, seqlen)* stride_x_token)[:, None]  # [seqlen, BLOCK_N]
                xs = tl.load(x_ptrs, (tl.arange(0, seqlen)<actual_seqlen)[:, None]&mask_bias[None, :], 0.0, cache_modifier=".cg")

                tl.debug_barrier()




                FULL_LEN: tl.constexpr = KERNEL_WIDTH - 1 + seqlen
                new_conv_state = tl.empty((FULL_LEN, BLOCK_N), dtype=conv_states_base.dtype.element_ty)
                new_conv_state[0:KERNEL_WIDTH - 1, :] = oldState
                new_conv_state[KERNEL_WIDTH - 1 : FULL_LEN, :] = xs

                # STEP 3: init accumulator
                if HAS_BIAS:
                    bias = bias_ptr + idx_feats
                    acc_preload = tl.load(bias, mask=mask_bias, other=0.0, cache_modifier=".cg").to(
                        tl.float32
                    )  # [BLOCK_N]
                else:
                    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

                # STEP 5: compute each token
                for idx_token in range(actual_seqlen):
                    mask_1d = (idx_token < actual_seqlen) & (
                        idx_feats < dim
                    )

                    acc = acc_preload

                    x_cur = xs[idx_token, :]

                    for idx_width in tl.static_range(KERNEL_WIDTH):
                        if idx_width == KERNEL_WIDTH - 1:
                            matrix_x = x_cur
                        else:
                            matrix_x = oldState[idx_width, :]
                        acc += (
                            ws[idx_width, pid_n * BLOCK_N:(pid_n + 1) * BLOCK_N]
                            * matrix_x
                        )

                    if KERNEL_WIDTH > 2:
                        oldState[0:KERNEL_WIDTH - 2, :] = oldState[1:KERNEL_WIDTH - 1, :]
                    oldState[KERNEL_WIDTH - 2, :] = x_cur
                    if SILU_ACTIVATION:
                        acc = acc / (1 + tl.exp(-acc))
                    o_ptrs = (
                        o_ptr + o_offset + idx_token * stride_o_token + (idx_feats * stride_o_dim)
                    )

                    tl.store(o_ptrs, acc, mask=mask_1d)


                if IS_SPEC_DECODING:
                    oldstart = 1
                else:
                    oldstart = actual_seqlen
                cover_lo = state_len - actual_seqlen
                new_state_rows = tl.empty((state_len, BLOCK_N), dtype=oldState.dtype)
                for gi in tl.static_range(state_len):
                    in_cover = (gi >= cover_lo) & (gi < state_len)
                    src_row = tl.where(
                        in_cover,
                        (gi - cover_lo) + (KERNEL_WIDTH - 1),
                        oldstart + gi,
                    )
                    new_state_rows[gi, :] = new_conv_state[src_row, :]
                new_state_data = tl.trans(new_state_rows)  # [BLOCK_N, state_len]

                idx_tokens_store = tl.arange(0, state_len)
                conv_state_ptrs_target = (
                    conv_state_ptr
                    + (conv_states_offset * stride_conv_state_seq)
                    + (idx_feats * stride_conv_state_dim)
                )[:, None] + (
                    idx_tokens_store * stride_conv_state_tok
                )[None, :]
                mask_store = (idx_tokens_store < actual_state_len)[None, :] & mask_bias[:, None]
                if NEED_MASK_N:
                    tl.store(conv_state_ptrs_target, new_state_data, mask_store)
                else:
                    tl.store(conv_state_ptrs_target, new_state_data,
                             mask=idx_tokens_store[None, :] < actual_state_len)

