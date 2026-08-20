# Copyright 2026 The xLLM Authors.
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

"""Triton causal-conv1d kernels for Qwen3.5.

The state update and convolution follow vLLM's ``causal_conv1d_update``
standard continuous-batching path. Cache row 0 is NULL_BLOCK_ID.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _causal_conv1d_update_kernel(
    input_ptr,
    weight_ptr,
    conv_state_ptr,
    state_indices_ptr,
    has_initial_state_ptr,
    query_start_loc_ptr,
    output_ptr,
    batch,
    dim: tl.constexpr,
    max_sequence_length: tl.constexpr,
    state_length: tl.constexpr,
    stride_input_sequence: tl.constexpr,
    stride_input_dim: tl.constexpr,
    stride_input_token: tl.int64,
    stride_weight_dim: tl.constexpr,
    stride_weight_width: tl.constexpr,
    stride_state_sequence: tl.constexpr,
    stride_state_dim: tl.constexpr,
    stride_state_token: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_output_sequence: tl.constexpr,
    stride_output_dim: tl.constexpr,
    stride_output_token: tl.int64,
    KERNEL_WIDTH: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_INITIAL_STATE_MASK: tl.constexpr,
    BLOCK_STATE: tl.constexpr,
    BLOCK_DIM: tl.constexpr,
):
    sequence = tl.program_id(0)
    if sequence >= batch:
        return

    features = tl.program_id(1) * BLOCK_DIM + tl.arange(0, BLOCK_DIM)
    feature_mask = features < dim
    state_index = tl.load(state_indices_ptr + sequence * stride_state_indices).to(tl.int64)
    if state_index == 0:
        return
    has_initial_state = True
    if USE_INITIAL_STATE_MASK:
        has_initial_state = tl.load(has_initial_state_ptr + sequence)

    if IS_VARLEN:
        query_start = tl.load(query_start_loc_ptr + sequence).to(tl.int64)
        query_end = tl.load(query_start_loc_ptr + sequence + 1).to(tl.int64)
        sequence_length = query_end - query_start
        input_offset = query_start * stride_input_token
        output_offset = query_start * stride_output_token
    else:
        query_start = sequence * max_sequence_length
        query_end = query_start + max_sequence_length
        sequence_length = max_sequence_length
        input_offset = sequence * stride_input_sequence
        output_offset = sequence * stride_output_sequence

    if query_start == query_end:
        return

    state_base = conv_state_ptr + state_index * stride_state_sequence + features * stride_state_dim
    history_mask = feature_mask & has_initial_state
    if KERNEL_WIDTH >= 2:
        history_0 = tl.load(state_base, mask=history_mask, other=0.0)
    if KERNEL_WIDTH >= 3:
        history_1 = tl.load(
            state_base + stride_state_token,
            mask=history_mask,
            other=0.0,
        )
    if KERNEL_WIDTH >= 4:
        history_2 = tl.load(
            state_base + 2 * stride_state_token,
            mask=history_mask,
            other=0.0,
        )

    state_tokens = tl.arange(0, BLOCK_STATE)
    old_state_ptrs = (
        conv_state_ptr
        + state_index * stride_state_sequence
        + features[None, :] * stride_state_dim
        + (state_tokens + sequence_length)[:, None] * stride_state_token
    )
    old_state_mask = (
        (state_tokens + sequence_length < state_length)[:, None] & feature_mask[None, :] & has_initial_state
    )
    old_state = tl.load(old_state_ptrs, mask=old_state_mask, other=0.0)

    input_base = input_ptr + input_offset + features * stride_input_dim
    first_tail_token = state_length - sequence_length
    tail_input_ptrs = input_base[None, :] + (state_tokens - first_tail_token)[:, None] * stride_input_token
    tail_input_mask = (
        (state_tokens - first_tail_token >= 0)[:, None]
        & (state_tokens - first_tail_token < sequence_length)[:, None]
        & feature_mask[None, :]
    )
    tail_input = tl.load(tail_input_ptrs, mask=tail_input_mask, other=0.0)
    tl.debug_barrier()
    updated_state = tl.where(old_state_mask, old_state, tail_input)
    state_output_ptrs = (
        conv_state_ptr
        + state_index * stride_state_sequence
        + features[None, :] * stride_state_dim
        + state_tokens[:, None] * stride_state_token
    )
    state_output_mask = (state_tokens < state_length)[:, None] & feature_mask[None, :]
    tl.store(state_output_ptrs, updated_state, mask=state_output_mask)

    weight_base = weight_ptr + features * stride_weight_dim
    if KERNEL_WIDTH >= 2:
        weight_0 = tl.load(weight_base, mask=feature_mask, other=0.0)
        weight_1 = tl.load(
            weight_base + stride_weight_width,
            mask=feature_mask,
            other=0.0,
        )
    if KERNEL_WIDTH >= 3:
        weight_2 = tl.load(
            weight_base + 2 * stride_weight_width,
            mask=feature_mask,
            other=0.0,
        )
    if KERNEL_WIDTH >= 4:
        weight_3 = tl.load(
            weight_base + 3 * stride_weight_width,
            mask=feature_mask,
            other=0.0,
        )

    for token in tl.range(sequence_length):
        accumulator = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
        matrix_weight = weight_0
        matrix_input = history_0
        for width_index in tl.static_range(KERNEL_WIDTH):
            if KERNEL_WIDTH == 2:
                if width_index == 1:
                    matrix_weight = weight_1
                    matrix_input = tl.load(
                        input_base + token * stride_input_token,
                        mask=feature_mask,
                        other=0.0,
                    )
            elif KERNEL_WIDTH == 3:
                if width_index == 1:
                    matrix_weight = weight_1
                    matrix_input = history_1
                elif width_index == 2:
                    matrix_weight = weight_2
                    matrix_input = tl.load(
                        input_base + token * stride_input_token,
                        mask=feature_mask,
                        other=0.0,
                    )
            elif KERNEL_WIDTH == 4:
                if width_index == 1:
                    matrix_weight = weight_1
                    matrix_input = history_1
                elif width_index == 2:
                    matrix_weight = weight_2
                    matrix_input = history_2
                elif width_index == 3:
                    matrix_weight = weight_3
                    matrix_input = tl.load(
                        input_base + token * stride_input_token,
                        mask=feature_mask,
                        other=0.0,
                    )
            accumulator += matrix_input * matrix_weight

        if KERNEL_WIDTH == 2:
            history_0 = matrix_input
        elif KERNEL_WIDTH == 3:
            history_0 = history_1
            history_1 = matrix_input
        elif KERNEL_WIDTH == 4:
            history_0 = history_1
            history_1 = history_2
            history_2 = matrix_input

        accumulator = accumulator / (1 + tl.exp(-accumulator))
        tl.store(
            output_ptr + output_offset + token * stride_output_token + features * stride_output_dim,
            accumulator,
            mask=feature_mask,
        )


def _conv_state_dim_first(conv_state: torch.Tensor, dim: int) -> torch.Tensor:
    if conv_state.ndim != 3:
        raise ValueError("conv_state must be a 3D tensor")
    if conv_state.shape[1] == dim:
        return conv_state
    if conv_state.shape[2] == dim:
        return conv_state.transpose(1, 2)
    raise ValueError("conv_state does not contain the convolution dimension")


def _launch_causal_conv1d_update(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor | None,
    query_start_loc: torch.Tensor | None,
    max_sequence_length: int,
) -> torch.Tensor:
    original_dtype = value.dtype
    value = value.to(conv_state.dtype)
    dim = value.shape[1]
    conv_state = _conv_state_dim_first(conv_state, dim)
    width = weight.shape[1]
    state_length = width - 1
    if width < 2 or width > 4:
        raise ValueError("causal_conv1d supports kernel widths from 2 through 4")
    if conv_state.shape[2] < state_length:
        raise ValueError("conv_state is shorter than the convolution history")

    is_varlen = query_start_loc is not None
    use_initial_state_mask = has_initial_state is not None
    if use_initial_state_mask and has_initial_state.shape != state_indices.shape:
        raise ValueError("has_initial_state must match state_indices")
    if is_varlen:
        batch = state_indices.shape[0]
        stride_input_sequence = 0
        stride_input_token, stride_input_dim = value.stride()
        stride_output_sequence = 0
    else:
        batch = value.shape[0]
        value = value.unsqueeze(-1)
        stride_input_sequence, stride_input_dim, stride_input_token = value.stride()

    output = torch.zeros_like(value)
    if is_varlen:
        stride_output_token, stride_output_dim = output.stride()
        stride_output_sequence = 0
    else:
        stride_output_sequence, stride_output_dim, stride_output_token = output.stride()

    block_state = triton.next_power_of_2(state_length)
    grid = (batch, triton.cdiv(dim, 256))
    _causal_conv1d_update_kernel[grid](
        value,
        weight,
        conv_state,
        state_indices,
        has_initial_state,
        query_start_loc,
        output,
        batch,
        dim,
        max_sequence_length,
        state_length,
        stride_input_sequence,
        stride_input_dim,
        stride_input_token,
        weight.stride(0),
        weight.stride(1),
        conv_state.stride(0),
        conv_state.stride(1),
        conv_state.stride(2),
        state_indices.stride(0),
        stride_output_sequence,
        stride_output_dim,
        stride_output_token,
        KERNEL_WIDTH=width,
        IS_VARLEN=is_varlen,
        USE_INITIAL_STATE_MASK=use_initial_state_mask,
        BLOCK_STATE=block_state,
        BLOCK_DIM=256,
    )
    if not is_varlen:
        output = output.squeeze(-1)
    return output.to(original_dtype)


def causal_conv1d_prefill(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    """Run vLLM-style varlen causal conv and update state in place."""
    if query_start_loc.numel() != state_indices.numel() + 1:
        raise ValueError("query_start_loc must contain one boundary per sequence")
    if has_initial_state.numel() != state_indices.numel():
        raise ValueError("has_initial_state must contain one value per sequence")
    # TODO: Avoid the device-to-host synchronization here. The varlen Triton
    # kernel already reads query_start_loc, so the max length should be passed
    # as a device-side value or eliminated in the next performance PR.
    sequence_lengths = query_start_loc.diff()
    max_sequence_length = int(sequence_lengths.max().item())
    return _launch_causal_conv1d_update(
        value,
        weight,
        conv_state,
        state_indices,
        has_initial_state,
        query_start_loc,
        max_sequence_length,
    )


def causal_conv1d_decode(
    value: torch.Tensor,
    weight: torch.Tensor,
    conv_state: torch.Tensor,
    state_indices: torch.Tensor,
) -> torch.Tensor:
    """Run vLLM-style single-token causal conv and update state in place."""
    if value.shape[0] != state_indices.shape[0]:
        raise ValueError("decode requires one state index per input row")
    return _launch_causal_conv1d_update(
        value,
        weight,
        conv_state,
        state_indices,
        None,
        None,
        1,
    )
