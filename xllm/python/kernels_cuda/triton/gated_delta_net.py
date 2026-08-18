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

"""Triton kernels for gated delta network recurrent updates.

The packed decode kernel is aligned with vLLM's vendored
flash-linear-attention implementation.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_recurrent_gated_delta_rule_packed_decode_kernel(
    mixed_qkv,
    a,
    b,
    a_log,
    dt_bias,
    output,
    initial_state,
    final_state,
    state_indices,
    scale,
    stride_mixed_qkv_token: tl.constexpr,
    stride_a_token: tl.constexpr,
    stride_b_token: tl.constexpr,
    stride_initial_state_token: tl.constexpr,
    stride_final_state_token: tl.constexpr,
    stride_indices_sequence: tl.constexpr,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_VALUE_HEADS: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    USE_QK_L2NORM: tl.constexpr,
):
    value_block = tl.program_id(0)
    sequence_head = tl.program_id(1)
    sequence = sequence_head // NUM_VALUE_HEADS
    value_head = sequence_head % NUM_VALUE_HEADS
    key_head = value_head // (NUM_VALUE_HEADS // NUM_KEY_HEADS)

    key_offsets = tl.arange(0, BLOCK_KEY)
    value_offsets = value_block * BLOCK_VALUE + tl.arange(0, BLOCK_VALUE)
    key_mask = key_offsets < KEY_DIM
    value_mask = value_offsets < VALUE_DIM
    state_mask = value_mask[:, None] & key_mask[None, :]

    state_index = tl.load(state_indices + sequence * stride_indices_sequence).to(tl.int64)
    output_ptr = output + (sequence * NUM_VALUE_HEADS + value_head) * VALUE_DIM + value_offsets

    # xLLM and vLLM both reserve cache row 0 as NULL_BLOCK_ID.
    if state_index <= 0:
        zeros = tl.zeros([BLOCK_VALUE], dtype=tl.float32).to(output_ptr.dtype.element_ty)
        tl.store(output_ptr, zeros, mask=value_mask)
        return

    initial_state_ptr = initial_state + state_index * stride_initial_state_token
    initial_state_ptr = (
        initial_state_ptr + value_head * VALUE_DIM * KEY_DIM + value_offsets[:, None] * KEY_DIM + key_offsets[None, :]
    )
    recurrent_state = tl.load(initial_state_ptr, mask=state_mask, other=0).to(tl.float32)

    mixed_ptr = mixed_qkv + sequence * stride_mixed_qkv_token
    query_offsets = key_head * KEY_DIM + key_offsets
    key_offsets_packed = NUM_KEY_HEADS * KEY_DIM + query_offsets
    value_offsets_packed = 2 * NUM_KEY_HEADS * KEY_DIM + value_head * VALUE_DIM + value_offsets
    query = tl.load(mixed_ptr + query_offsets, mask=key_mask, other=0).to(tl.float32)
    key = tl.load(mixed_ptr + key_offsets_packed, mask=key_mask, other=0).to(tl.float32)
    value = tl.load(mixed_ptr + value_offsets_packed, mask=value_mask, other=0).to(tl.float32)

    if USE_QK_L2NORM:
        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
    query = query * scale

    a_value = tl.load(a + sequence * stride_a_token + value_head).to(tl.float32)
    b_value = tl.load(b + sequence * stride_b_token + value_head).to(tl.float32)
    a_log_value = tl.load(a_log + value_head).to(tl.float32)
    dt_bias_value = tl.load(dt_bias + value_head).to(tl.float32)
    softplus_input = a_value + dt_bias_value
    softplus = tl.where(
        softplus_input <= SOFTPLUS_THRESHOLD,
        tl.log(1.0 + tl.exp(softplus_input)),
        softplus_input,
    )
    decay = -tl.exp(a_log_value) * softplus
    beta = tl.sigmoid(b_value).to(b.dtype.element_ty).to(tl.float32)

    recurrent_state *= tl.exp(decay)
    value -= tl.sum(recurrent_state * key[None, :], 1)
    value *= beta
    recurrent_state += value[:, None] * key[None, :]
    result = tl.sum(recurrent_state * query[None, :], 1)
    tl.store(output_ptr, result.to(output_ptr.dtype.element_ty), mask=value_mask)

    final_state_ptr = final_state + state_index * stride_final_state_token
    final_state_ptr = (
        final_state_ptr + value_head * VALUE_DIM * KEY_DIM + value_offsets[:, None] * KEY_DIM + key_offsets[None, :]
    )
    tl.store(
        final_state_ptr,
        recurrent_state.to(final_state_ptr.dtype.element_ty),
        mask=state_mask,
    )


def fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    state_indices: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Run vLLM's packed single-token GDN recurrent update in place."""
    if mixed_qkv.ndim != 2 or mixed_qkv.stride(-1) != 1:
        raise ValueError("mixed_qkv must be 2D and contiguous in its last dim")
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("a and b must be 2D tensors")
    if a.stride(-1) != 1 or b.stride(-1) != 1:
        raise ValueError("a and b must be contiguous in their last dim")
    if a_log.ndim != 1 or dt_bias.ndim != 1:
        raise ValueError("a_log and dt_bias must be 1D tensors")
    if state_indices.ndim != 1:
        raise ValueError("state_indices must be a 1D tensor")
    if initial_state.ndim != 4 or initial_state.stride(-1) != 1:
        raise ValueError("initial_state must be 4D and contiguous in its last dim")

    batch = mixed_qkv.shape[0]
    if a.shape[0] != batch or b.shape[0] != batch:
        raise ValueError("mixed_qkv, a and b batch dimensions must match")
    if state_indices.shape[0] != batch:
        raise ValueError("state_indices must contain one slot per decode row")

    num_value_heads, value_dim, key_dim = initial_state.shape[-3:]
    if a.shape[1] != num_value_heads or b.shape[1] != num_value_heads:
        raise ValueError("a and b value-head dimensions do not match state")
    if a_log.numel() != num_value_heads or dt_bias.numel() != num_value_heads:
        raise ValueError("a_log and dt_bias do not match the value-head count")

    qkv_dim = mixed_qkv.shape[1]
    query_key_dim = qkv_dim - num_value_heads * value_dim
    if query_key_dim <= 0 or query_key_dim % 2 != 0:
        raise ValueError("mixed_qkv has an invalid packed dimension")
    query_dim = query_key_dim // 2
    if query_dim % key_dim != 0:
        raise ValueError("packed query dimension is not divisible by key_dim")
    num_key_heads = query_dim // key_dim
    if num_key_heads <= 0 or num_value_heads % num_key_heads != 0:
        raise ValueError("packed QKV has an invalid grouped-head configuration")

    output = torch.empty(
        batch,
        1,
        num_value_heads,
        value_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    block_key = triton.next_power_of_2(key_dim)
    block_value = min(triton.next_power_of_2(value_dim), 32)
    grid = (triton.cdiv(value_dim, block_value), batch * num_value_heads)
    _fused_recurrent_gated_delta_rule_packed_decode_kernel[grid](
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        a_log=a_log,
        dt_bias=dt_bias,
        output=output,
        initial_state=initial_state,
        final_state=initial_state,
        state_indices=state_indices,
        scale=scale,
        stride_mixed_qkv_token=mixed_qkv.stride(0),
        stride_a_token=a.stride(0),
        stride_b_token=b.stride(0),
        stride_initial_state_token=initial_state.stride(0),
        stride_final_state_token=initial_state.stride(0),
        stride_indices_sequence=state_indices.stride(0),
        NUM_KEY_HEADS=num_key_heads,
        NUM_VALUE_HEADS=num_value_heads,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        BLOCK_KEY=block_key,
        BLOCK_VALUE=block_value,
        SOFTPLUS_THRESHOLD=20.0,
        USE_QK_L2NORM=True,
        num_warps=1,
        num_stages=3,
    )
    return output
