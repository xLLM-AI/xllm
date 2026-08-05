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

"""Fused post-convolution preparation for Qwen3.5 GDN prefill."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_gdn_prefill_post_conv_kernel(
    mixed_qkv_ptr,
    a_ptr,
    b_ptr,
    a_log_ptr,
    dt_bias_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    g_ptr,
    beta_ptr,
    stride_mixed_qkv_token,
    stride_a_token,
    stride_b_token,
    stride_q_token,
    stride_k_token,
    stride_v_token,
    num_tokens,
    NUM_KEY_HEADS: tl.constexpr,
    NUM_VALUE_HEADS: tl.constexpr,
    KEY_HEAD_DIM: tl.constexpr,
    VALUE_HEAD_DIM: tl.constexpr,
    L2_NORM_EPS: tl.constexpr,
    SOFTPLUS_THRESHOLD: tl.constexpr,
    BLOCK_TOKEN: tl.constexpr,
    BLOCK_KEY: tl.constexpr,
    BLOCK_VALUE: tl.constexpr,
):
    token_block = tl.program_id(0)
    head = tl.program_id(1)
    key_size: tl.constexpr = NUM_KEY_HEADS * KEY_HEAD_DIM

    tokens = (
        token_block * BLOCK_TOKEN + tl.arange(0, BLOCK_TOKEN)
    ).to(tl.int64)
    token_mask = tokens < num_tokens

    if head < NUM_KEY_HEADS:
        features = tl.arange(0, BLOCK_KEY)
        feature_mask = features < KEY_HEAD_DIM
        mask = token_mask[:, None] & feature_mask[None, :]

        q_offsets = (
            tokens[:, None] * stride_mixed_qkv_token
            + head * KEY_HEAD_DIM
            + features[None, :]
        )
        q = tl.load(mixed_qkv_ptr + q_offsets, mask=mask, other=0).to(
            tl.float32
        )
        k_offsets = q_offsets + key_size
        k = tl.load(mixed_qkv_ptr + k_offsets, mask=mask, other=0).to(
            tl.float32
        )

        q_square_sum = tl.sum(q * q, axis=1)
        q_inverse_norm = 1.0 / tl.sqrt(q_square_sum + L2_NORM_EPS)
        q *= q_inverse_norm[:, None]
        k_square_sum = tl.sum(k * k, axis=1)
        k_inverse_norm = 1.0 / tl.sqrt(k_square_sum + L2_NORM_EPS)
        k *= k_inverse_norm[:, None]

        q_output_offsets = (
            tokens[:, None] * stride_q_token
            + head * KEY_HEAD_DIM
            + features[None, :]
        )
        k_output_offsets = (
            tokens[:, None] * stride_k_token
            + head * KEY_HEAD_DIM
            + features[None, :]
        )
        tl.store(
            q_ptr + q_output_offsets,
            q.to(q_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            k_ptr + k_output_offsets,
            k.to(k_ptr.dtype.element_ty),
            mask=mask,
        )
    else:
        value_head = head - NUM_KEY_HEADS
        features = tl.arange(0, BLOCK_VALUE)
        feature_mask = features < VALUE_HEAD_DIM
        mask = token_mask[:, None] & feature_mask[None, :]
        value_offset: tl.constexpr = 2 * NUM_KEY_HEADS * KEY_HEAD_DIM
        v_offsets = (
            tokens[:, None] * stride_mixed_qkv_token
            + value_offset
            + value_head * VALUE_HEAD_DIM
            + features[None, :]
        )
        v = tl.load(mixed_qkv_ptr + v_offsets, mask=mask, other=0)
        v_output_offsets = (
            tokens[:, None] * stride_v_token
            + value_head * VALUE_HEAD_DIM
            + features[None, :]
        )
        tl.store(v_ptr + v_output_offsets, v, mask=mask)

        a_log = tl.load(a_log_ptr + value_head).to(tl.float32)
        dt_bias = tl.load(dt_bias_ptr + value_head).to(tl.float32)
        a_offsets = tokens * stride_a_token + value_head
        b_offsets = tokens * stride_b_token + value_head
        a = tl.load(a_ptr + a_offsets, mask=token_mask, other=0).to(tl.float32)
        b = tl.load(b_ptr + b_offsets, mask=token_mask, other=0).to(tl.float32)

        softplus_input = a + dt_bias
        softplus = tl.where(
            softplus_input > 0,
            softplus_input
            + tl.log(1.0 + tl.exp(-softplus_input)),
            tl.log(1.0 + tl.exp(softplus_input)),
        )
        softplus = tl.where(
            softplus_input <= SOFTPLUS_THRESHOLD,
            softplus,
            softplus_input,
        )
        g = -tl.exp(a_log) * softplus
        beta = tl.sigmoid(b)

        gate_offsets = tokens * NUM_VALUE_HEADS + value_head
        tl.store(g_ptr + gate_offsets, g, mask=token_mask)
        tl.store(beta_ptr + gate_offsets, beta, mask=token_mask)


def fused_gdn_prefill_post_conv(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_key_heads: int,
    key_head_dim: int,
    value_head_dim: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Prepare normalized Q/K and GDN gates from convolution output."""
    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    a_log = a_log.contiguous()
    dt_bias = dt_bias.contiguous()

    num_tokens = mixed_qkv.shape[0]
    num_value_heads = a_log.numel()
    expected_qkv_dim = (
        2 * num_key_heads * key_head_dim
        + num_value_heads * value_head_dim
    )
    if mixed_qkv.ndim != 2 or mixed_qkv.shape[1] != expected_qkv_dim:
        raise ValueError("mixed_qkv has an unexpected shape")
    expected_gate_shape = (num_tokens, num_value_heads)
    if a.shape != expected_gate_shape or b.shape != expected_gate_shape:
        raise ValueError("a and b must match the token and value-head dimensions")
    if dt_bias.shape != (num_value_heads,):
        raise ValueError("dt_bias must match the value-head dimension")

    q = torch.empty(
        num_tokens,
        num_key_heads,
        key_head_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    k = torch.empty_like(q)
    v = torch.empty(
        num_tokens,
        num_value_heads,
        value_head_dim,
        dtype=mixed_qkv.dtype,
        device=mixed_qkv.device,
    )
    g = torch.empty(
        num_tokens,
        num_value_heads,
        dtype=torch.float32,
        device=mixed_qkv.device,
    )
    beta = torch.empty_like(g)
    if num_tokens == 0:
        return q, k, v, g, beta

    block_token = 16
    grid = (
        triton.cdiv(num_tokens, block_token),
        num_key_heads + num_value_heads,
    )
    _fused_gdn_prefill_post_conv_kernel[grid](
        mixed_qkv,
        a,
        b,
        a_log,
        dt_bias,
        q,
        k,
        v,
        g,
        beta,
        mixed_qkv.stride(0),
        a.stride(0),
        b.stride(0),
        q.stride(0),
        k.stride(0),
        v.stride(0),
        num_tokens,
        NUM_KEY_HEADS=num_key_heads,
        NUM_VALUE_HEADS=num_value_heads,
        KEY_HEAD_DIM=key_head_dim,
        VALUE_HEAD_DIM=value_head_dim,
        L2_NORM_EPS=1e-6,
        SOFTPLUS_THRESHOLD=20.0,
        BLOCK_TOKEN=block_token,
        BLOCK_KEY=triton.next_power_of_2(key_head_dim),
        BLOCK_VALUE=triton.next_power_of_2(value_head_dim),
        num_warps=4,
        num_stages=2,
    )
    return q, k, v, g, beta
