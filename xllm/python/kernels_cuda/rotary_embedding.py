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

"""CUDA rotary-embedding kernels."""

from __future__ import annotations

import torch


def fused_qk_norm_rope(
    qkv: torch.Tensor,
    *,
    num_heads_q: int,
    num_heads_k: int,
    num_heads_v: int,
    head_dim: int,
    eps: float,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    interleaved: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply per-head QK-RMSNorm and RoPE to a packed QKV projection.

    Args:
        qkv: Packed projection of shape ``[num_tokens, q_size + 2 * kv_size]``.
        num_heads_q: Query heads on this rank.
        num_heads_k: Key heads on this rank.
        num_heads_v: Value heads on this rank.
        head_dim: Size of one head.
        eps: RMSNorm epsilon shared by the query and key norms.
        q_weight: Query RMSNorm weight of shape ``[head_dim]``.
        k_weight: Key RMSNorm weight of shape ``[head_dim]``.
        cos_sin_cache: Interleaved cosine/sine table indexed by position.
        position_ids: Position of every token.
        cos: Unused on CUDA; the kernel reads ``cos_sin_cache`` directly.
        sin: Unused on CUDA; the kernel reads ``cos_sin_cache`` directly.
        interleaved: Whether the rotary halves are interleaved.

    Returns:
        Query, key and value, each of shape ``[num_tokens, heads * head_dim]``.
        They may be views into one fused buffer, so a caller must not write
        into them in place.
    """
    del cos, sin
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    fused = torch.ops.xllm_ops.fused_qk_norm_rope(
        qkv,
        num_heads_q,
        num_heads_k,
        num_heads_v,
        head_dim,
        eps,
        q_weight,
        k_weight,
        cos_sin_cache,
        interleaved,
        position_ids,
    )
    return (
        fused[:, :q_size],
        fused[:, q_size : q_size + kv_size],
        fused[:, q_size + kv_size :],
    )


def interleaved_rotary_embedding(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE to a tensor whose rotary halves are interleaved.

    Args:
        value: Tensor of shape ``[num_tokens, num_heads, head_dim]``.
        cosine: Cosine table broadcastable over ``value``.
        sine: Sine table broadcastable over ``value``.

    Returns:
        A tensor with the same shape and dtype as ``value``.
    """
    del value, cosine, sine
    raise NotImplementedError(
        "interleaved_rotary_embedding has no CUDA kernel; models on CUDA use "
        "the non-interleaved rotary path in xllm.python.layers.rotary_embedding"
    )


__all__ = ["fused_qk_norm_rope", "interleaved_rotary_embedding"]
