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

"""NPU rotary-embedding kernels."""

from __future__ import annotations

import torch
import torch_npu


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
        num_heads_v: Value heads on this rank; derived from ``qkv`` here.
        head_dim: Size of one head.
        eps: RMSNorm epsilon shared by the query and key norms.
        q_weight: Query RMSNorm weight of shape ``[head_dim]``.
        k_weight: Key RMSNorm weight of shape ``[head_dim]``.
        cos_sin_cache: Interleaved cosine/sine table indexed by position.
        position_ids: Position of every token.
        cos: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
        sin: Unused on NPU; the kernel reads ``cos_sin_cache`` directly.
        interleaved: Unused on NPU; the kernel always uses the split layout.

    Returns:
        Query, key and value, each of shape ``[num_tokens, heads * head_dim]``.
        They may be views into one fused buffer, so a caller must not write
        into them in place.
    """
    del num_heads_v, cos, sin, interleaved
    from .triton.split_qkv_rmsnorm_rope import (
        split_qkv_rmsnorm_rope,
    )

    return split_qkv_rmsnorm_rope(
        qkv,
        cos_sin_cache,
        position_ids,
        q_weight,
        k_weight,
        num_heads_q * head_dim,
        num_heads_k * head_dim,
        head_dim,
        eps,
    )


@torch.library.custom_op(
    "xllm_python::interleaved_rotary_embedding", mutates_args=()
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
        A tensor with the shape and dtype of ``value``.
    """
    num_tokens, num_heads, head_dim = value.shape
    output = torch_npu.npu_interleave_rope(
        value.view(num_tokens, num_heads, 1, head_dim), cosine, sine
    )
    return output.view(num_tokens, num_heads, head_dim)


@interleaved_rotary_embedding.register_fake
def _interleaved_rotary_embedding_fake(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> torch.Tensor:
    del cosine, sine
    return torch.empty_like(value)


def mrope(
    positions: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_dim: int,
    *,
    mrope_section: list[int],
    rotary_mode: str = "half",
    cache_mode: str = "interleave",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply multi-dimensional RoPE (mRoPE) to query and key tensors.

    Wraps ``torch_npu.npu_mrope``.

    Args:
        positions: ``[3, num_tokens]`` position ids (time/height/width), dtype int64.
        q: ``[num_tokens, q_size]`` query tensor.
        k: ``[num_tokens, kv_size]`` key tensor.
        cos_sin_cache: ``[max_pos, head_dim]`` cos/sin table.
        head_dim: Per-head dimension.
        mrope_section: Section sizes for time/height/width.
        rotary_mode: Rotation mode (default ``"half"``).
        cache_mode: Cache layout mode (default ``"interleave"``).

    Returns:
        Rotated (q, k) with same shapes as inputs.
    """
    import torch_npu

    return torch_npu.npu_mrope(
        positions,
        q,
        k,
        cos_sin_cache,
        head_dim,
        mrope_section=list(mrope_section),
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )


def vision_rotary_mul(
    value: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
) -> torch.Tensor:
    """Apply RoPE via ``torch_npu.npu_rotary_mul`` (neox/half mode).

    Args:
        value: ``(total_tokens, num_heads, head_dim)``.
        cos_full: ``(1, total_tokens, 1, head_dim)``.
        sin_full: ``(1, total_tokens, 1, head_dim)``.

    Returns:
        Rotated tensor with same shape as ``value``.
    """
    import torch_npu

    return torch_npu.npu_rotary_mul(
        value.unsqueeze(0).contiguous(), cos_full, sin_full
    ).squeeze(0)


__all__ = [
    "fused_qk_norm_rope",
    "interleaved_rotary_embedding",
    "mrope",
    "vision_rotary_mul",
]
