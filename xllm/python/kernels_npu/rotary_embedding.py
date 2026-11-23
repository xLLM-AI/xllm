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
from xllm_kernel.ops.positional_encoding import (
    apply_qk_rotary,
    apply_rotary_embedding,
    build_split_qkv_rmsnorm_mrope_gather_pattern,
    fused_qk_norm_rope,
    has_split_qkv_rmsnorm_mrope_specialization,
    mrope,
    split_qkv_rmsnorm_mrope,
    vision_rotary_mul,
)
from xllm_kernel.ops.positional_encoding import inplace_partial_rotary_mul as _partial_rotary_mul


def has_fused_qk_rotary(head_dim: int, dtype: torch.dtype, device: torch.device) -> bool:
    return device.type == "npu" and dtype in (torch.float16, torch.bfloat16, torch.float32) and head_dim in (64, 128)


@torch.library.custom_op("xllm_python::interleaved_rotary_embedding", mutates_args=())
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
    output = torch_npu.npu_interleave_rope(value.view(num_tokens, num_heads, 1, head_dim), cosine, sine)
    return output.view(num_tokens, num_heads, head_dim)


def inplace_partial_rotary_mul(
    value: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    start: int,
    end: int,
) -> None:
    """Apply interleaved RoPE to one slice of ``value`` in place."""
    if value.dim() == 2:
        value_4d = value.unsqueeze(1).unsqueeze(1)
    elif value.dim() == 3:
        value_4d = value.unsqueeze(1)
    else:
        raise ValueError("inplace_partial_rotary_mul expects a 2D or 3D tensor")
    _partial_rotary_mul(value_4d, cosine, sine, start, end)


def npu_inplace_partial_rotary_mul(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_start_dim: int,
    rope_head_dim: int,
    inverse: bool = False,
) -> torch.Tensor:
    """In-place partial interleaved RoPE on the ``[rope_start_dim:rope_start_dim + rope_head_dim]`` slice.

    Mirrors C++ ``apply_partial_rope`` (deepseek_sparse_attention.cpp:151-190):
    x is 3D ``[M, n_head, head_dim]``; cos/sin are 2D ``[M, rope_head_dim]``
    (per-token, no head dim). Reshaped to 4D for the NPU kernel
    (``aclnnInplacePartialRotaryMul``, rotary_mode="interleave",
    partial_slice=[rope_start_dim, rope_start_dim+rope_head_dim] -- a half-open
    range, NOT [start, length]). Modifies x in place.
    """
    x4d = x.unsqueeze(2)  # [M, n_head, 1, head_dim]
    cos4d = cos.view(cos.size(0), 1, 1, cos.size(1))
    sin_cache = -sin if inverse else sin
    sin4d = sin_cache.view(sin.size(0), 1, 1, sin.size(1))
    _partial_rotary_mul(x4d, cos4d, sin4d, int(rope_start_dim), int(rope_start_dim + rope_head_dim))
    return x


__all__ = [
    "apply_qk_rotary",
    "apply_rotary_embedding",
    "build_split_qkv_rmsnorm_mrope_gather_pattern",
    "fused_qk_norm_rope",
    "has_fused_qk_rotary",
    "has_split_qkv_rmsnorm_mrope_specialization",
    "inplace_partial_rotary_mul",
    "interleaved_rotary_embedding",
    "mrope",
    "npu_inplace_partial_rotary_mul",
    "split_qkv_rmsnorm_mrope",
    "vision_rotary_mul",
]
