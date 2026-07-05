# Copyright 2025-2026 The xLLM Authors.
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

"""Op dispatch layer.

Every op calls the fused xLLM kernel exposed as ``torch.ops.xllm_ops.*``, the
*same* vendor kernel the C++ decoder path uses. These are registered inside the
xLLM binary / embedded interpreter; there is no pure-torch fallback, so the
Python model graph always runs the identical fused operators as C++. This is
where "multi-hardware operator compatibility" is realized on the Python side:
the graph calls a single symbol per op with no ``#ifdef``.
"""

from __future__ import annotations

import os
from typing import Tuple

import torch

# WS4/M8 gray switch: when set, silu_and_mul routes to the Python Triton kernel
# (xllm_triton) instead of the C++ vendor kernel (xllm_ops). Default off keeps
# the byte-identical C++ baseline.
_USE_TRITON = os.environ.get("XLLM_USE_TRITON", "0").strip().lower() not in (
    "",
    "0",
    "off",
    "false",
    "none",
)


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.ops.xllm_ops.rms_norm(x, weight, eps)


def fused_add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (RMSNorm(x + residual), x + residual).

    The xllm_ops kernel is IN-PLACE (declared ``(a!)`` void): it rewrites ``x``
    -> RMSNorm(x + residual) and ``residual`` -> x + residual, so we simply hand
    the (mutated) tensors back. No clone — the caller always passes a fresh
    ``x`` (attn/mlp output) and the carried ``residual`` accumulator, which are
    never the same tensor and are not reused after this call."""
    torch.ops.xllm_ops.fused_add_rms_norm(x, residual, weight, eps)
    return x, residual


# --------------------------------------------------------------------------
# Gated SiLU MLP activation
# --------------------------------------------------------------------------
def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    if _USE_TRITON:
        # Lazy import registers xllm_triton::silu_and_mul on first use; keeps the
        # default (C++) path free of the triton dependency.
        from ..kernels import triton_ops

        return triton_ops.silu_and_mul(x)
    return torch.ops.xllm_ops.silu_and_mul(x)


# --------------------------------------------------------------------------
# Fused per-head QK-RMSNorm + RoPE (Qwen3)
# --------------------------------------------------------------------------
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
    interleaved: bool = False,
) -> torch.Tensor:
    """Per-head RMSNorm (q_weight/k_weight over head_dim) then RoPE applied to
    the q and k slices of a packed ``qkv`` [num_tokens, (nq+nk+nv)*head_dim];
    the v slice is left untouched. Returns the updated packed qkv.

    Calls the fused xLLM CUDA kernel — the *same* op the C++ qwen3 path uses
    (``xllm_ops.fused_qk_norm_rope``). ``cos_sin_cache`` is the
    [max_pos, head_dim] cos||sin table from RotaryEmbedding (matching C++
    ``MRotaryEmbedding::precomputed_cos_sin_cache``).
    """
    # C++ passes int64 contiguous position_ids; mirror that for the kernel.
    pos = position_ids.to(torch.int64).contiguous()
    # IN-PLACE (declared ``(a!)`` void): the kernel normalizes+ropes the q/k
    # slices of ``qkv`` and leaves v untouched. ``qkv`` is a fresh qkv_proj
    # output that is not reused, so we mutate it directly (no clone) and return
    # it.
    torch.ops.xllm_ops.fused_qk_norm_rope(
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
        pos,
    )
    return qkv


# --------------------------------------------------------------------------
# Attention (paged, KV write + flashinfer) — C++ owns the paged-KV semantics.
# --------------------------------------------------------------------------
def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layer_id: int,
) -> torch.Tensor:
    """Attention over q/k/v (2D: [num_tokens, heads*head_dim]).

    Defers entirely to ``torch.ops.xllm_ops.attention`` which reads the KV
    cache, per-layer scale/head config and flashinfer plan from the C++
    thread-local forward context (registered in M3), keyed by ``layer_id``.
    """
    return torch.ops.xllm_ops.attention(q, k, v, layer_id)


# --------------------------------------------------------------------------
# Tensor-parallel collectives (WS1). No-op at tp_size==1: the C++ impl
# short-circuits when the TP group is null or has world_size 1, so the eager
# TP=1 parity path is byte-identical. The TP process group is read from the
# thread-local PyForwardContext, mirroring how ``attention`` reaches its C++
# state; the underlying calls are the same ``parallel_state::reduce`` /
# ``parallel_state::gather`` the native parallel layers use.
# --------------------------------------------------------------------------
def all_reduce(x: torch.Tensor) -> torch.Tensor:
    """SUM all-reduce across the TP group (RowParallel output combine)."""
    return torch.ops.xllm_ops.all_reduce(x)


def all_gather(x: torch.Tensor, dim: int, world_size: int) -> torch.Tensor:
    """All-gather across the TP group, concatenated along ``dim`` in rank order
    (ColumnParallel / embedding output combine). ``world_size`` (== tp_size) is
    forwarded so the op's ``register_fake`` can compute the gathered shape
    (``size(dim) * world_size``) at trace time; the live TP group in the C++
    forward context is authoritative at runtime."""
    return torch.ops.xllm_ops.all_gather(x, dim, world_size)
