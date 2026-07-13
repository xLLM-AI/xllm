from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
rms_norm = torch.ops.xllm_ops.rms_norm


@torch.library.register_fake("xllm_ops::rms_norm")
def _(input, weight, eps):
    return torch.empty_like(input)


# ---------------------------------------------------------------------------
# Fused residual + RMSNorm
# ---------------------------------------------------------------------------
fused_add_rms_norm = torch.ops.xllm_ops.fused_add_rms_norm


@torch.library.register_fake("xllm_ops::fused_add_rms_norm")
def _(input, residual, weight, eps):
    return input, residual


# ---------------------------------------------------------------------------
# Gated SiLU (SwiGLU activation)
# ---------------------------------------------------------------------------
silu_and_mul = torch.ops.xllm_ops.silu_and_mul


@torch.library.register_fake("xllm_ops::silu_and_mul")
def _(input):
    shape = list(input.shape)
    shape[-1] //= 2
    return input.new_empty(shape)


# ---------------------------------------------------------------------------
# Fused per-head QK-RMSNorm + RoPE
# ---------------------------------------------------------------------------
_fused_qk_norm_rope_impl = torch.ops.xllm_ops.fused_qk_norm_rope


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
    # position_ids must already be int64 and contiguous (see the kernel's
    # scalar-type check). Callers hoist this conversion out of the per-layer
    # loop, so it is not repeated here.
    return _fused_qk_norm_rope_impl(
        qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
        q_weight, k_weight, cos_sin_cache, interleaved,
        position_ids,
    )


@torch.library.register_fake("xllm_ops::fused_qk_norm_rope")
def _(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, interleaved, position_ids):
    return qkv
