from __future__ import annotations

import torch

from xllm.python.model_executor.forward_context import get_forward_context

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
    q_size = num_heads_q * head_dim
    kv_size = num_heads_k * head_dim
    device_type = get_forward_context().device.type
    if device_type == "cuda":
        qkv = torch.ops.xllm_ops.fused_qk_norm_rope(
            qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps,
            q_weight, k_weight, cos_sin_cache, interleaved, position_ids,
        )
        return qkv[:, :q_size], qkv[:, q_size:q_size + kv_size], qkv[:, q_size + kv_size:]
    if device_type in ("npu", "privateuseone"):
        from xllm.python.ops.triton.split_qkv_rmsnorm_rope import (
            split_qkv_rmsnorm_rope,
        )
        return split_qkv_rmsnorm_rope(
            qkv, cos_sin_cache, position_ids,
            q_weight, k_weight,
            q_size, kv_size, head_dim, eps,
        )
    raise NotImplementedError(
        f"fused_qk_norm_rope is not supported on device type '{device_type}'"
    )


@torch.library.register_fake("xllm_ops::fused_qk_norm_rope")
def _(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, interleaved, position_ids):
    return qkv
