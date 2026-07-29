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
        return _fused_qk_norm_rope_npu(
            qkv, num_heads_q, num_heads_k, head_dim, eps,
            q_weight, k_weight, cos, sin,
        )
    raise NotImplementedError(
        f"fused_qk_norm_rope is not supported on device type '{device_type}'"
    )


def _fused_qk_norm_rope_npu(
    qkv: torch.Tensor,
    num_heads_q: int, num_heads_k: int,
    head_dim: int, eps: float,
    q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos: torch.Tensor | None,
    sin: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.size(0)
    q_size = num_heads_q * head_dim
    k_size = num_heads_k * head_dim

    q = torch.ops.xllm_ops.rms_norm(
        qkv[:, :q_size].reshape(num_tokens * num_heads_q, head_dim), q_weight, eps,
    ).view(num_tokens, q_size)
    k = torch.ops.xllm_ops.rms_norm(
        qkv[:, q_size:q_size + k_size].reshape(num_tokens * num_heads_k, head_dim),
        k_weight, eps,
    ).view(num_tokens, k_size)

    q_out = torch.ops.npu.npu_rotary_mul(
        q.view(1, num_tokens, num_heads_q, head_dim), cos, sin,
    ).view(num_tokens, q_size)
    k_out = torch.ops.npu.npu_rotary_mul(
        k.view(1, num_tokens, num_heads_k, head_dim), cos, sin,
    ).view(num_tokens, k_size)

    v = qkv[:, q_size + k_size:]
    return q_out, k_out, v


@torch.library.register_fake("xllm_ops::fused_qk_norm_rope")
def _(qkv, num_heads_q, num_heads_k, num_heads_v, head_dim, eps, q_weight, k_weight, cos_sin_cache, interleaved, position_ids):
    return qkv
