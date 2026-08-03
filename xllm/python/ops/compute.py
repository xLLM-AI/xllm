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


quant_matmul = torch.ops.xllm_ops.quant_matmul


@torch.library.register_fake("xllm_ops::quant_matmul")
def _(x1, x2, transpose2, scale, offset, pertoken_scale, bias, output_dtype):
    out_last = x2.size(0) if transpose2 else x2.size(1)
    out_shape = list(x1.shape[:-1]) + [out_last]
    dtype = output_dtype if output_dtype is not None else torch.int8
    return x1.new_empty(out_shape, dtype=dtype)


quantize_per_tensor = torch.ops.xllm_ops.quantize_per_tensor


@torch.library.register_fake("xllm_ops::quantize_per_tensor")
def _(self, scales, zero_points, dtype, axis):
    return self.new_empty(self.shape, dtype=dtype)


dynamic_quant = torch.ops.xllm_ops.dynamic_quant


@torch.library.register_fake("xllm_ops::dynamic_quant")
def _(input, smooth_scales, group_index, dst_type):
    dtype = dst_type if dst_type is not None else torch.int8
    out = input.new_empty(input.shape, dtype=dtype)
    scale = input.new_empty(input.shape[:-1], dtype=torch.float32)
    return out, scale


lightning_indexer = torch.ops.xllm_ops.lightning_indexer


@torch.library.register_fake("xllm_ops::lightning_indexer")
def _(
    query,
    key,
    weights,
    query_seq_lengths,
    key_seq_lengths,
    block_table,
    layout_query,
    layout_key,
    selected_count,
    sparse_mode,
    pre_tokens,
    next_tokens,
    return_value,
):
    if layout_query == "BSND":
        key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
        out_shape = (query.size(0), query.size(1), key_head_num, selected_count)
    else:
        key_head_num = key.size(1) if layout_key == "TND" else key.size(2)
        out_shape = (query.size(0), key_head_num, selected_count)
    return query.new_zeros(out_shape, dtype=torch.int32)


scatter_nd_update = torch.ops.xllm_ops.scatter_nd_update


@torch.library.register_fake("xllm_ops::scatter_nd_update")
def _(var, indices, updates):
    return


sparse_flash_attention = torch.ops.xllm_ops.sparse_flash_attention


@torch.library.register_fake("xllm_ops::sparse_flash_attention")
def _(
    query,
    key,
    value,
    sparse_indices,
    block_table,
    actual_seq_lengths_query,
    actual_seq_lengths_kv,
    query_rope,
    key_rope,
    scale_value,
    sparse_block_size,
    layout_query,
    layout_kv,
    sparse_mode,
):
    return query.new_empty(query.shape, dtype=query.dtype)
