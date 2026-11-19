# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Weight preparation for the fused MLA decode kernel."""

import torch


def _pack_nz(weight: torch.Tensor) -> torch.Tensor:
    if weight.device.type == "cpu":
        return weight

    import torch_npu

    return torch_npu.npu_format_cast(weight, torch_npu.Format.FRACTAL_NZ)


def _reorder_rope_axis(
    tensor: torch.Tensor,
    rope_dim: int,
    dim: int,
) -> torch.Tensor:
    """Move even RoPE channels before odd channels on one tensor axis."""
    axis_size = tensor.shape[dim]
    rope_start = axis_size - rope_dim
    prefix = tensor.narrow(dim, 0, rope_start)
    rope = tensor.narrow(dim, rope_start, rope_dim)
    even = rope.index_select(
        dim,
        torch.arange(0, rope_dim, 2, device=tensor.device),
    )
    odd = rope.index_select(
        dim,
        torch.arange(1, rope_dim, 2, device=tensor.device),
    )
    return torch.cat((prefix, even, odd), dim=dim).contiguous()


def prepare_mla_preprocess_v2_qkv(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare fused KV-A/Q-A tensors for ``MlaPreprocessV2``."""
    kv_dim = kv_lora_rank + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        kv = _reorder_rope_axis(tensor.narrow(0, 0, kv_dim), qk_rope_head_dim, 0)
        q = tensor.narrow(0, kv_dim, tensor.shape[0] - kv_dim)
        return torch.cat((kv, q), dim=0).contiguous()

    return _pack_nz(_prepare(weight)), _prepare(descale), _prepare(bias)


def prepare_mla_preprocess_v2_q_b(
    weight: torch.Tensor,
    descale: torch.Tensor,
    bias: torch.Tensor,
    num_heads: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Prepare Q-B tensors for ``MlaPreprocessV2``."""
    head_dim = qk_nope_head_dim + qk_rope_head_dim

    def _prepare(tensor: torch.Tensor) -> torch.Tensor:
        shape = tensor.shape
        viewed = tensor.view(num_heads, head_dim, *shape[1:])
        reordered = _reorder_rope_axis(viewed, qk_rope_head_dim, 1)
        return reordered.reshape(shape).contiguous()

    return _pack_nz(_prepare(weight)), _prepare(descale), _prepare(bias)
