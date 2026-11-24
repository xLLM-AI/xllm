# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Torch implementations for partial rotary embedding and TorchNPU RoPE."""

from __future__ import annotations

import torch

from xllm_kernel.ops.positional_encoding import PARTIAL_ROTARY_EMBEDDING_CONTRACT
from xllm_kernel.platform import PlatformEnum, current_platform
from xllm_kernel.registry import register_kernel


def _partial_rotary_embedding(value: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    half_dim = cos_half.size(-1)
    first, second, passthrough = value.split((half_dim, half_dim, value.size(-1) - 2 * half_dim), dim=-1)
    cos = cos_half.unsqueeze(1)
    sin = sin_half.unsqueeze(1)
    return torch.cat((first * cos + (-second) * sin, second * cos + first * sin, passthrough), dim=-1)


for _platform, _modes in (
    (PlatformEnum.CPU, ("eager",)),
    (PlatformEnum.CUDA, ("eager", "cudagraph")),
    (PlatformEnum.NPU, ("eager", "aclgraph")),
):
    register_kernel(
        name=f"{_platform.value}.torch.partial_rotary_embedding",
        platform=_platform,
        execution_modes=_modes,
        input_domain="[T,H,D] with [T,R] cosine and sine tables; 0 < 2*R <= D.",
        solution="torch",
        binding="torch",
        contract=PARTIAL_ROTARY_EMBEDDING_CONTRACT,
    )(_partial_rotary_embedding)


if current_platform.is_npu():
    import torch_npu

    from xllm_kernel.ops.positional_encoding import TORCH_NPU_ROTARY_OPERATIONS

    def apply_rotary_embedding(
        q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor
    ) -> None:
        cos_sin = cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)
        rotary_dim = cos.size(-1)
        torch_npu.npu_apply_rotary_pos_emb(
            q.view(1, q.size(0), -1, rotary_dim),
            k.view(1, k.size(0), -1, rotary_dim),
            cos.contiguous().view(1, -1, 1, rotary_dim),
            sin.contiguous().view(1, -1, 1, rotary_dim),
            layout="BSND",
        )

    def apply_qk_rotary(
        query: torch.Tensor,
        key: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotate [tokens, heads, dim] Q/K using [1, tokens, 1, dim] tables.

        Consumes Q/K: contiguous inputs are rotated in place; strided views are
        materialized contiguous first. Do not re-read the original Q/K storage.
        """
        head_dim = query.size(-1)
        if key.size(-1) != head_dim:
            raise ValueError(f"q head_dim {head_dim} does not match k head_dim {key.size(-1)}")
        if cosine.size(-1) != head_dim or sine.size(-1) != head_dim:
            raise ValueError(
                f"cos/sin table width ({cosine.size(-1)}, {sine.size(-1)}) does not match head_dim {head_dim}"
            )
        query_out, key_out = torch_npu.npu_apply_rotary_pos_emb(
            query.contiguous().unsqueeze(0),
            key.contiguous().unsqueeze(0),
            cosine,
            sine,
            layout="BSND",
        )
        return query_out.squeeze(0), key_out.squeeze(0)

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
        return torch_npu.npu_rotary_mul(value.unsqueeze(0).contiguous(), cos_full, sin_full).squeeze(0)

    for operation, function in zip(
        TORCH_NPU_ROTARY_OPERATIONS, (apply_rotary_embedding, apply_qk_rotary, mrope, vision_rotary_mul), strict=True
    ):
        register_kernel(
            name=operation.implementations[0],
            platform=PlatformEnum.NPU,
            execution_modes=("eager", "aclgraph"),
            input_domain="TorchNPU rotary input and coefficient tensors.",
            solution="torch",
            binding="torch_npu",
            contract=operation.contract,
        )(function)
