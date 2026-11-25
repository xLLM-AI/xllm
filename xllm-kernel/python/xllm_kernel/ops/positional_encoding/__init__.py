# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Positional encoding contracts and startup-bound APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.ops.positional_encoding._tilelang._mrope_geometry import (
    ALL_HEAD_CONFIGS,
    _validate_mrope_section,
    build_mrope_gather_pattern_merged,
)
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

PARTIAL_ROTARY_EMBEDDING_CONTRACT = OpContract("partial_rotary_embedding", 1, (), False, "torch_half_rope")
PARTIAL_ROTARY_EMBEDDING_OPERATION = Operation(
    PARTIAL_ROTARY_EMBEDDING_CONTRACT,
    tuple(
        BackendModule(
            module="xllm_kernel.ops.positional_encoding.torch",
            platform=platform,
            requires=("torch",),
            implementations=(f"{platform.value}.torch.partial_rotary_embedding",),
        )
        for platform in (PlatformEnum.CPU, PlatformEnum.CUDA, PlatformEnum.NPU)
    ),
)
INPLACE_PARTIAL_ROTARY_MUL_CONTRACT = OpContract(
    "inplace_partial_rotary_mul", 1, ("value",), False, "interleaved_partial_rope"
)
INPLACE_PARTIAL_ROTARY_MUL_OPERATION = Operation(
    INPLACE_PARTIAL_ROTARY_MUL_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.positional_encoding.ascendc",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.ascendc.inplace_partial_rotary_mul",),
        ),
    ),
)
FUSED_QK_NORM_ROPE_CONTRACT = OpContract("fused_qk_norm_rope", 1, (), False, "fp32_rmsnorm_bf16_rope")
FUSED_QK_NORM_ROPE_OPERATION = Operation(
    FUSED_QK_NORM_ROPE_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.positional_encoding.triton",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "triton"),
            implementations=("npu.triton.fused_qk_norm_rope",),
        ),
    ),
)
SPLIT_QKV_RMSNORM_MROPE_CONTRACT = OpContract("split_qkv_rmsnorm_mrope", 1, (), False, "fp32_rmsnorm_partial_mrope")
SPLIT_QKV_RMSNORM_MROPE_OPERATION = Operation(
    SPLIT_QKV_RMSNORM_MROPE_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.positional_encoding.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch_npu", "tilelang", "triton"),
            implementations=("npu.tilelang.split_qkv_rmsnorm_mrope",),
        ),
    ),
)
TORCH_NPU_ROTARY_OPERATIONS = tuple(
    Operation(
        OpContract(name, 1, mutates, aliases, name),
        (
            BackendModule(
                module="xllm_kernel.ops.positional_encoding.torch",
                platform=PlatformEnum.NPU,
                requires=("torch_npu",),
                implementations=(f"npu.torch.{name}",),
            ),
        ),
    )
    for name, mutates, aliases in (
        ("apply_rotary_embedding", ("q", "k"), False),
        ("apply_qk_rotary", ("query", "key"), True),
        ("mrope", (), False),
        ("vision_rotary_mul", (), False),
    )
)
_APPLY_ROTARY, _APPLY_QK_ROTARY, _MROPE, _VISION_ROTARY = TORCH_NPU_ROTARY_OPERATIONS
OPERATIONS = (
    PARTIAL_ROTARY_EMBEDDING_OPERATION,
    INPLACE_PARTIAL_ROTARY_MUL_OPERATION,
    FUSED_QK_NORM_ROPE_OPERATION,
    SPLIT_QKV_RMSNORM_MROPE_OPERATION,
    *TORCH_NPU_ROTARY_OPERATIONS,
)


def partial_rotary_embedding(value: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor) -> torch.Tensor:
    """Rotate the leading 2*R channels of [T,H,D] with half-style [T,R] tables.

    Cosine and sine are already gathered for the tokens and share the value's
    device. The suffix passes through unchanged. The result is fresh storage
    and inputs are read only. Works in eager and Python graph execution.
    """
    return PARTIAL_ROTARY_EMBEDDING_OPERATION.get_prepared().function(value, cos_half, sin_half)


def inplace_partial_rotary_mul(
    value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor, start: int, end: int
) -> None:
    """Apply interleaved RoPE to the half-open slice [start, end) in place.

    Value and coefficient tables are 4D NPU tensors of the same dtype.
    Coefficients span the slice width and broadcast over the leading axes.
    The rest of value and both coefficient tables are read only.
    """
    INPLACE_PARTIAL_ROTARY_MUL_OPERATION.get_prepared().function(value, cosine, sine, start, end)


def has_split_qkv_rmsnorm_mrope_specialization(num_q_heads: int, num_kv_heads: int, head_size: int) -> bool:
    """Whether the worker has the fused BF16 mRoPE implementation for these heads."""
    return (
        SPLIT_QKV_RMSNORM_MROPE_OPERATION.prepared is not None
        and head_size == 256
        and (num_q_heads, num_kv_heads) in ALL_HEAD_CONFIGS
    )


def prepare_split_qkv_rmsnorm_mrope(num_q_heads: int, num_kv_heads: int, head_size: int) -> None:
    """Prepare the head specialization before capturing a model graph."""
    if not has_split_qkv_rmsnorm_mrope_specialization(num_q_heads, num_kv_heads, head_size):
        raise ValueError("unsupported fused QKV mRoPE head configuration")
    from xllm_kernel.ops.positional_encoding.tilelang import prepare_mrope

    prepare_mrope(num_q_heads, num_kv_heads, head_size)


def build_split_qkv_rmsnorm_mrope_gather_pattern(
    rope_dim: int, mrope_section: list[int], is_interleaved: bool, device: torch.device
) -> torch.Tensor:
    """Build the persistent uint32 byte offsets for the packed cosine and sine tables."""
    if rope_dim <= 0 or rope_dim % 2:
        raise ValueError("rope_dim must be positive and even")
    sections = tuple(mrope_section)
    _validate_mrope_section(rope_dim=rope_dim, mrope_section=sections)
    return build_mrope_gather_pattern_merged(
        half_rope_dim=rope_dim // 2,
        rope_dim=rope_dim,
        mrope_section=sections,
        is_interleaved=is_interleaved,
        gather_pad_dim=((3 * rope_dim + 127) // 128) * 128,
        device=device,
    )


def split_qkv_rmsnorm_mrope(
    qkvg: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    gather_pattern: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split Q|G|K|V and return the normalized and rotated Q/K, V and gate."""
    if not has_split_qkv_rmsnorm_mrope_specialization(num_q_heads, num_kv_heads, head_size):
        raise ValueError("unsupported fused QKV mRoPE head configuration")
    return SPLIT_QKV_RMSNORM_MROPE_OPERATION.get_prepared().function(
        qkvg,
        q_weight,
        k_weight,
        cos_sin,
        gather_pattern,
        eps,
        num_q_heads,
        num_kv_heads,
        head_size,
    )


def apply_rotary_embedding(
    q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor, positions: torch.Tensor
) -> None:
    """Apply the legacy cache-indexed rotary operation in place to Q and K."""
    _APPLY_ROTARY.get_prepared().function(q, k, cos_sin_cache, positions)


def apply_qk_rotary(
    query: torch.Tensor, key: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate [tokens, heads, dim] Q/K with already gathered cosine/sine."""
    return _APPLY_QK_ROTARY.get_prepared().function(query, key, cosine, sine)


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
    """Apply multimodal RoPE with per-axis positions and section widths."""
    return _MROPE.get_prepared().function(
        positions,
        q,
        k,
        cos_sin_cache,
        head_dim,
        mrope_section=mrope_section,
        rotary_mode=rotary_mode,
        cache_mode=cache_mode,
    )


def vision_rotary_mul(value: torch.Tensor, cos_full: torch.Tensor, sin_full: torch.Tensor) -> torch.Tensor:
    """Apply half-style vision RoPE to token-major heads."""
    return _VISION_ROTARY.get_prepared().function(value, cos_full, sin_full)


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
        Fresh query, key and value tensors of shape ``[num_tokens, heads * head_dim]``.
    """
    del num_heads_v, cos, sin, interleaved

    return FUSED_QK_NORM_ROPE_OPERATION.get_prepared().function(
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
