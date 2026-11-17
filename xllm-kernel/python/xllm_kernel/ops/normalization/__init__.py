# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free normalization contracts and startup-bound public APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract, PreparedKernel

TRITON_JIT_REQUIRES = (
    "torch_npu",
    "triton",
    "triton.language.extra.cann",
    "triton.language.extra.cann.extension",
)

FUSED_ADD_RMS_NORM_CONTRACT = OpContract("fused_add_rms_norm", 1, (), False, "fp32_residual_norm_cast")
LAYER_NORM_CONTRACT = OpContract("layer_norm", 1, (), False, "fp32_norm_cast")
RMS_NORM_CONTRACT = OpContract("rms_norm", 1, (), False, "fp32_norm_cast")
GEMMA_RMS_NORM_CONTRACT = OpContract("gemma_rms_norm", 1, (), False, "fp32_gemma_norm_cast")
FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_CONTRACT = OpContract(
    "fused_add_rms_norm_dynamic_quant", 1, (), False, "int8_per_token_residual_norm"
)
FUSED_ADD_GEMMA_RMS_NORM_CONTRACT = OpContract(
    "fused_add_gemma_rms_norm", 1, (), False, "fp32_gemma_residual_norm_cast"
)
RMS_NORM_DYNAMIC_QUANT_CONTRACT = OpContract("rms_norm_dynamic_quant", 1, (), False, "int8_per_token_norm")
L2_NORM_CONTRACT = OpContract("l2_norm", 1, (), False, "fp32_l2_norm_cast")
RMS_NORM_GATED_CONTRACT = OpContract("rms_norm_gated", 1, (), False, "fp32_rms_silu_gate_cast")
RMS_NORM_SIGMOID_GATED_CONTRACT = OpContract("rms_norm_sigmoid_gated", 1, (), False, "fp32_rms_sigmoid_gate_cast")
FUSED_EH_NORM_CONTRACT = OpContract("fused_eh_norm", 1, (), False, "fp32_eh_norm_concat_cast")
FUSED_QK_NORM_ROPE_CONTRACT = OpContract("fused_qk_norm_rope", 1, (), False, "fp32_qk_norm_rope_cast")
SPLIT_QKV_RMSNORM_MROPE_CONTRACT = OpContract("split_qkv_rmsnorm_mrope", 1, (), False, "fp32_qk_gemma_norm_mrope_cast")
HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_CONTRACT = OpContract(
    "has_split_qkv_rmsnorm_mrope_specialization", 1, (), False, "specialization_query"
)
BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_CONTRACT = OpContract(
    "build_split_qkv_rmsnorm_mrope_gather_pattern", 1, (), False, "mrope_byte_offsets"
)

FUSED_ADD_RMS_NORM_OPERATION = Operation(
    FUSED_ADD_RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.fused_add_rms_norm",),
        ),
    ),
)
LAYER_NORM_OPERATION = Operation(
    LAYER_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.layer_norm",),
        ),
    ),
)
RMS_NORM_OPERATION = Operation(
    RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.rms_norm",),
        ),
    ),
)
GEMMA_RMS_NORM_OPERATION = Operation(
    GEMMA_RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch_npu",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.gemma_rms_norm",),
        ),
    ),
)
FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_OPERATION = Operation(
    FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.torch_npu",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.torch.fused_add_rms_norm_dynamic_quant",),
        ),
    ),
)
FUSED_ADD_GEMMA_RMS_NORM_OPERATION = Operation(
    FUSED_ADD_GEMMA_RMS_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.aclnn",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.aclnn.fused_add_gemma_rms_norm",),
        ),
    ),
)
RMS_NORM_DYNAMIC_QUANT_OPERATION = Operation(
    RMS_NORM_DYNAMIC_QUANT_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.aclnn",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.aclnn.rms_norm_dynamic_quant",),
        ),
    ),
)
L2_NORM_OPERATION = Operation(
    L2_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.triton",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.triton.l2_norm.aot",),
        ),
    ),
)
RMS_NORM_GATED_OPERATION = Operation(
    RMS_NORM_GATED_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.aclnn",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.aclnn.rms_norm_gated",),
        ),
    ),
)
RMS_NORM_SIGMOID_GATED_OPERATION = Operation(
    RMS_NORM_SIGMOID_GATED_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.triton",
            platform=PlatformEnum.NPU,
            requires=TRITON_JIT_REQUIRES,
            implementations=("npu.triton.rms_norm_sigmoid_gated",),
        ),
    ),
)
FUSED_EH_NORM_OPERATION = Operation(
    FUSED_EH_NORM_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.triton",
            platform=PlatformEnum.NPU,
            requires=TRITON_JIT_REQUIRES,
            implementations=("npu.triton.fused_eh_norm",),
        ),
    ),
)
FUSED_QK_NORM_ROPE_OPERATION = Operation(
    FUSED_QK_NORM_ROPE_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.triton",
            platform=PlatformEnum.NPU,
            requires=TRITON_JIT_REQUIRES,
            implementations=("npu.triton.fused_qk_norm_rope",),
        ),
    ),
)
SPLIT_QKV_RMSNORM_MROPE_OPERATION = Operation(
    SPLIT_QKV_RMSNORM_MROPE_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.tilelang.split_qkv_rmsnorm_mrope",),
        ),
    ),
)
HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_OPERATION = Operation(
    HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.tilelang.has_split_qkv_rmsnorm_mrope_specialization",),
        ),
    ),
)
BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_OPERATION = Operation(
    BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.normalization.tilelang",
            platform=PlatformEnum.NPU,
            requires=("torch", "xllm"),
            implementations=("npu.tilelang.build_split_qkv_rmsnorm_mrope_gather_pattern",),
        ),
    ),
)

OPERATIONS = (
    FUSED_ADD_RMS_NORM_OPERATION,
    LAYER_NORM_OPERATION,
    RMS_NORM_OPERATION,
    GEMMA_RMS_NORM_OPERATION,
    FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_OPERATION,
    FUSED_ADD_GEMMA_RMS_NORM_OPERATION,
    RMS_NORM_DYNAMIC_QUANT_OPERATION,
    L2_NORM_OPERATION,
    RMS_NORM_GATED_OPERATION,
    RMS_NORM_SIGMOID_GATED_OPERATION,
    FUSED_EH_NORM_OPERATION,
    FUSED_QK_NORM_ROPE_OPERATION,
    SPLIT_QKV_RMSNORM_MROPE_OPERATION,
    HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_OPERATION,
    BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_OPERATION,
)


def prepare_rms_norm() -> PreparedKernel:
    """Return the RMSNorm binding retained for the worker and graph lifetime."""
    return RMS_NORM_OPERATION.get_prepared()


def fused_add_rms_norm(
    value: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute the startup binding without selection or retry."""

    return FUSED_ADD_RMS_NORM_OPERATION.get_prepared().function(value, residual, weight, eps)


def layer_norm(value: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Execute the startup binding without selection or retry."""

    return LAYER_NORM_OPERATION.get_prepared().function(value, weight, bias, eps)


def rms_norm(input: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize without mutation or aliasing using the startup binding."""

    return RMS_NORM_OPERATION.get_prepared().function(input, weight, eps)


def gemma_rms_norm(
    value: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply Gemma RMSNorm with the checkpoint's additive gamma."""

    return GEMMA_RMS_NORM_OPERATION.get_prepared().function(value, weight, eps)


def fused_add_rms_norm_dynamic_quant(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Add residual, apply RMSNorm, and dynamically quantize the result."""

    return FUSED_ADD_RMS_NORM_DYNAMIC_QUANT_OPERATION.get_prepared().function(value, residual, weight, eps)


def fused_add_gemma_rms_norm(
    value: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Add the residual and apply Gemma RMSNorm on NPU."""

    return FUSED_ADD_GEMMA_RMS_NORM_OPERATION.get_prepared().function(value, residual, weight, eps)


def rms_norm_dynamic_quant(value: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RMSNorm and return int8 values with per-token scales on NPU."""

    return RMS_NORM_DYNAMIC_QUANT_OPERATION.get_prepared().function(value, weight, eps)


def l2_norm(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Normalize the last dimension of ``value`` to unit L2 norm.

    Args:
        value: Tensor whose last dimension is normalized.
        eps: Added to the squared norm before the reciprocal square root.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """

    return L2_NORM_OPERATION.get_prepared().function(value, eps)


def rms_norm_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``silu(gate)``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization (SiLU is applied internally).
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """

    return RMS_NORM_GATED_OPERATION.get_prepared().function(value, gate, weight, eps)


def rms_norm_sigmoid_gated(
    value: torch.Tensor,
    gate: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Apply RMSNorm to ``value`` and gate the result with ``sigmoid(gate)``.

    Args:
        value: Tensor to normalize.
        gate: Gate applied after normalization (sigmoid is applied internally).
        weight: RMSNorm weight over the last dimension.
        eps: RMSNorm epsilon.

    Returns:
        A tensor with the shape and dtype of ``value``.
    """

    return RMS_NORM_SIGMOID_GATED_OPERATION.get_prepared().function(value, gate, weight, eps)


def fused_eh_norm(
    embed: torch.Tensor,
    carried: torch.Tensor,
    enorm_weight: torch.Tensor,
    hnorm_weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Normalize embedding and carried states, then concatenate the results.

    Args:
        embed: Token embedding to normalize.
        carried: Carried hidden state to normalize.
        enorm_weight: RMSNorm weight for ``embed``.
        hnorm_weight: RMSNorm weight for ``carried``.
        eps: RMSNorm epsilon shared by both inputs.

    Returns:
        The normalized inputs concatenated along the final dimension.
    """

    return FUSED_EH_NORM_OPERATION.get_prepared().function(embed, carried, enorm_weight, hnorm_weight, eps)


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

    return FUSED_QK_NORM_ROPE_OPERATION.get_prepared().function(
        qkv,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        num_heads_v=num_heads_v,
        head_dim=head_dim,
        eps=eps,
        q_weight=q_weight,
        k_weight=k_weight,
        cos_sin_cache=cos_sin_cache,
        position_ids=position_ids,
        cos=cos,
        sin=sin,
        interleaved=interleaved,
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
    """Split Q/G/K/V and apply fused Q/K RMSNorm plus mRoPE."""

    return SPLIT_QKV_RMSNORM_MROPE_OPERATION.get_prepared().function(
        qkvg, q_weight, k_weight, cos_sin, gather_pattern, eps, num_q_heads, num_kv_heads, head_size
    )


def has_split_qkv_rmsnorm_mrope_specialization(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> bool:
    """Return whether the AOT fused QKV specialization is available."""

    return HAS_SPLIT_QKV_RMSNORM_MROPE_SPECIALIZATION_OPERATION.get_prepared().function(
        num_q_heads, num_kv_heads, head_size
    )


def build_split_qkv_rmsnorm_mrope_gather_pattern(
    rope_dim: int,
    mrope_section: list[int],
    is_interleaved: bool,
    device: torch.device,
) -> torch.Tensor:
    """Build the fused kernel's persistent mRoPE byte-offset table."""

    return BUILD_SPLIT_QKV_RMSNORM_MROPE_GATHER_PATTERN_OPERATION.get_prepared().function(
        rope_dim, mrope_section, is_interleaved, device
    )


__all__ = [
    "build_split_qkv_rmsnorm_mrope_gather_pattern",
    "fused_add_gemma_rms_norm",
    "fused_add_rms_norm",
    "fused_add_rms_norm_dynamic_quant",
    "fused_eh_norm",
    "fused_qk_norm_rope",
    "gemma_rms_norm",
    "has_split_qkv_rmsnorm_mrope_specialization",
    "l2_norm",
    "layer_norm",
    "prepare_rms_norm",
    "rms_norm",
    "rms_norm_dynamic_quant",
    "rms_norm_gated",
    "rms_norm_sigmoid_gated",
    "split_qkv_rmsnorm_mrope",
]
