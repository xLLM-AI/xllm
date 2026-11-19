# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Fused MLA decode preprocessing contract and startup-bound API."""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

from xllm_kernel.catalog import BackendModule, Operation
from xllm_kernel.platform import PlatformEnum
from xllm_kernel.registry import OpContract

if TYPE_CHECKING:
    import torch

MLA_PREPROCESS_V2_CONTRACT = OpContract(
    "mla_preprocess_v2", 1, ("kv_cache", "rope_cache"), False, "fused_quantized_mla"
)
MLA_PREPROCESS_V2_OPERATION = Operation(
    MLA_PREPROCESS_V2_CONTRACT,
    (
        BackendModule(
            module="xllm_kernel.ops.attention.mla.ascendc",
            platform=PlatformEnum.NPU,
            requires=("torch_npu",),
            implementations=("npu.ascendc.mla_preprocess_v2",),
        ),
    ),
)
OPERATIONS = (MLA_PREPROCESS_V2_OPERATION,)


@cache
def has_mla_preprocess_v2() -> bool:
    """Return whether the installed device package exports the fused ACLNN API."""
    import torch

    from xllm_kernel import _ascendc_opp  # noqa: F401

    return bool(torch.ops.xllm_kernel.has_mla_preprocess_v2())


def mla_preprocess_v2(
    hidden: torch.Tensor,
    input_norm_weight: torch.Tensor,
    input_norm_bias: torch.Tensor,
    qkv_input_scale: torch.Tensor,
    qkv_input_offset: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_deq_scale: torch.Tensor,
    qkv_quant_bias: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_norm_bias: torch.Tensor,
    q_b_input_scale: torch.Tensor,
    q_b_input_offset: torch.Tensor,
    q_b_weight: torch.Tensor,
    q_b_deq_scale: torch.Tensor,
    q_b_quant_bias: torch.Tensor,
    kv_norm_weight: torch.Tensor,
    rope_cos: torch.Tensor,
    rope_sin: torch.Tensor,
    w_uk: torch.Tensor,
    kv_cache: torch.Tensor,
    rope_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    q_lora_rank: int,
    qk_rope_head_dim: int,
    norm_epsilon: float,
    cache_mode: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return Q-down, Q-latent and Q-RoPE and update the selected KV cache slots.

    The weights use the MLAPO v2 packed layouts. The cache mode selects ND or
    FRACTAL_NZ storage. The caller owns both caches throughout graph replay.
    """
    return MLA_PREPROCESS_V2_OPERATION.get_prepared().function(
        hidden,
        input_norm_weight,
        input_norm_bias,
        qkv_input_scale,
        qkv_input_offset,
        qkv_weight,
        qkv_deq_scale,
        qkv_quant_bias,
        q_norm_weight,
        q_norm_bias,
        q_b_input_scale,
        q_b_input_offset,
        q_b_weight,
        q_b_deq_scale,
        q_b_quant_bias,
        kv_norm_weight,
        rope_cos,
        rope_sin,
        w_uk,
        kv_cache,
        rope_cache,
        slot_mapping,
        q_lora_rank,
        qk_rope_head_dim,
        norm_epsilon,
        cache_mode,
    )
