# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free public operator APIs."""

from xllm_kernel.ops.activation import silu_and_mul
from xllm_kernel.ops.normalization import (
    build_split_qkv_rmsnorm_mrope_gather_pattern,
    fused_add_gemma_rms_norm,
    fused_add_rms_norm,
    fused_add_rms_norm_dynamic_quant,
    fused_eh_norm,
    fused_qk_norm_rope,
    gemma_rms_norm,
    has_split_qkv_rmsnorm_mrope_specialization,
    l2_norm,
    layer_norm,
    rms_norm,
    rms_norm_dynamic_quant,
    rms_norm_gated,
    rms_norm_sigmoid_gated,
    split_qkv_rmsnorm_mrope,
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
    "rms_norm",
    "rms_norm_dynamic_quant",
    "rms_norm_gated",
    "rms_norm_sigmoid_gated",
    "silu_and_mul",
    "split_qkv_rmsnorm_mrope",
]
