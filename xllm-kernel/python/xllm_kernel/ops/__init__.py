# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free public operator APIs."""

from xllm_kernel.ops.activation import silu_and_mul
from xllm_kernel.ops.gemm import einsum, einsum_out
from xllm_kernel.ops.normalization import fused_add_rms_norm, layer_norm, rms_norm

__all__ = [
    "einsum",
    "einsum_out",
    "fused_add_rms_norm",
    "layer_norm",
    "rms_norm",
    "silu_and_mul",
]
