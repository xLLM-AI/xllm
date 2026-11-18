# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free public operator APIs."""

from xllm_kernel.ops.activation import silu_and_mul
from xllm_kernel.ops.attention.gdn import (
    fused_qkvzba_split_reshape_cat,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from xllm_kernel.ops.attention.kda import causal_conv1d_update_v2
from xllm_kernel.ops.gemm import matmul_ein_sum, matmul_ein_sum_out
from xllm_kernel.ops.kv_cache import reshape_paged_cache
from xllm_kernel.ops.normalization import fused_add_rms_norm, l2_norm, layer_norm, rms_norm
from xllm_kernel.ops.positional_encoding import partial_rotary_embedding

__all__ = [
    "causal_conv1d_update_v2",
    "fused_add_rms_norm",
    "fused_qkvzba_split_reshape_cat",
    "fused_recurrent_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "l2_norm",
    "layer_norm",
    "rms_norm",
    "matmul_ein_sum",
    "matmul_ein_sum_out",
    "partial_rotary_embedding",
    "reshape_paged_cache",
    "silu_and_mul",
]
