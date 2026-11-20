# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""SDK-free public operator APIs."""

from xllm_kernel.ops.activation import silu_and_mul
from xllm_kernel.ops.attention.fia import get_fused_attention
from xllm_kernel.ops.attention.gdn import (
    causal_conv1d_qkv_prefill,
    chunk_gated_delta_rule,
    fused_qkvzba_split_reshape_cat,
    fused_recurrent_gated_delta_rule,
    fused_sigmoid_gating_delta_rule_update,
)
from xllm_kernel.ops.attention.gdn.mega import (
    mega_gdn_decode,
    mega_gdn_mtp_decode,
    mega_gdn_mtp_decode_varlen,
    mega_gdn_prefill,
    shared_linear_state_indices_out,
)
from xllm_kernel.ops.attention.kda import causal_conv1d, causal_conv1d_update_v2
from xllm_kernel.ops.attention.mla import mla_preprocess_v2
from xllm_kernel.ops.gemm import matmul_ein_sum, matmul_ein_sum_out, quant_matmul_out
from xllm_kernel.ops.hyper_connection import hc_post, hc_pre
from xllm_kernel.ops.kv_cache import reshape_paged_cache
from xllm_kernel.ops.moe import (
    dequant_swiglu_quant,
    moe_gating_top_k_hash,
    moe_grouped_matmul_swiglu_quant,
    moe_init_routing_v3,
)
from xllm_kernel.ops.normalization import (
    fused_add_gemma_rms_norm,
    fused_add_rms_norm,
    l2_norm,
    layer_norm,
    rms_norm,
    rms_norm_dynamic_quant,
    rms_norm_gated,
)
from xllm_kernel.ops.positional_encoding import inplace_partial_rotary_mul, partial_rotary_embedding

__all__ = [
    "causal_conv1d",
    "causal_conv1d_qkv_prefill",
    "causal_conv1d_update_v2",
    "chunk_gated_delta_rule",
    "dequant_swiglu_quant",
    "fused_add_gemma_rms_norm",
    "fused_add_rms_norm",
    "fused_qkvzba_split_reshape_cat",
    "fused_recurrent_gated_delta_rule",
    "fused_sigmoid_gating_delta_rule_update",
    "hc_post",
    "hc_pre",
    "inplace_partial_rotary_mul",
    "l2_norm",
    "layer_norm",
    "rms_norm",
    "rms_norm_dynamic_quant",
    "rms_norm_gated",
    "matmul_ein_sum",
    "matmul_ein_sum_out",
    "mega_gdn_decode",
    "mega_gdn_mtp_decode",
    "mega_gdn_mtp_decode_varlen",
    "mega_gdn_prefill",
    "mla_preprocess_v2",
    "moe_gating_top_k_hash",
    "moe_grouped_matmul_swiglu_quant",
    "moe_init_routing_v3",
    "quant_matmul_out",
    "partial_rotary_embedding",
    "reshape_paged_cache",
    "shared_linear_state_indices_out",
    "silu_and_mul",
]
