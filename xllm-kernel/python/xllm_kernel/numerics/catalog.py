# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
"""Explicit discovery of operation-owned suites; loading must stay SDK-free."""

from importlib import import_module

from xllm_kernel.numerics.suite import OperationSuite

# Only this catalog needs a new entry when adding an operation's tool support.
_BUILTINS = (
    "xllm_kernel.numerics.all_to_all_single",
    "xllm_kernel.numerics.quantize_per_tensor",
    "xllm_kernel.numerics.einsum_out",
    "xllm_kernel.numerics.reduce_scatter",
    "xllm_kernel.numerics.all_gather",
    "xllm_kernel.numerics.all_reduce",
    "xllm_kernel.numerics.sparse_flash_attention_out",
    "xllm_kernel.numerics.sparse_flash_attention",
    "xllm_kernel.numerics.lightning_indexer_out",
    "xllm_kernel.numerics.lightning_indexer",
    "xllm_kernel.numerics.sfa_dcp_remap",
    "xllm_kernel.numerics.reshape_paged_cache",
    "xllm_kernel.numerics.scatter_nd_update",
    "xllm_kernel.numerics.inplace_partial_rotary_mul",
    "xllm_kernel.numerics.interleaved_rotary_embedding",
    "xllm_kernel.numerics.moe_finalize_routing",
    "xllm_kernel.numerics.moe_token_unpermute",
    "xllm_kernel.numerics.moe_gmm1",
    "xllm_kernel.numerics.grouped_matmul_out",
    "xllm_kernel.numerics.grouped_matmul",
    "xllm_kernel.numerics.moe_token_dispatch",
    "xllm_kernel.numerics.moe_gate_routing",
    "xllm_kernel.numerics.dequant_swiglu_quant",
    "xllm_kernel.numerics.quant_matmul_out",
    "xllm_kernel.numerics.quant_matmul",
    "xllm_kernel.numerics.fused_add_rms_norm_dynamic_quant",
    "xllm_kernel.numerics.rms_norm_dynamic_quant",
    "xllm_kernel.numerics.dynamic_quant",
    "xllm_kernel.numerics.router_matmul",
    "xllm_kernel.numerics.einsum",
    "xllm_kernel.numerics.linear",
    "xllm_kernel.numerics.fused_add_rms_norm",
    "xllm_kernel.numerics.layer_norm",
    "xllm_kernel.numerics.embedding",
    "xllm_kernel.numerics.silu_and_mul",
    "xllm_kernel.numerics.add",
    "xllm_kernel.numerics.rms_norm",
)


def load_suites() -> tuple[OperationSuite, ...]:
    suites = tuple(import_module(module).SUITE for module in _BUILTINS)
    operations = [suite.operation for suite in suites]
    implementations = [name for suite in suites for name in suite.implementations]
    if len(set(operations)) != len(operations) or len(set(implementations)) != len(implementations):
        raise ValueError("operation suites must have unique operation and implementation names")
    return suites
