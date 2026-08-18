# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""NPU kernel semantic API.

Ordinary package import is build-safe and does not inspect native operators.
The embedded runtime calls :func:`_initialize_runtime` through
``xllm.python.initialize_runtime`` after ``torch.ops.xllm_ops`` is registered.
Leaf DSL packages under ``tilelang`` and ``triton`` therefore remain directly
importable by build tooling without initializing this semantic API.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "activation": ("silu_and_mul",),
    "attention": (
        "reshape_paged_cache",
        "update_decode_graph_metadata",
        "vision_fusion_attention",
    ),
    "causal_conv1d": ("causal_conv1d_decode", "causal_conv1d_prefill"),
    "gated_delta_net": (
        "chunk_gated_delta_rule",
        "fused_gdn_prefill_post_conv",
        "fused_recurrent_gated_delta_rule_packed_decode",
        "resolve_gdn_prefill_backend",
    ),
    "linear": ("prepare_row_parallel_weight",),
    "moe": (
        "cutlass_fused_moe",
        "fused_moe",
        "grouped_moe",
        "moe_fused_topk",
        "prepare_grouped_moe_weights",
        "supports_cutlass_moe",
    ),
    "normalization": (
        "fused_add_rms_norm",
        "l2_norm",
        "rms_norm",
        "rms_norm_gated",
    ),
    "quantization": ("dynamic_quant", "quant_matmul", "quantize_per_tensor"),
    "rotary_embedding": (
        "fused_qk_norm_rope",
        "interleaved_rotary_embedding",
        "mrope",
        "vision_rotary_mul",
    ),
    "sparse_attention": (
        "lightning_indexer",
        "lightning_indexer_out",
        "scatter_nd_update",
        "sparse_flash_attention",
        "sparse_flash_attention_out",
    ),
}

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "l2_norm",
    "rms_norm_gated",
    "silu_and_mul",
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "vision_fusion_attention",
    "fused_qk_norm_rope",
    "interleaved_rotary_embedding",
    "mrope",
    "vision_rotary_mul",
    "moe_fused_topk",
    "cutlass_fused_moe",
    "fused_moe",
    "grouped_moe",
    "prepare_grouped_moe_weights",
    "supports_cutlass_moe",
    "prepare_row_parallel_weight",
    "quant_matmul",
    "quantize_per_tensor",
    "dynamic_quant",
    "lightning_indexer",
    "lightning_indexer_out",
    "scatter_nd_update",
    "sparse_flash_attention",
    "sparse_flash_attention_out",
    "causal_conv1d_prefill",
    "causal_conv1d_decode",
    "resolve_gdn_prefill_backend",
    "fused_gdn_prefill_post_conv",
    "fused_recurrent_gated_delta_rule_packed_decode",
    "chunk_gated_delta_rule",
]
_runtime_initialized = False


def _initialize_runtime() -> None:
    """Load native-op bindings and publish the NPU semantic API once."""

    global _runtime_initialized
    if _runtime_initialized:
        return

    importlib.import_module(f"{__name__}._custom_op")
    exported: dict[str, Any] = {}
    for module_name, names in _EXPORTS.items():
        module = importlib.import_module(f"{__name__}.{module_name}")
        exported.update((name, getattr(module, name)) for name in names)

    globals().update(exported)
    _runtime_initialized = True


def __getattr__(name: str) -> Any:
    if name in __all__:
        raise RuntimeError(
            "xllm.python.kernels_npu runtime is not initialized; call "
            "xllm.python.initialize_runtime() after registering native "
            "torch operators"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
