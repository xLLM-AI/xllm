# Copyright 2025-2026 The xLLM Authors.
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

"""Python-defined model graphs executed by xLLM's C++ worker.

Importing this package is intentionally runtime-neutral. The embedded C++
bootstrap calls :func:`initialize_runtime` after registering native operators
and before importing a Python model module. That call selects the platform
kernel package and publishes it as :mod:`xllm.python.kernels`.

Keeping ordinary package import free of platform and native-operator side
effects lets build tools import leaf DSL modules such as
``xllm.python.kernels_npu.tilelang`` before the xLLM binary exists.
"""

from __future__ import annotations

import importlib
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any

_runtime_kernels: ModuleType | None = None


def _load_kernel_package() -> ModuleType:
    source_root = Path(__file__).resolve().parents[2]
    if (source_root / "setup.py").is_file() and (source_root / "CMakeLists.txt").is_file():
        kernel_path = str(source_root / "xllm-kernel" / "python")
        if kernel_path not in sys.path:
            sys.path.insert(0, kernel_path)
    return importlib.import_module("xllm_kernel")


def initialize_runtime() -> None:
    """Initialize and publish the active platform's Python kernel package."""

    global _runtime_kernels
    if _runtime_kernels is not None:
        return

    from xllm.python.platform import current_platform

    if current_platform.is_cuda():
        _load_kernel_package().initialize(device="cuda", operations=("partial_rotary_embedding",))
        backend_name = "xllm.python.kernels_cuda"
    elif current_platform.is_npu():
        xllm_kernel = _load_kernel_package()

        from xllm_kernel.ops.attention.gdn.mega import has_mega_gdn_mtp_decode_varlen
        from xllm_kernel.ops.attention.mla import has_mla_preprocess_v2
        from xllm_kernel.ops.collectives import HCCL_OPERATIONS, has_matmul_reduce_scatter
        from xllm_kernel.ops.gemm import has_grouped_matmul_out
        from xllm_kernel.ops.moe import (
            FUSED_MOE_GMM1_OPERATIONS,
            MOE_DISTRIBUTE_OPERATIONS,
            has_fused_moe_gmm1,
            has_mega_moe,
            has_moe_distribute,
        )

        from scripts.logger import logger

        prepared = xllm_kernel.initialize(
            device="npu",
            operations=(
                "rms_norm",
                "gemma_rms_norm",
                "fused_add_rms_norm",
                "silu_and_mul",
                "fused_add_gemma_rms_norm",
                "rms_norm_gated",
                "rms_norm_dynamic_quant",
                "hc_pre",
                "hc_post",
                "l2_norm",
                "matmul_ein_sum",
                "matmul_ein_sum_out",
                "compressor",
                "sparse_attn_sharedkv",
                "sparse_attn_sharedkv_metadata",
                "sparse_flash_attention_lse",
                "sparse_flash_attention_lse_out",
                "lightning_indexer",
                "lightning_indexer_out",
                "quant_lightning_indexer",
                "quant_lightning_indexer_metadata",
                "scatter_nd_update",
                "quant_matmul",
                "quant_matmul_out",
                "quantize_per_tensor",
                "dynamic_quant",
                "grouped_matmul",
                *(("grouped_matmul_out",) if has_grouped_matmul_out() else ()),
                *(("mla_preprocess_v2",) if has_mla_preprocess_v2() else ()),
                "dequant_swiglu_quant",
                "moe_gating_top_k_hash",
                "moe_gating_top_k_softmax",
                "moe_gating_top_k",
                "moe_init_routing_v2",
                "moe_token_unpermute",
                "moe_finalize_routing",
                *(MOE_DISTRIBUTE_OPERATIONS if has_moe_distribute() else ()),
                *(("mega_moe",) if has_mega_moe() else ()),
                *HCCL_OPERATIONS,
                *(("matmul_reduce_scatter",) if has_matmul_reduce_scatter() else ()),
                *(FUSED_MOE_GMM1_OPERATIONS if has_fused_moe_gmm1() else ()),
                "fused_recurrent_gated_delta_rule",
                "fused_sigmoid_gating_delta_rule_update",
                "fused_qkvzba_split_reshape_cat",
                "mega_gdn_prefill",
                "mega_gdn_decode",
                "mega_gdn_mtp_decode",
                *(("mega_gdn_mtp_decode_varlen",) if has_mega_gdn_mtp_decode_varlen() else ()),
                "shared_linear_state_indices_out",
                "causal_conv1d",
                "causal_conv1d_prefill",
                "causal_conv1d_qkv_prefill",
                "chunk_gated_delta_rule",
                "causal_conv1d_update_v2",
                "partial_rotary_embedding",
                "inplace_partial_rotary_mul",
                "apply_rotary_embedding",
                "apply_qk_rotary",
                "fused_qk_norm_rope",
                "mrope",
                "vision_rotary_mul",
                "reshape_paged_cache",
                "update_decode_graph_metadata",
                *(
                    (
                        "shared_full_indices_out",
                        "sfa_dcp_remap_out",
                        "split_qkv_rmsnorm_mrope",
                        "fused_gdn_gating",
                        "causal_conv1d_decode",
                    )
                    if find_spec("tilelang") is not None
                    else ()
                ),
                "fused_attention",
            ),
        )
        for operation, kernel in prepared.items():
            logger.info("Python kernel %s: %s (generation=%s)", operation, kernel.spec.name, kernel.generation)
        backend_name = "xllm.python.kernels_npu"
    else:
        device_type = current_platform.device_type()
        raise ImportError(
            f"no Python kernel package for platform '{device_type}'; add "
            f"xllm/python/kernels_{device_type}/ with the APIs required by "
            "that platform's supported models"
        )

    backend = importlib.import_module(backend_name)
    backend_initializer = getattr(backend, "_initialize_runtime", None)
    if backend_initializer is not None:
        backend_initializer()

    globals()["kernels"] = backend
    sys.modules[f"{__name__}.kernels"] = backend
    _runtime_kernels = backend


def __getattr__(name: str) -> Any:
    if name == "kernels":
        raise RuntimeError(
            "xllm.python runtime is not initialized; call "
            "xllm.python.initialize_runtime() after registering native "
            "torch operators and before importing model or layer modules"
        )
    if name in {"get_model_class", "register_model"}:
        registry = importlib.import_module("xllm.python.registry")
        value = getattr(registry, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["get_model_class", "initialize_runtime", "kernels", "register_model"]
