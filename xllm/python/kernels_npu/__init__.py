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

"""NPU kernels.

``xllm/python/__init__.py`` binds this package as ``xllm.python.kernels`` when
the active platform is NPU, so layers and models import one fixed path and
carry no hardware branch. Its peers -- ``kernels_cuda`` and any package added
for new hardware -- are bound the same way on their own platform. Exactly one of
them is imported in a process; they share no code and never import each other.
``setup.py`` ships only the package matching ``--device``.

Launchers live under ``triton/``; the modules here bind the public NPU kernel
API declared in ``__all__``. Peer packages own their APIs independently and
need not export the same names. Existing unsupported stubs remain explicit NPU
failure paths, but they are not a cross-platform export contract.
"""

from __future__ import annotations

# FakeTensor implementations of the C++ operators. Imported first so that a
# graph capture reaching any kernel below finds a registered fake.
from . import _custom_op  # noqa: F401
from .activation import silu_and_mul
from .attention import (
    reshape_paged_cache,
    update_decode_graph_metadata,
    vision_fusion_attention,
)
from .causal_conv1d import (
    causal_conv1d_decode,
    causal_conv1d_prefill,
)
from .gated_delta_net import (
    chunk_gated_delta_rule,
    fused_gdn_prefill_post_conv,
    fused_recurrent_gated_delta_rule_packed_decode,
    resolve_gdn_prefill_backend,
)
from .linear import prepare_row_parallel_weight
from .moe import (
    cutlass_fused_moe,
    fused_moe,
    grouped_moe,
    moe_fused_topk,
    prepare_grouped_moe_weights,
    supports_cutlass_moe,
)
from .normalization import (
    fused_add_rms_norm,
    l2_norm,
    rms_norm,
    rms_norm_gated,
)
from .quantization import (
    dynamic_quant,
    quant_matmul,
    quantize_per_tensor,
)
from .rotary_embedding import (
    fused_qk_norm_rope,
    interleaved_rotary_embedding,
    mrope,
    vision_rotary_mul,
)
from .sparse_attention import (
    lightning_indexer,
    lightning_indexer_out,
    scatter_nd_update,
    sparse_flash_attention,
    sparse_flash_attention_out,
)

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
