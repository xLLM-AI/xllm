# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hardware-neutral op interfaces for the Python model executor.

Public op modules own device-independent contracts and lazily select
platform-specific implementations when needed. C++ ``torch.ops.xllm_ops``
bindings remain registered by the modules that own those interfaces.
"""

from xllm.python.ops.attention import (
    reshape_paged_cache,
    update_decode_graph_metadata,
)
from xllm.python.ops.collectives import (
    all_gather,
    all_reduce_,
    init_tp_group,
    tp_rank,
)
from xllm.python.ops.compute import (
    fused_add_rms_norm,
    fused_qk_norm_rope,
    rms_norm,
    silu_and_mul,
)
from xllm.python.ops.linear import prepare_row_parallel_weight
from xllm.python.ops.moe import (
    cutlass_fused_moe,
    fused_moe,
    grouped_moe,
    moe_fused_topk,
    prepare_grouped_moe_weights,
    supports_cutlass_moe,
)
from xllm.python.ops.quantization import (
    dynamic_quant,
    quant_matmul,
    quantize_per_tensor,
)
from xllm.python.ops.rotary_embedding import interleaved_rotary_embedding
from xllm.python.ops.sparse_attention import (
    lightning_indexer,
    scatter_nd_update,
    sparse_flash_attention,
)

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "fused_qk_norm_rope",
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "all_reduce_",
    "all_gather",
    "init_tp_group",
    "tp_rank",
    "prepare_row_parallel_weight",
    "prepare_grouped_moe_weights",
    "supports_cutlass_moe",
    "moe_fused_topk",
    "cutlass_fused_moe",
    "fused_moe",
    "grouped_moe",
    "quant_matmul",
    "quantize_per_tensor",
    "dynamic_quant",
    "interleaved_rotary_embedding",
    "lightning_indexer",
    "scatter_nd_update",
    "sparse_flash_attention",
]
