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

"""Op dispatch layer for the Python model executor.

Each op is a direct binding to the C++ ``torch.ops.xllm_ops.*`` kernel (routed
by PyTorch DispatchKey per device).  FakeTensor / disallow_in_graph semantics
are registered as import-time side effects in the submodules.

Attention kernels (batch_prefill/batch_decode) are provided by the flashinfer
Python package directly via ``layers/attention.py``, not through this module.
"""

from xllm.python.ops.compute import (
    dynamic_quant,
    fused_add_rms_norm,
    fused_qk_norm_rope,
    lightning_indexer,
    quant_matmul,
    quantize_per_tensor,
    rms_norm,
    scatter_nd_update,
    silu_and_mul,
    sparse_flash_attention,
)
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

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "fused_qk_norm_rope",
    "quant_matmul",
    "quantize_per_tensor",
    "dynamic_quant",
    "lightning_indexer",
    "scatter_nd_update",
    "sparse_flash_attention",
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "all_reduce_",
    "all_gather",
    "init_tp_group",
    "tp_rank",
]
