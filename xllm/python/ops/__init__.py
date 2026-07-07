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

"""Op dispatch layer for the Python model graph.

Each op forwards to the fused kernel exposed under ``torch.ops.xllm_ops.*``.
Attention is no longer a single opaque op here — it lives in the
``PagedAttention`` layer (``python.layers.attention``) which calls
``kv_cache_write`` + ``paged_attention`` kernel ops directly with explicit
parameters.
"""

from .dispatch import (
    all_gather,
    all_reduce,
    fused_add_rms_norm,
    fused_qk_norm_rope,
    rms_norm,
    set_tp_group,
    silu_and_mul,
)

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "fused_qk_norm_rope",
    "all_reduce",
    "all_gather",
    "set_tp_group",
]
