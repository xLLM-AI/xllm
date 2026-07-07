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
"""

from .compute import (
    fused_add_rms_norm,
    fused_qk_norm_rope,
    rms_norm,
    silu_and_mul,
)
from .attention import (
    batch_chunked_prefill,
    batch_decode,
    batch_prefill,
    reshape_paged_cache,
    update_chunked_prefill_plan,
    update_decode_plan,
    update_prefill_plan,
)
from .collectives import (
    all_gather,
    all_reduce,
    set_tp_group,
)

__all__ = [
    "rms_norm",
    "fused_add_rms_norm",
    "silu_and_mul",
    "fused_qk_norm_rope",
    "reshape_paged_cache",
    "batch_prefill",
    "batch_decode",
    "batch_chunked_prefill",
    "update_prefill_plan",
    "update_decode_plan",
    "update_chunked_prefill_plan",
    "all_reduce",
    "all_gather",
    "set_tp_group",
]
