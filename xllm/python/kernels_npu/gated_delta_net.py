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

"""NPU gated-delta-network kernel entry points.

Prefill uses the production MegaGdn operator. The remaining helpers preserve
the existing Python composition for paths not migrated to MegaGdn yet.
"""

from __future__ import annotations

from xllm_kernel.ops.attention.gdn import chunk_gated_delta_rule, fused_gdn_gating
from xllm_kernel.ops.attention.gdn.mega import (
    mega_gdn_decode as mega_gdn_decode,
)
from xllm_kernel.ops.attention.gdn.mega import (
    mega_gdn_mtp_decode as mega_gdn_mtp_decode,
)
from xllm_kernel.ops.attention.gdn.mega import (
    mega_gdn_mtp_decode_varlen as mega_gdn_mtp_decode_varlen,
)
from xllm_kernel.ops.attention.gdn.mega import (
    mega_gdn_prefill as mega_gdn_prefill,
)

__all__ = [
    "mega_gdn_decode",
    "mega_gdn_mtp_decode",
    "mega_gdn_mtp_decode_varlen",
    "mega_gdn_prefill",
    "fused_gdn_gating",
    "chunk_gated_delta_rule",
]
