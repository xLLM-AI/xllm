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

import torch
from xllm_kernel.ops.attention.gdn import chunk_gated_delta_rule as chunk_gated_delta_rule
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


def fused_gdn_gating(
    a_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute decay gate g and beta from raw projections via TileLang kernel."""
    from .tilelang.fused_gdn_gating import fused_gdn_gating_kernel_jit

    num_batches, num_heads = a.shape
    kernel = fused_gdn_gating_kernel_jit(
        num_batches=num_batches,
        compile_max_batch=num_batches,
        num_heads=num_heads,
    )

    g_out = torch.empty(1, num_batches, num_heads, dtype=torch.float32, device=a.device)
    beta_out = torch.empty(1, num_batches, num_heads, dtype=a.dtype, device=a.device)
    kernel(
        a_log.to(torch.float32).contiguous(),
        a.contiguous(),
        b.contiguous(),
        dt_bias.to(torch.float32).contiguous(),
        g_out.squeeze(0),
        beta_out.squeeze(0),
        num_batches,
        1.0,  # softplus_beta
        20.0,  # softplus_threshold
    )
    return g_out, beta_out


__all__ = [
    "mega_gdn_decode",
    "mega_gdn_mtp_decode",
    "mega_gdn_mtp_decode_varlen",
    "mega_gdn_prefill",
    "fused_gdn_gating",
    "chunk_gated_delta_rule",
]
