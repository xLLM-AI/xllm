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

"""NPU paged-attention support kernels.

Both operators are capturable: they read only their arguments, so a decode
graph can replay them without re-planning.
"""

from __future__ import annotations

import torch

reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata
_TRANSPOSE_BATCHMATMUL = getattr(
    torch.ops.npu,
    "npu_transpose_batchmatmul",
    None,
)


def vision_fusion_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    actual_seq_qlen: list[int],
    actual_seq_kvlen: list[int],
    num_heads: int,
    scale: float,
    input_layout: str = "TND",
) -> torch.Tensor:
    """Run fused self-attention for ViT blocks via ``torch_npu.npu_fusion_attention``.

    Args:
        q, k, v: Query/key/value in the layout specified by ``input_layout``.
        actual_seq_qlen: Cumulative query sequence lengths.
        actual_seq_kvlen: Cumulative key/value sequence lengths.
        num_heads: Number of attention heads.
        scale: Attention scaling factor.
        input_layout: Tensor layout (default ``"TND"``).

    Returns:
        Attention output with same shape as ``q``.
    """
    import torch_npu

    return torch_npu.npu_fusion_attention(
        q,
        k,
        v,
        actual_seq_qlen=actual_seq_qlen,
        actual_seq_kvlen=actual_seq_kvlen,
        head_num=num_heads,
        scale=scale,
        input_layout=input_layout,
    )[0]


def batch_matmul_transpose(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Project MLA values with the dedicated NPU transposed-BMM kernel."""
    if x.device.type not in ("npu", "privateuseone") or _TRANSPOSE_BATCHMATMUL is None:
        return torch.bmm(x, weight).transpose(0, 1)
    return _TRANSPOSE_BATCHMATMUL(x, weight, perm_y=(1, 0, 2))


__all__ = [
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "vision_fusion_attention",
    "batch_matmul_transpose",
]
