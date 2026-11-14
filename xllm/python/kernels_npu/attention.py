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
import torch.nn.functional as F

from .linear import atb_matmul_ein_sum

_MIN_PAD_HEAD_DIM = 64
_MAX_PAD_HEAD_DIM = 128
_FIA_BLOCK_SIZE = 128
_FIA_FULL_ATTENTION_TOKENS = 2147483647

reshape_paged_cache = torch.ops.xllm_ops.reshape_paged_cache
update_decode_graph_metadata = torch.ops.xllm_ops.update_decode_graph_metadata


def batch_matmul_transpose(x: torch.Tensor, weight: torch.Tensor, *, graph_mode: bool) -> torch.Tensor:
    """Project MLA rows [T,H,D] with weight [H,D,O] into [T,H,O]."""
    if graph_mode and x.dtype == torch.bfloat16:
        return atb_matmul_ein_sum(x, weight)
    return torch.bmm(x.transpose(0, 1), weight).transpose(0, 1)


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
    """Run TND vision attention through Fused Infer Attention.

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

    if input_layout == "TND":
        # Match vLLM-Ascend's MM encoder FIA path. Padding changes storage,
        # while scale continues to describe the original head dimension.
        head_dim = q.shape[-1]
        if _MIN_PAD_HEAD_DIM < head_dim < _MAX_PAD_HEAD_DIM:
            q, k, v = (F.pad(tensor, (0, _MAX_PAD_HEAD_DIM - head_dim)) for tensor in (q, k, v))
        output, _ = torch_npu.npu_fused_infer_attention_score(
            query=q.contiguous(),
            key=k.contiguous(),
            value=v.contiguous(),
            atten_mask=None,
            block_table=None,
            input_layout="TND",
            block_size=_FIA_BLOCK_SIZE,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_kvlen,
            num_key_value_heads=k.shape[1],
            num_heads=num_heads,
            scale=scale,
            sparse_mode=0,
            pre_tokens=_FIA_FULL_ATTENTION_TOKENS,
            next_tokens=_FIA_FULL_ATTENTION_TOKENS,
        )
        return output[..., :head_dim]

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


__all__ = [
    "batch_matmul_transpose",
    "reshape_paged_cache",
    "update_decode_graph_metadata",
    "vision_fusion_attention",
]
