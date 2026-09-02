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
"""Torch reference ops for DCP SFA decode remap and merge."""

from __future__ import annotations

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout


def remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    topk_count = topk_indices.shape[-1]
    if topk_count > index_topk:
        raise RuntimeError(f"topk_indices last dimension ({topk_count}) exceeds configured index_topk ({index_topk}).")

    local_table = layout.localize_slots(topk_indices)
    owned_entries = local_table >= 0
    original_order = torch.arange(
        topk_count,
        dtype=torch.float32,
        device=topk_indices.device,
    ).expand_as(topk_indices)
    pack_keys = original_order + (~owned_entries).to(torch.float32) * topk_count
    _, pack_order = torch.sort(pack_keys, dim=-1)
    return torch.gather(local_table, dim=-1, index=pack_order.to(torch.int32))


def merge_dcp_outputs(
    output_recv: torch.Tensor,
    lse_recv: torch.Tensor,
) -> torch.Tensor:
    if output_recv.ndim != 4 or lse_recv.ndim != 3 or output_recv.shape[:3] != lse_recv.shape:
        raise RuntimeError(
            "DCP output merge expects matching rank/token/head dimensions, "
            f"got {tuple(output_recv.shape)} and {tuple(lse_recv.shape)}."
        )
    output_dtype = output_recv.dtype
    lse_recv = lse_recv.masked_fill(~torch.isfinite(lse_recv), float("-inf"))
    weights = torch.softmax(lse_recv, dim=0)
    weights = torch.nan_to_num(weights, nan=0.0)

    output = (output_recv.to(lse_recv.dtype) * weights.unsqueeze(-1)).sum(dim=0)
    return output.movedim(1, 0).contiguous().to(output_dtype)
