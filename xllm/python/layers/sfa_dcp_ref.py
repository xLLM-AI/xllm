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
"""Torch reference ops for DCP SFA decode remap."""

from __future__ import annotations

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout


def remap_sparse_indices(
    topk_indices: torch.Tensor,
    layout: KVShardLayout,
    index_topk: int,
) -> torch.Tensor:
    """Pack this rank's owned sparse slots to the front of the SFA width.

    ``index_topk`` is the configured SFA width and is only a lower bound here: a
    kPool indexer may emit a wider last dim (``topk + index_kpool - 1`` when
    always-select-tail is on). SFA stops at the first ``-1``, so the tail columns
    have to be packed into the same leading run as the top-k ones.
    """
    last_dim = int(topk_indices.shape[-1])
    if last_dim < index_topk:
        raise RuntimeError(
            f"sparse indices last dim ({last_dim}) is narrower than the configured index_topk ({index_topk})."
        )
    return layout.pack_owned_slots(topk_indices)
