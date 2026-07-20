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

"""Tensor-parallel word embedding (sharded on the hidden dim)."""

from __future__ import annotations

import torch
import torch.nn as nn

from xllm.python import ops


class HiddenParallelEmbedding(nn.Module):
    """Embedding sharded on the hidden/embedding dim (dim 1), NOT the vocab dim:
    each rank owns ``[vocab, hidden_per_partition]`` (the full vocab rows, a
    slice of hidden columns) and all-gathers along the last dim to rebuild the
    full hidden vector. Reproduces native WordEmbedding
    (word_embedding_impl.cpp: dim-1 shard + gather), required for byte parity.
    """

    def __init__(
        self,
        num_embeddings: int,
        hidden_per_partition: int,
        tp_size: int,
        dtype: torch.dtype | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.tp_size = tp_size
        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                hidden_per_partition,
                dtype=dtype,
                device=device,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.embedding(input_ids, self.weight)
        if self.tp_size > 1:
            out = ops.all_gather(out, dim=-1, world_size=self.tp_size)
        return out
