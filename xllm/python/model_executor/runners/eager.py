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

from __future__ import annotations

import torch

from xllm.python.attention.backend import AttentionMetadata
from xllm.python.model_executor.cp_utils import build_cp_context
from xllm.python.model_executor.forward_context import (
    ForwardContext,
    LayerSynchronizer,
    forward_context,
)
from xllm.python.model_executor.runners.base import BaseRunner


def _per_seq_lens_from_metadata(metadata: AttentionMetadata) -> list[int] | None:
    """Per-sequence query lengths for the packed prefill batch, or None.

    Read straight from ``q_seq_lens_host`` — the host-side per-sequence query
    lengths (NPU keeps these non-cumulative, one entry per sequence), so no
    D2H copy and no diff. Returns None when the field is absent so the caller
    falls back to the non-CP path.
    """
    lens = metadata.q_seq_lens_host
    if lens is None:
        return None
    return lens.tolist()


class EagerRunner(BaseRunner):
    # Context-Parallel config, set by ModelExecutor when cp_size > 1. CP shards
    # the prefill sequence across these ranks; decode is left on the non-CP path.
    cp_size: int = 1
    cp_rank: int = 0

    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
    ) -> torch.Tensor:
        self.attention_backend.prepare(metadata)

        cp_context = None
        if self.cp_size > 1 and metadata.is_prefill:
            seq_lens = _per_seq_lens_from_metadata(metadata)
            if seq_lens is not None:
                cp_context = build_cp_context(seq_lens, self.cp_size, self.cp_rank, self.device)

        with forward_context(
            ForwardContext(
                self.attention_backend,
                self.device,
                metadata,
                self.layer_caches,
                layer_synchronizer=layer_synchronizer,
                cp_context=cp_context,
            )
        ):
            if input_embedding is None:
                return self.model(input_ids, positions)
            return self.model(input_ids, positions, input_embedding)
