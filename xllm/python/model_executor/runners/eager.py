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
from xllm.python.model_executor.runners.base import BaseRunner, ModelExecutionOutput


def _per_seq_lens_from_metadata(
    metadata: AttentionMetadata,
    *,
    include_prefix: bool,
) -> tuple[list[int], list[int]] | None:
    """Per-sequence query and KV lengths for the packed prefill batch.

    Read the host-side, non-cumulative lengths without a D2H copy. MLA chunked
    prefill keeps the full KV length so its cached prefix is visible; non-MLA
    CP preserves the existing query-only contract by using the query length for
    both values. Returns None when a required field is absent.
    """
    q_lens = metadata.q_seq_lens_host
    kv_lens = metadata.kv_seq_lens_host if include_prefix else q_lens
    if q_lens is None or kv_lens is None:
        return None
    return q_lens.tolist(), kv_lens.tolist()


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
    ) -> ModelExecutionOutput:
        cp_context = None
        is_mla = self.attention_backend.is_mla
        is_mla_cp_prefill = self.cp_size > 1 and is_mla and (metadata.is_prefill or metadata.is_chunked_prefill)
        if is_mla_cp_prefill and metadata.is_spec_verify:
            raise NotImplementedError("Python Context-Parallel does not support MTP speculative verification")
        if is_mla_cp_prefill and metadata.is_mixed:
            raise NotImplementedError("Python Context-Parallel does not support mixed batches")
        use_cp_context = self.cp_size > 1 and (metadata.is_prefill or (is_mla and metadata.is_chunked_prefill))
        if use_cp_context:
            seq_lens = _per_seq_lens_from_metadata(
                metadata,
                include_prefix=is_mla,
            )
            if seq_lens is None:
                if is_mla:
                    raise RuntimeError("Python Context-Parallel requires host query and KV sequence lengths")
            else:
                q_seq_lens, kv_seq_lens = seq_lens
                cp_context = build_cp_context(
                    q_seq_lens,
                    kv_seq_lens,
                    self.cp_size,
                    self.cp_rank,
                    self.device,
                )

        # Admission and context construction must finish before prepare(). A
        # sharded MLA backend enters CP collectives during prepare, so rejecting
        # unsupported batches afterwards could leave peer ranks deadlocked.
        self.attention_backend.prepare(metadata)

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
