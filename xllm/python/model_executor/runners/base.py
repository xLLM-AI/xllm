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

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn

from xllm.python.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    LayerCache,
)
from xllm.python.model_executor.forward_context import EplbRuntimeState, ExecutionMetadataBuilder, LayerSynchronizer
from xllm.python.model_executor.input_batch import InputBatch

ModelExecutionOutput = (
    torch.Tensor | tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]
)


@dataclass(frozen=True)
class SpeculativeRuntimeOutput:
    """Owned snapshots consumed by the C++ MTP worker and API serialization."""

    committed_tokens: torch.Tensor
    accepted_count: torch.Tensor
    # Compact [2 * batch, hidden], interleaved previous/current selected rows.
    # At count=0 the previous row is ignored and prepare uses its placeholder.
    target_embeddings: torch.Tensor
    # int64 token matrix followed by int32 counts; views share this lease.
    token_state: torch.Tensor
    logprobs: torch.Tensor | None = None
    top_logprobs: torch.Tensor | None = None
    top_tokens: torch.Tensor | None = None


class BaseRunner(ABC):
    def __init__(
        self,
        model: nn.Module,
        attention_backend: AttentionBackend,
        device: torch.device,
    ) -> None:
        self.model = model
        self.attention_backend = attention_backend
        self.device = device
        self.layer_caches: list[LayerCache] = []
        builders = tuple(getattr(model, "execution_metadata_builders", ()))
        seen_types: set[object] = set()
        for builder in builders:
            metadata_type = builder.metadata_type
            if metadata_type in seen_types:
                raise RuntimeError(f"duplicate execution metadata builder for {metadata_type.__name__}")
            seen_types.add(metadata_type)
        self.execution_metadata_builders: tuple[ExecutionMetadataBuilder, ...] = builders

    def bind_layer_caches(self, layer_caches: list[LayerCache]) -> None:
        for builder in self.execution_metadata_builders:
            builder.bind_layer_caches(layer_caches)
        self.layer_caches = layer_caches

    def _build_execution_contexts(
        self,
        metadata: AttentionMetadata,
        input_batch: InputBatch | None,
    ) -> dict[object, object]:
        if not self.execution_metadata_builders:
            return {}
        if input_batch is None:
            raise RuntimeError("execution metadata builders require upstream InputBatch metadata")
        if input_batch.num_tokens != int(metadata.slot_mapping.numel()):
            raise ValueError("execution metadata must contain one slot per input token")
        execution_contexts: dict[object, object] = {}
        for builder in self.execution_metadata_builders:
            execution_contexts[builder.metadata_type] = builder.build(input_batch, metadata)
        return execution_contexts

    @staticmethod
    def _slice_output(
        output: ModelExecutionOutput,
        batch_size: int,
        *,
        detach: bool = True,
    ) -> ModelExecutionOutput:
        """Slice graph outputs to the active rows, optionally copying out of graph storage.

        Graph replay writes the same persistent output allocation on every
        step, so callers that retain the output past the next replay need a
        detached copy; an ordered-stream view avoids the extra device copy.
        """

        def _trim(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor[:batch_size]
            return tensor.clone() if detach else tensor

        if not isinstance(output, tuple):
            return _trim(output)
        hidden, aux_hidden = output[:2]
        hidden = _trim(hidden)
        if aux_hidden is not None:
            aux_hidden = aux_hidden[:batch_size]
            if detach:
                aux_hidden = aux_hidden.clone()
        if len(output) == 2:
            return hidden, aux_hidden
        topk = output[2]
        if topk is not None:
            topk = topk[:batch_size]
            if detach:
                topk = topk.clone()
        return hidden, aux_hidden, topk

    @abstractmethod
    def execute(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        metadata: AttentionMetadata,
        input_embedding: torch.Tensor | None = None,
        layer_synchronizer: LayerSynchronizer | None = None,
        eplb: EplbRuntimeState | None = None,
        input_batch: InputBatch | None = None,
    ) -> ModelExecutionOutput:
        pass
