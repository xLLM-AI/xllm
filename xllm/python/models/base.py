# Copyright 2025-2026 The xLLM Authors.
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

"""Shared base for Python model-executor causal LMs.

A concrete model builds its own graph, lm_head, and config; the wiring that is
identical across models -- the weight-loading entry point and logits
computation -- lives here.

Model forward is driven by the runners owned by Python ModelExecutor;
PyCausalLM no longer calls model.forward() directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


@dataclass(frozen=True)
class MultimodalEmbeddings:
    """Per-step decoder inputs owned by the caller across request preparation.

    DeepStack models supply a tuple, including an empty tuple for text-only
    chunks. Other models leave it unset so decoder graph admission is preserved.
    """

    input_embedding: torch.Tensor
    deepstack_input_embeds: tuple[torch.Tensor, ...] | None = None


if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    class _EmbeddingBackbone(Protocol):
        embed_tokens: Callable[[torch.Tensor], torch.Tensor]

        def __call__(
            self,
            embeddings: torch.Tensor,
            positions: torch.Tensor,
            *args: torch.Tensor | tuple[torch.Tensor, ...] | None,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


class TokenEmbeddingRunner(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.backbone = cast("_EmbeddingBackbone", backbone)
        self.execution_metadata_builders = getattr(backbone, "execution_metadata_builders", ())

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embedding: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.backbone.embed_tokens(input_ids) if input_embedding is None else input_embedding
        return self.backbone(embeddings, positions)


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` and ``self.lm_head``;
      * implement ``load_weights()``.
    """

    model: nn.Module
    lm_head: nn.Module
    runner_model_cls: type[nn.Module] | None = None

    @property
    def runner_model(self) -> nn.Module:
        if self.runner_model_cls is None:
            return self.model
        return self.runner_model_cls(self.model)

    @staticmethod
    def resolve_dtype(dtype: object) -> torch.dtype:
        """Resolve a torch dtype from a ``torch.dtype`` or its string name."""
        if isinstance(dtype, torch.dtype):
            return dtype
        name = dtype if dtype else "bfloat16"
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return resolved

    def compute_logits(self, hidden: torch.Tensor, selected_idxes: torch.Tensor | None) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        return self.lm_head(hidden)

    # -- weight loading -------------------------------------------------------
    def load_weights(
        self,
        state_dicts: list[StateDict],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Load checkpoint weights into this model's parameters.

        Called by the C++ bridge (``PyCausalLM::load_model``).
        The model owns ALL weight transform logic: TP slicing, fused-weight
        concatenation, format conversion, etc.
        """
        raise NotImplementedError

    def adapt_weights_for_reference_model(
        self,
        reference_model_path: str,
    ) -> None:
        """Adapt loaded draft weights to the reference model's runtime basis."""
        del reference_model_path
