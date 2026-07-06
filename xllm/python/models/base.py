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

"""Shared base for Python model-executor causal LMs.

A concrete model only builds its own graph (``self.model``), its ``self.lm_head``
and its config, then calls ``self._init_runner()``; the wiring that is identical
across models -- GraphRunner (torch.compile / cudagraph) hookup, the C++ bridge
calling convention for ``forward`` / ``compute_logits``, and the weight-loading
entry point -- lives here so adding a model does not re-implement it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from ..forward_batch import ForwardBatch
from ..model_runner import GraphRunner

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` (the pure-GPU forward graph) and ``self.lm_head``;
      * call ``self._init_runner()`` once, after ``self.model`` exists;
      * implement ``load_weights()``.
    """

    model: nn.Module
    lm_head: nn.Module
    _runner: GraphRunner

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

    def _init_runner(self) -> None:
        """Wrap ``self.model`` in the graph runner (torch.compile / cudagraph)."""
        self._runner = GraphRunner(self.model)

    # -- inference ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        return self._runner(input_ids, positions)

    def compute_logits(
        self, hidden: torch.Tensor, selected_idxes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        return self.lm_head(hidden)

    # -- weight loading -------------------------------------------------------
    def load_weights(
        self,
        state_dicts: List["StateDict"],
        tp_rank: int,
        tp_size: int,
    ) -> None:
        """Load checkpoint weights into this model's parameters.

        Called by the C++ bridge (``PyCausalLM::load_model``) with:
          * ``state_dicts`` — list of ``xllm_weight_loader.StateDict`` objects
            wrapping the raw checkpoint shards (supports ``get_tensor``,
            ``get_sharded_tensor``, ``has``, ``keys``).
          * ``tp_rank`` / ``tp_size`` — this rank's position in the TP group.

        The model owns ALL weight transform logic: TP slicing, fused-weight
        concatenation, format conversion (NZ, quantization), etc.  C++ only
        provides the raw tensor access; no sharding knowledge lives there.
        """
        raise NotImplementedError
