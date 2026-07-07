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

A concrete model builds its own graph, lm_head, and config; the wiring that is
identical across models -- the C++ bridge calling convention for ``forward`` /
``compute_logits``, and the weight-loading entry point -- lives here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch
import torch.nn as nn

from ..attn_metadata import AttentionMetadata
from ..model_runner import GraphRunner

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` and ``self.lm_head``;
      * call ``self._init_runner()`` once, after ``self.model`` exists;
      * implement ``load_weights()``.
    """

    model: nn.Module
    lm_head: nn.Module
    _runner: GraphRunner

    def _init_runner(self, config: dict | None = None) -> None:
        """Wrap ``self.model`` in the graph runner (torch.compile / cudagraph).

        The backend is selected by the C++ --python_graph_backend flag
        (passed in config["python_graph_backend"]). The max decode batch for
        graph capture is --python_graph_max_batch.
        """
        cfg = config or {}
        backend = cfg.get("python_graph_backend", "off")
        max_batch = int(cfg.get("python_graph_max_batch", 256))
        self._runner = GraphRunner(self.model, backend=backend, max_batch=max_batch)

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

    # -- inference ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata_dict: dict,
        kv_caches: list,
    ) -> torch.Tensor:
        """Called by C++ PyCausalLM::forward with explicit metadata + kv_caches.

        Args:
            input_ids: [num_tokens] token IDs
            positions: [num_tokens] position IDs
            attn_metadata_dict: dict built by C++ with all attention metadata
            kv_caches: list of (k_cache, v_cache) tuples, one per layer

        The runner owns plan + forward-context setup and graph orchestration:
        the decode full-graph path re-plans against fixed-address static buffers
        (not the raw C++ tensors), so this method just wraps the metadata and
        delegates everything to the runner.
        """
        meta = AttentionMetadata(attn_metadata_dict)
        return self._runner(input_ids, positions, meta, kv_caches)

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

        Called by the C++ bridge (``PyCausalLM::load_model``).
        The model owns ALL weight transform logic: TP slicing, fused-weight
        concatenation, format conversion, etc.
        """
        raise NotImplementedError
