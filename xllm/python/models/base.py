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
calling convention for ``forward`` / ``compute_logits``, the pure-copy
``load_assembled_weights`` -- lives here so adding a model does not re-implement
it. ``sharding_plan`` stays abstract because it encodes the model's own layout.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from ..forward_batch import ForwardBatch
from ..model_runner import GraphRunner

# sharding_plan() shape (see PyModelBase.sharding_plan):
#   (target_param_name, [(candidate_source_names, shard_dim, is_kv)], cat_dim)
ShardingSource = Tuple[List[str], int, bool]
ShardingEntry = Tuple[str, List[ShardingSource], int]
ShardingPlan = List[ShardingEntry]


class PyModelBase(nn.Module):
    """Base class for causal LMs the C++ ``PyCausalLM`` bridge drives.

    Subclass contract:
      * build ``self.model`` (the pure-GPU forward graph) and ``self.lm_head``;
      * call ``self._init_runner()`` once, after ``self.model`` exists;
      * implement ``sharding_plan()``.
    """

    model: nn.Module
    lm_head: nn.Module
    _runner: GraphRunner

    @staticmethod
    def resolve_dtype(dtype: object) -> torch.dtype:
        """Resolve a torch dtype from a ``torch.dtype`` or its string name.

        Names are resolved directly against ``torch`` (e.g. ``"bfloat16"`` ->
        ``torch.bfloat16``), which natively covers every real dtype name; an
        empty/None value defaults to bfloat16. An unrecognized name is a hard
        error rather than a silent fallback, so a misconfigured dtype surfaces
        instead of quietly running in the wrong precision.
        """
        if isinstance(dtype, torch.dtype):
            return dtype
        name = dtype if dtype else "bfloat16"
        resolved = getattr(torch, name, None)
        if not isinstance(resolved, torch.dtype):
            raise ValueError(f"Unknown dtype: {dtype!r}")
        return resolved

    def _init_runner(self) -> None:
        """Wrap ``self.model`` in the graph runner (torch.compile / cudagraph
        capture, gated by ``XLLM_TC_BACKEND``).

        GraphRunner is a plain object (not an nn.Module), so assigning it does
        not re-register the wrapped (possibly OptimizedModule) callable as a
        duplicate submodule -- weights stay shared through ``self.model``.
        Capture is lazy (first forward), so weights loaded via
        ``load_assembled_weights`` are in place before tracing.
        """
        self._runner = GraphRunner(self.model)

    # -- inference ------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        batch: ForwardBatch,
    ) -> torch.Tensor:
        # `batch` is kept for the C++ bridge's calling convention but no longer
        # enters the (compilable) model graph; positions is passed explicitly.
        # Inference no-grad is owned by the C++ bridge: PyCausalLM::forward wraps
        # this call in torch::NoGradGuard (py_causal_lm.cpp), which shares the
        # thread-local GradMode with the embedded interpreter. That elides the
        # AOTAutograd backward graph -- whose "pending, uninvoked backwards" would
        # force cudagraph_trees off its fast-path replay -- with no per-model guard.
        return self._runner(input_ids, positions)

    def compute_logits(
        self, hidden: torch.Tensor, selected_idxes: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # no-grad owned by PyCausalLM::logits (torch::NoGradGuard); see forward().
        if selected_idxes is not None and selected_idxes.numel() > 0:
            hidden = hidden.index_select(0, selected_idxes)
        return self.lm_head(hidden)

    # -- weight loading -------------------------------------------------------
    def sharding_plan(self) -> ShardingPlan:
        """Declare how each parameter is assembled from checkpoint tensors.

        Returns a list of ``(target_param_name, sources, cat_dim)`` where
        ``sources`` is a list of ``(candidate_names, shard_dim, is_kv)``:
          * ``candidate_names`` -- checkpoint tensor names tried in order;
          * ``shard_dim`` -- dim to shard on this rank, or ``< 0`` for replicated;
          * ``is_kv`` -- shard with the GQA KV-replica ``(rank, world)`` coords.

        The C++ bridge (``PyCausalLM::load_model``) EXECUTES this plan via the
        native ``StateDict::get_sharded_tensor`` and feeds back per-rank tensors,
        so the sharding RULES live with the model that owns its layout while the
        EXECUTION reuses the single validated native chunk path.
        """
        raise NotImplementedError

    def load_assembled_weights(
        self, items: Iterable[Tuple[str, torch.Tensor]]
    ) -> None:
        """Copy the C++-assembled per-rank tensors into the matching params.

        ``items`` are ``(target_param_name, tensor)`` produced by executing
        ``sharding_plan`` in C++; each tensor is already sharded/concatenated for
        this rank, so this is a pure copy -- no sharding logic remains in Python.
        """
        params = dict(self.named_parameters())
        for target, tensor in items:
            param = params[target]
            param.data.copy_(tensor.to(dtype=param.dtype, device=param.device))
