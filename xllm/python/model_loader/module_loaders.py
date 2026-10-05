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

"""Shared ``load_weights`` recipes for standard module layouts.

These sit above the checkpoint-access primitives in :mod:`scoped_weight_loader`
and the pure shard math in :mod:`sharding`: they know a common module's
submodule structure, so multiple model families load the same layout once.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch

from .parallel_load_context import ParallelLoadContext
from .scoped_weight_loader import ScopedWeightLoader
from .sharding import gqa_qkv_shards


class _WeightModule(Protocol):
    """A submodule exposing a single ``weight`` tensor (embedding, final norm)."""

    @property
    def weight(self) -> torch.Tensor: ...


class _LinearModule(_WeightModule, Protocol):
    """A projection exposing ``weight`` and an optional ``bias``."""

    @property
    def bias(self) -> torch.nn.Parameter | None: ...


class _GqaAttentionModule(Protocol):
    """Structural contract :func:`load_gqa_fused_attention` requires of ``module``."""

    @property
    def qkv_proj(self) -> _LinearModule: ...
    @property
    def o_proj(self) -> _LinearModule: ...
    @property
    def q_norm(self) -> _WeightModule: ...
    @property
    def k_norm(self) -> _WeightModule: ...


class _BackboneLayer(Protocol):
    """A decoder layer that loads its own weights from a layer-scoped loader."""

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None: ...


class _CausalLMBackbone(Protocol):
    """Structural contract :func:`load_causal_lm_weights` requires of ``model``.

    ``embed_tokens`` may be ``None`` for callers that pass ``load_embedding=False``
    (a speculative draft that shares its target's embedding).
    """

    @property
    def embed_tokens(self) -> _WeightModule | None: ...
    @property
    def norm(self) -> _WeightModule: ...
    @property
    def layers(self) -> Sequence[_BackboneLayer]: ...


def load_gqa_fused_attention(
    module: _GqaAttentionModule,
    state: ScopedWeightLoader,
    context: ParallelLoadContext,
    n_kv_heads: int,
) -> None:
    """Load a fused-QKV GQA attention block's shared weights.

    Backend-specific finalization (e.g. row-parallel weight prep) differs by
    caller and stays with the caller after this returns.
    """
    state.load_fused(
        module.qkv_proj.weight,
        gqa_qkv_shards(".weight", n_kv_heads, context.tp_rank, context.tp_size),
        "qkv_proj.weight",
    )
    state.load_tensor(module.o_proj.weight, "o_proj.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size)
    if module.qkv_proj.bias is not None:
        state.load_fused(
            module.qkv_proj.bias,
            gqa_qkv_shards(".bias", n_kv_heads, context.tp_rank, context.tp_size),
            "qkv_proj.bias",
        )
    # o_proj bias is replicated (added after the all-reduce), so load unsharded.
    if module.o_proj.bias is not None:
        state.load_tensor(module.o_proj.bias, "o_proj.bias")
    state.load_tensor(module.q_norm.weight, "q_norm.weight")
    state.load_tensor(module.k_norm.weight, "k_norm.weight")


def load_causal_lm_weights(
    model: _CausalLMBackbone,
    lm_head_weight: torch.Tensor | None,
    all_weights: ScopedWeightLoader,
    context: ParallelLoadContext,
    *,
    tie_word_embeddings: bool,
    load_embedding: bool = True,
    embed_fallback: bool = False,
) -> ScopedWeightLoader:
    """Load a standard causal-LM backbone (embedding, layers, norm) and its output head.

    Returns the backbone loader, locked to the checkpoint root holding ``norm.weight``,
    so a caller (e.g. a speculative draft) can load extra tensors from the same root.
    ``load_embedding=False`` / ``lm_head_weight=None`` skip endpoints the caller loads
    itself or shares from its target model; ``embed_fallback`` lets an untied head absent
    from the checkpoint fall back to the tied embedding.
    """
    backbone = all_weights.bind_source_root("norm.weight")
    if load_embedding:
        if model.embed_tokens is None:
            raise ValueError("load_embedding=True requires a model.embed_tokens module")
        backbone.load_tensor(
            model.embed_tokens.weight, "embed_tokens.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size
        )
    for i, layer in enumerate(model.layers):
        layer.load_weights(backbone.with_prefix(f"layers.{i}."), context)
    backbone.load_tensor(model.norm.weight, "norm.weight")
    if lm_head_weight is not None:
        if not tie_word_embeddings and all_weights.has("lm_head.weight"):
            all_weights.load_tensor(
                lm_head_weight, "lm_head.weight", dim=0, rank=context.tp_rank, world_size=context.tp_size
            )
        elif tie_word_embeddings or embed_fallback:
            backbone.load_tensor(
                lm_head_weight, "embed_tokens.weight", dim=0, rank=context.tp_rank, world_size=context.tp_size
            )
        else:
            raise KeyError("checkpoint output-head weight 'lm_head.weight' not found")
    return backbone
