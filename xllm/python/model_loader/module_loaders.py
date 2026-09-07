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

    weight: torch.Tensor


class _LinearModule(Protocol):
    """A projection exposing ``weight`` and an optional ``bias``."""

    weight: torch.Tensor
    bias: torch.Tensor | None


class _GqaAttentionModule(Protocol):
    """Structural contract :func:`load_gqa_fused_attention` requires of ``module``."""

    qkv_proj: _LinearModule
    o_proj: _LinearModule
    q_norm: _WeightModule
    k_norm: _WeightModule


class _BackboneLayer(Protocol):
    """A decoder layer that loads its own weights from a layer-scoped loader."""

    def load_weights(self, weights: ScopedWeightLoader, context: ParallelLoadContext) -> None: ...


class _CausalLMBackbone(Protocol):
    """Structural contract :func:`load_causal_lm_weights` requires of ``model``.

    ``embed_tokens`` may be ``None`` for callers that pass ``load_embedding=False``
    (a speculative draft that shares its target's embedding).
    """

    embed_tokens: _WeightModule | None
    norm: _WeightModule
    layers: Sequence[_BackboneLayer]


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
        state.load_tensor(module.o_proj.bias, "o_proj.bias")
    state.load_tensor(module.q_norm.weight, "q_norm.weight")
    state.load_tensor(module.k_norm.weight, "k_norm.weight")


def _load_lm_head(
    lm_head_weight: torch.Tensor,
    backbone: ScopedWeightLoader,
    all_weights: ScopedWeightLoader,
    context: ParallelLoadContext,
    *,
    tie_word_embeddings: bool,
    tied_name: str,
    untied_names: Sequence[str],
    embed_fallback: bool,
) -> None:
    """Load the output head: tied → the embedding (``tied_name``) from ``backbone``;
    untied → the first present ``untied_names`` from the unlocked ``all_weights``.

    ``embed_fallback`` covers models that ship no separate head tensor: an untied head
    absent from the checkpoint then falls back to the tied embedding instead of raising.
    """
    if not tie_word_embeddings:
        name = all_weights.first_present(untied_names)
        if name is not None:
            all_weights.load_tensor(lm_head_weight, name, dim=0, rank=context.tp_rank, world_size=context.tp_size)
            return
        if not embed_fallback:
            raise KeyError(f"checkpoint output-head weight not found among {tuple(untied_names)}")
    backbone.load_tensor(lm_head_weight, tied_name, dim=0, rank=context.tp_rank, world_size=context.tp_size)


def load_causal_lm_weights(
    model: _CausalLMBackbone,
    lm_head_weight: torch.Tensor | None,
    all_weights: ScopedWeightLoader,
    context: ParallelLoadContext,
    *,
    tie_word_embeddings: bool,
    load_embedding: bool = True,
    embed_names: Sequence[str] = ("embed_tokens.weight",),
    untied_names: Sequence[str] = ("lm_head.weight",),
    embed_fallback: bool = False,
    root_probe: str = "norm.weight",
) -> ScopedWeightLoader:
    """Load a standard causal-LM backbone (embedding, layers, norm, tied/own head).

    Locks the backbone to the single root under ``root_probe`` (default ``norm.weight``,
    present in every causal LM); the head resolves via the unlocked ``all_weights``. Each layer loads
    itself through its own ``load_weights``. Returns the locked backbone so a caller
    (e.g. a speculative MTP head, or DeepSeek-V4's ``hc_head``) can load extra tensors
    from the same root. ``load_embedding=False`` / ``lm_head_weight=None`` skip the
    endpoints a caller loads itself or shares from its target model. ``embed_names``
    lists the checkpoint aliases for the embedding (e.g. DeepSeek-V4's ``embed.weight``);
    a tied head reuses whichever alias the embedding resolved to. ``untied_names`` /
    ``embed_fallback`` are forwarded to :func:`_load_lm_head`.

    A tied or ``embed_fallback`` head reads the resolved embedding name, so it requires
    ``load_embedding=True``; callers passing ``load_embedding=False`` share the target's
    embedding and pass ``lm_head_weight=None``.
    """
    backbone = all_weights.bind_source_root(root_probe)
    if not embed_names:
        raise ValueError("embed_names must be non-empty")
    if lm_head_weight is not None and not load_embedding and (tie_word_embeddings or embed_fallback):
        raise ValueError("tied / embed_fallback head requires load_embedding=True to resolve the embedding name")
    embed_name = embed_names[0]
    if load_embedding:
        if model.embed_tokens is None:
            raise ValueError("load_embedding=True requires a model.embed_tokens module")
        resolved = backbone.first_present(embed_names)
        if resolved is None:
            raise KeyError(f"checkpoint embedding weight not found among {tuple(embed_names)}")
        embed_name = resolved
        backbone.load_tensor(
            model.embed_tokens.weight, embed_name, dim=1, rank=context.tp_rank, world_size=context.tp_size
        )
    for i, layer in enumerate(model.layers):
        layer.load_weights(backbone.with_prefix(f"layers.{i}."), context)
    backbone.load_tensor(model.norm.weight, "norm.weight")
    if lm_head_weight is not None:
        _load_lm_head(
            lm_head_weight,
            backbone,
            all_weights,
            context,
            tie_word_embeddings=tie_word_embeddings,
            tied_name=embed_name,
            untied_names=untied_names,
            embed_fallback=embed_fallback,
        )
    return backbone
