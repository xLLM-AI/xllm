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
from typing import Protocol, TypeGuard

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


class _DynamicW8A8WeightLoader(Protocol):
    """Loader operations needed by :func:`load_w8a8_dynamic_projection`."""

    def has(self, name: str, /) -> bool: ...

    def get_tensor(self, name: str, /) -> torch.Tensor: ...

    def shard(self, tensor: torch.Tensor, /, dim: int) -> torch.Tensor: ...

    def copy_in(self, param_name: str, tensor: torch.Tensor) -> None: ...


class _GqaAttentionModule(Protocol):
    """Shared QKV and output projections."""

    @property
    def qkv_proj(self) -> _LinearModule: ...
    @property
    def o_proj(self) -> _LinearModule: ...


class _GqaNorms(_GqaAttentionModule, Protocol):
    """Per-head normalization required only when Q/K norms are enabled."""

    @property
    def q_norm(self) -> _WeightModule: ...
    @property
    def k_norm(self) -> _WeightModule: ...


def load_w8a8_dynamic_projection(
    loader: _DynamicW8A8WeightLoader,
    checkpoint_module: str,
    parameter_module: str,
    shard_dims: dict[str, int] | None = None,
    *,
    require_weight_and_scale: bool = False,
    fill_missing_offset: bool = False,
    description: str = "dynamic W8A8 projection",
) -> None:
    """Load a dynamic-W8A8 projection while preserving caller key policy."""
    loaded_scale: torch.Tensor | None = None
    for suffix in ("weight", "weight_scale"):
        checkpoint_key = checkpoint_module + "." + suffix
        if not loader.has(checkpoint_key):
            if require_weight_and_scale:
                raise KeyError(f"{description} {suffix} not found; expected one of: {checkpoint_key}")
            continue
        tensor = loader.get_tensor(checkpoint_key)
        shard_dim = (shard_dims or {}).get(suffix)
        if shard_dim is not None:
            tensor = loader.shard(tensor, dim=shard_dim)
        loader.copy_in(parameter_module + "." + suffix, tensor)
        if suffix == "weight_scale":
            loaded_scale = tensor

    offset_key = checkpoint_module + ".weight_offset"
    if loader.has(offset_key):
        offset = loader.get_tensor(offset_key)
        shard_dim = (shard_dims or {}).get("weight_offset")
        if shard_dim is not None:
            offset = loader.shard(offset, dim=shard_dim)
        loader.copy_in(parameter_module + ".weight_offset", offset)
    elif fill_missing_offset:
        if loaded_scale is None:
            raise KeyError(f"{description} weight_scale is required to initialize a missing weight_offset")
        loader.copy_in(parameter_module + ".weight_offset", torch.zeros_like(loaded_scale))


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


def _has_gqa_norms(module: _GqaAttentionModule) -> TypeGuard[_GqaNorms]:
    # Registered nn.Module children are resolved through __getattr__.
    return hasattr(module, "q_norm") and hasattr(module, "k_norm")


def load_gqa_fused_attention(
    module: _GqaAttentionModule,
    state: ScopedWeightLoader,
    context: ParallelLoadContext,
    n_kv_heads: int,
    qk_norm: bool = True,
) -> None:
    """Load a fused-QKV GQA attention block's shared weights.

    Backend-specific finalization (e.g. row-parallel weight prep) differs by
    caller and stays with the caller after this returns. ``qk_norm=False``
    fits Qwen2-family towers, which carry no per-head q/k RMSNorm.
    """
    norms: _GqaNorms | None = None
    if qk_norm:
        if not _has_gqa_norms(module):
            raise TypeError("qk_norm=True requires q_norm and k_norm modules")
        norms = module
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
    if norms is not None:
        state.load_tensor(norms.q_norm.weight, "q_norm.weight")
        state.load_tensor(norms.k_norm.weight, "k_norm.weight")


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
