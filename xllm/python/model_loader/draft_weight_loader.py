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

"""Load optional draft vocabulary and restore QuaRot target weights."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import torch
import torch.nn as nn

from .parallel_load_context import ParallelLoadContext
from .scoped_weight_loader import ScopedWeightLoader
from .sharding import shard_tensor

if TYPE_CHECKING:
    from xllm.python.layers import ColumnParallelLinear, HiddenParallelEmbedding


class _DraftModelWithConfig(Protocol):
    cfg: Any
    dtype: torch.dtype
    device: torch.device


class _DraftEmbedding(_DraftModelWithConfig, Protocol):
    embed_tokens: HiddenParallelEmbedding | None


class _DraftLMHead(_DraftModelWithConfig, Protocol):
    lm_head: nn.Module | None


class _DraftCausalLM(_DraftLMHead, Protocol):
    model: Any


def _create_embedding_shard(model: _DraftModelWithConfig, context: ParallelLoadContext) -> HiddenParallelEmbedding:
    from xllm.python.layers import HiddenParallelEmbedding

    return HiddenParallelEmbedding(
        model.cfg.vocab_size,
        model.cfg.hidden_size // context.tp_size,
        context.tp_size,
        dtype=model.dtype,
        device=model.device,
    )


def _create_lm_head_shard(model: _DraftModelWithConfig, context: ParallelLoadContext) -> ColumnParallelLinear:
    from xllm.python.layers import ColumnParallelLinear

    cfg = model.cfg
    return ColumnParallelLinear(
        cfg.hidden_size,
        (getattr(cfg, "draft_vocab_size", 0) or cfg.vocab_size) // context.tp_size,
        context.tp_size,
        gather_output=True,
        dtype=model.dtype,
        device=model.device,
    )


def load_draft_embedding_if_present(
    model: _DraftEmbedding,
    weights: ScopedWeightLoader,
    *,
    context: ParallelLoadContext,
) -> None:
    if weights.has("embed_tokens.weight"):
        module = _create_embedding_shard(model, context)
        weights.load_tensor(
            module.weight, "embed_tokens.weight", dim=1, rank=context.tp_rank, world_size=context.tp_size
        )
        model.embed_tokens = module


def load_draft_lm_head_if_present(
    model: _DraftLMHead,
    weights: ScopedWeightLoader,
    *,
    context: ParallelLoadContext,
) -> None:
    if weights.has("lm_head.weight"):
        module = _create_lm_head_shard(model, context)
        weights.load_tensor(module.weight, "lm_head.weight", dim=0, rank=context.tp_rank, world_size=context.tp_size)
        model.lm_head = module


def _target_ties_word_embeddings(target_model_path: str | Path) -> bool:
    config_path = Path(target_model_path) / "config.json"
    if not config_path.is_file():
        return False
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    text_config = config.get("text_config") or {}
    return bool(text_config.get("tie_word_embeddings", config.get("tie_word_embeddings", False)))


def _find_target_weight(
    model_path: Path,
    names: tuple[str, ...],
) -> tuple[Path, str]:
    """Locate a target tensor without loading unrelated weight shards."""
    from safetensors import safe_open

    for index_path in sorted(model_path.glob("*.safetensors.index.json")):
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        for name in names:
            if name in weight_map:
                return model_path / weight_map[name], name
    for shard_path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
        for name in names:
            if name in keys:
                return shard_path, name
    raise KeyError(f"target weights have none of {names}: {model_path}")


@torch.no_grad()
def _load_target_weight_in_draft_basis(
    weight: torch.Tensor,
    target_model_path: str,
    names: tuple[str, ...],
    rotation: torch.Tensor,
    *,
    shard_dim: int,
    context: ParallelLoadContext,
) -> None:
    """Undo the target's hidden rotation into one draft TP weight shard."""
    from safetensors import safe_open

    shard_path, name = _find_target_weight(Path(target_model_path), names)
    expected_shape = list(weight.shape)
    expected_shape[shard_dim] *= context.tp_size
    if rotation.shape != (expected_shape[1], expected_shape[1]):
        raise ValueError("target rotation must match the draft hidden size")
    row_offset = 0
    if shard_dim == 0:
        row_offset = context.tp_rank * weight.size(0)
    else:
        rotation = shard_tensor(rotation, 0, context.tp_rank, context.tp_size, name="global_rotation", contiguous=False)
    rotation_t = rotation.to(device=weight.device, dtype=torch.float32).T

    with safe_open(str(shard_path), framework="pt", device="cpu") as shard:
        target_slice = shard.get_slice(name)
        if target_slice.get_shape() != expected_shape:
            raise ValueError(f"target {name} has shape {target_slice.get_shape()}, expected {expected_shape}")
        # Chunk rows to bound temporary device memory independently of vocab
        # size; each rank reads only its own slice, so no collective is needed.
        rows_per_chunk = 1024
        for start in range(0, weight.size(0), rows_per_chunk):
            end = min(start + rows_per_chunk, weight.size(0))
            block = target_slice[row_offset + start : row_offset + end]
            if not block.is_floating_point():
                raise ValueError(f"target {name} must be a floating-point weight")
            aligned = block.to(device=weight.device, dtype=torch.float32) @ rotation_t
            weight[start:end].copy_(aligned)


def load_missing_draft_vocab_from_quarot_target(
    draft: _DraftCausalLM,
    target_model_path: str,
    rotation: torch.Tensor,
) -> None:
    """Fill missing embedding/head modules in the draft's original hidden basis.

    This runs after ``load_draft_*_if_present`` and before target sharing.
    Existing modules are preserved; only missing modules need target weights.
    """
    cfg = draft.cfg
    context = ParallelLoadContext(cfg.tp_rank, cfg.tp_size)
    embed_names = (
        "model.language_model.embed_tokens.weight",
        "language_model.model.embed_tokens.weight",
        "model.embed_tokens.weight",
        "embed_tokens.weight",
    )

    def _load_vocab_shard(weight: torch.Tensor, names: tuple[str, ...], shard_dim: int) -> None:
        _load_target_weight_in_draft_basis(
            weight,
            target_model_path,
            names,
            rotation,
            shard_dim=shard_dim,
            context=context,
        )

    if draft.model.embed_tokens is None:
        embedding = _create_embedding_shard(draft, context)
        _load_vocab_shard(embedding.weight, embed_names, shard_dim=1)
        draft.model.embed_tokens = embedding
    if draft.lm_head is None:
        head = _create_lm_head_shard(draft, context)
        # A tied target may save only embed_tokens.weight. The output head uses
        # the same full tensor, but shards its vocabulary dimension.
        head_names = (
            embed_names
            if _target_ties_word_embeddings(target_model_path)
            else ("lm_head.weight", "language_model.lm_head.weight", "model.lm_head.weight")
        )
        _load_vocab_shard(head.weight, head_names, shard_dim=0)
        draft.lm_head = head
