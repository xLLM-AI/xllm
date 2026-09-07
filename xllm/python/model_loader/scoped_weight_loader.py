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

"""Model-independent helpers for scoped checkpoint access."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol

import torch

from .sharding import shard_tensor


class StateDictLike(Protocol):
    def has(self, name: str) -> bool: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class ScopedWeightLoader:
    """A lightweight view over one or more checkpoint shards."""

    def __init__(
        self,
        state_dicts: Sequence[StateDictLike],
        prefix: str = "",
        src_prefixes: Sequence[str] = ("",),
        name_aliases: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self._state_dicts = state_dicts
        self._prefix = prefix
        self._src_prefixes = tuple(src_prefixes)
        # Per requested name, resolution tries each alias under each src prefix
        # prepended; bare names match both prefixed and unprefixed checkpoints.
        self._name_aliases: Mapping[str, Sequence[str]] = name_aliases or {}

    @property
    def prefix(self) -> str:
        return self._prefix

    def with_prefix(self, prefix: str) -> ScopedWeightLoader:
        return ScopedWeightLoader(
            self._state_dicts,
            self._prefix + prefix,
            src_prefixes=self._src_prefixes,
            name_aliases=self._name_aliases,
        )

    def _resolve(self, local_name: str) -> tuple[StateDictLike, str] | None:
        """First present ``(state_dict, resolved_name)`` for the scoped name,
        trying each alias under each ``src_prefix`` prepended."""
        name = self._prefix + local_name
        for alias in self._name_aliases.get(name, (name,)):
            for prefix in self._src_prefixes:
                full = prefix + alias
                for state in self._state_dicts:
                    if state.has(full):
                        return state, full
        return None

    def has(self, local_name: str) -> bool:
        return self._resolve(local_name) is not None

    def first_present(self, local_names: Sequence[str]) -> str | None:
        """First of ``local_names`` present in the checkpoint, or ``None``."""
        for name in local_names:
            if self.has(name):
                return name
        return None

    def bind_source_root(self, probe: str) -> ScopedWeightLoader:
        """Lock to the single ``src_prefix`` whose scope contains ``probe``.

        A multi-root loader tries every ``src_prefix`` per tensor; once the model
        root is chosen by a probe (e.g. ``embed_tokens.weight``), backbone weights
        must all resolve under that one root instead of silently mixing roots.
        Probing honors ``name_aliases`` (via :meth:`has`), matching normal resolution.
        """
        for prefix in self._src_prefixes:
            candidate = ScopedWeightLoader(
                self._state_dicts,
                self._prefix,
                src_prefixes=(prefix,),
                name_aliases=self._name_aliases,
            )
            if candidate.has(probe):
                return candidate
        raise KeyError(f"no checkpoint root among {self._src_prefixes} has {probe!r}")

    def get_tensor(self, local_name: str) -> torch.Tensor:
        resolved = self._resolve(local_name)
        if resolved is None:
            raise KeyError(f"checkpoint tensor not found: {self._prefix + local_name}")
        state, full = resolved
        return state.get_tensor(full)

    def _shard(
        self,
        local_name: str,
        dim: int,
        rank: int,
        world_size: int,
        *,
        contiguous: bool = True,
    ) -> torch.Tensor:
        return shard_tensor(
            self.get_tensor(local_name),
            dim,
            rank,
            world_size,
            name=self._prefix + local_name,
            contiguous=contiguous,
        )

    def _fuse(
        self,
        sources: Sequence[str | tuple[str, int, int]],
        rank: int | None,
        world_size: int | None,
    ) -> torch.Tensor:
        """Column-fuse (concat on dim 0) sharded projections into one packed tensor.

        Each source is either a name sharded by the shared ``(rank, world_size)``
        (gate_up), or a ``(name, rank, world_size)`` triple carrying its own split
        (GQA qkv: q on the attention split, k/v on the replicated-kv split).
        """
        shards = []
        for src in sources:
            name, r, w = (src, rank, world_size) if isinstance(src, str) else src
            if r is None or w is None:
                raise ValueError(f"fuse source {name!r} needs an explicit (rank, world_size)")
            shards.append(self._shard(name, 0, r, w, contiguous=False))
        return torch.cat(shards, dim=0)

    def load_fused(
        self,
        param: torch.Tensor,
        sources: Sequence[str | tuple[str, int, int]],
        name: str,
        rank: int | None = None,
        world_size: int | None = None,
    ) -> None:
        """Column-fuse ``sources`` and copy the packed tensor into ``param``.

        The fuse counterpart to :meth:`load_tensor`: the loader owns both the concat and
        the source-name reporting, where ``name`` is the fused target's checkpoint-relative
        name (e.g. ``"qkv_proj.weight"``). Each source is either a name sharded by the shared
        ``(rank, world_size)`` (gate_up), or a ``(name, rank, world_size)`` triple carrying
        its own split (GQA qkv: q on the attention split, k/v on the replicated-kv split).
        """
        _copy_parameter(param, self._fuse(sources, rank, world_size), self._prefix + name)

    def load_tensor(
        self,
        param: torch.Tensor,
        local_name: str,
        *,
        dim: int | None = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        """Copy one checkpoint tensor into ``param``.

        Copies the whole tensor, or its ``(rank, world_size)`` shard on ``dim`` when
        ``dim`` is given. The loader owns source-name reporting (``prefix + local_name``),
        so the name is named once instead of reconstructed at every call site.
        """
        value = (
            self._shard(local_name, dim, rank, world_size, contiguous=False)
            if dim is not None
            else self.get_tensor(local_name)
        )
        _copy_parameter(param, value, self._prefix + local_name)

    def copy(self, param: torch.Tensor, value: torch.Tensor, local_name: str) -> None:
        """Copy an already-materialized ``value`` into ``param``, owning the source name.

        The computed-value counterpart to :meth:`load_tensor`: packed/transposed/sharded
        values a caller builds are named once here (``prefix + local_name``) instead of
        rebuilding ``prefix + name`` at each call site.
        """
        _copy_parameter(param, value, self._prefix + local_name)

    def shard_value(
        self,
        value: torch.Tensor,
        local_name: str,
        dim: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        """Shard an already-materialized ``value`` on ``dim``, owning the source name.

        The shard counterpart to :meth:`copy`: a caller that built a value (a split or
        repacked chunk) shards it here so ``prefix + local_name`` is named once instead
        of hand-threading it into ``shard_tensor``. Returns the narrow view for cat/copy.
        """
        return shard_tensor(value, dim, rank, world_size, name=self._prefix + local_name, contiguous=False)


def _copy_parameter(
    param: torch.Tensor,
    value: torch.Tensor,
    source_name: str,
) -> None:
    if param.shape != value.shape:
        raise ValueError(
            f"checkpoint tensor {source_name} has shape {tuple(value.shape)}, expected {tuple(param.shape)}"
        )
    with torch.no_grad():
        param.copy_(value)
