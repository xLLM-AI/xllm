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

"""Checkpoint tensor lookup and explicit tensor-destination loading."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

import torch

from .sharding import shard_tensor


class StateDictLike(Protocol):
    def has(self, name: str) -> bool: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class ScopedWeightLoader:
    """Load checkpoint tensors into explicit tensor destinations with exact shape checks."""

    def __init__(
        self,
        state_dicts: Sequence[StateDictLike],
        prefix: str = "",
        src_prefixes: Sequence[str] = ("",),
        name_aliases: Mapping[str, Sequence[str]] | Callable[[str], Sequence[str]] | None = None,
        *,
        strip_src_prefixes: bool = False,
    ) -> None:
        self._state_dicts = state_dicts
        self._prefix = prefix
        self._src_prefixes = tuple(src_prefixes)
        # Resolve aliases in order, then source prefixes and checkpoint shards.
        # Named model loaders also accept checkpoint keys with a source prefix removed.
        self._name_aliases = {} if name_aliases is None else name_aliases
        self._name_aliases_callable = callable(self._name_aliases)
        self._strip_src_prefixes = strip_src_prefixes

    def with_prefix(self, prefix: str) -> ScopedWeightLoader:
        return ScopedWeightLoader(
            self._state_dicts,
            self._prefix + prefix,
            src_prefixes=self._src_prefixes,
            name_aliases=self._name_aliases,
            strip_src_prefixes=self._strip_src_prefixes,
        )

    def _resolve(self, local_name: str) -> tuple[StateDictLike, str] | None:
        """First present ``(state_dict, resolved_name)`` for the scoped name,
        trying prepended source prefixes, then optional stripped prefixes per alias."""
        name = self._prefix + local_name
        aliases = self._name_aliases(name) if self._name_aliases_callable else self._name_aliases.get(name, (name,))
        for alias in aliases:
            for prefix in self._src_prefixes:
                full = prefix + alias
                for state in self._state_dicts:
                    if state.has(full):
                        return state, full
            if self._strip_src_prefixes:
                for prefix in self._src_prefixes:
                    if prefix and alias.startswith(prefix):
                        stripped = alias[len(prefix) :]
                        for state in self._state_dicts:
                            if state.has(stripped):
                                return state, stripped
        return None

    def find(self, local_name: str) -> StateDictLike | None:
        resolved = self._resolve(local_name)
        return resolved[0] if resolved is not None else None

    def has(self, local_name: str) -> bool:
        return self._resolve(local_name) is not None

    def bind_source_root(self, probe: str) -> ScopedWeightLoader:
        """Lock to the single ``src_prefix`` whose scope contains ``probe``.

        A multi-root loader tries every ``src_prefix`` per tensor; once the model
        root is chosen by a probe (e.g. ``embed_tokens.weight``), backbone weights
        must all resolve under that one root instead of silently mixing roots.
        Probing honors ``name_aliases`` and only accepts prepended prefixes; the
        returned loader never strips prefixes, so all reads stay under the chosen root.
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
        shards = []
        for src in sources:
            source_name, source_rank, source_world = (src, rank, world_size) if isinstance(src, str) else src
            if source_rank is None or source_world is None:
                raise ValueError(f"fuse source {source_name!r} needs an explicit (rank, world_size)")
            shards.append(self._shard(source_name, 0, source_rank, source_world, contiguous=False))
        _copy_parameter(param, torch.cat(shards, dim=0), self._prefix + name)

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
