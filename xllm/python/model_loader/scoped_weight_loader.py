# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Model-independent helpers for scoped checkpoint access."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import torch


class StateDictLike(Protocol):
    def has(self, name: str) -> bool: ...

    def get_tensor(self, name: str) -> torch.Tensor: ...


class ScopedWeightLoader:
    """A lightweight view over one or more checkpoint shards."""

    def __init__(
        self,
        state_dicts: Sequence[StateDictLike],
        prefix: str = "",
    ) -> None:
        self._state_dicts = state_dicts
        self._prefix = prefix

    @property
    def prefix(self) -> str:
        return self._prefix

    def with_prefix(self, prefix: str) -> ScopedWeightLoader:
        return ScopedWeightLoader(self._state_dicts, self._prefix + prefix)

    def _find(self, local_name: str) -> StateDictLike | None:
        name = self._prefix + local_name
        return next((state for state in self._state_dicts if state.has(name)), None)

    def has(self, local_name: str) -> bool:
        return self._find(local_name) is not None

    def tensor(self, local_name: str) -> torch.Tensor:
        state = self._find(local_name)
        name = self._prefix + local_name
        if state is None:
            raise KeyError(f"checkpoint tensor not found: {name}")
        return state.get_tensor(name)

    def shard(
        self,
        local_name: str,
        dim: int,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        value = self.tensor(local_name)
        if world_size == 1:
            return value
        if value.size(dim) % world_size:
            raise ValueError(f"cannot shard {self._prefix + local_name} across {world_size} ranks")
        return value.chunk(world_size, dim=dim)[rank].contiguous()

    def find_root(
        self,
        prefixes: Sequence[str],
        probe_name: str,
    ) -> ScopedWeightLoader:
        for prefix in prefixes:
            candidate = self.with_prefix(prefix)
            if candidate.has(probe_name):
                return candidate
        raise KeyError(f"checkpoint root containing {probe_name!r} was not found")


def copy_parameter(
    parameter: torch.nn.Parameter,
    value: torch.Tensor,
    source_name: str,
) -> None:
    if parameter.shape != value.shape:
        raise ValueError(
            f"checkpoint tensor {source_name} has shape {tuple(value.shape)}, expected {tuple(parameter.shape)}"
        )
    with torch.no_grad():
        parameter.copy_(value)
