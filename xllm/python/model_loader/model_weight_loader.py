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

"""Checkpoint loading into named model parameters and buffers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import torch
import torch.nn as nn

from .scoped_weight_loader import ScopedWeightLoader, StateDictLike
from .sharding import shard_tensor


class ModelWeightLoader(ScopedWeightLoader):
    """Bind model parameter/buffer names and default TP splits to a tensor loader."""

    def __init__(
        self,
        model: nn.Module,
        state_dicts: Sequence[StateDictLike],
        tp_size: int,
        tp_rank: int,
        src_prefixes: Sequence[str] = ("",),
        name_aliases: Mapping[str, Sequence[str]] | Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self._model = model
        self._tensors_by_name: dict[str, torch.Tensor] = dict(model.named_parameters())
        self._tensors_by_name.update(model.named_buffers())
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        super().__init__(
            state_dicts,
            src_prefixes=src_prefixes,
            name_aliases=name_aliases,
            strip_src_prefixes=True,
        )

    def load_shard(self, name: str, dim: int, world: int | None = None, rank: int | None = None) -> torch.Tensor:
        return self.shard(self.get_tensor(name), dim=dim, world=world, rank=rank)

    def shard(
        self,
        t: torch.Tensor,
        dim: int,
        world: int | None = None,
        rank: int | None = None,
        contiguous: bool = True,
    ) -> torch.Tensor:
        return shard_tensor(
            t,
            dim,
            self.tp_rank if rank is None else rank,
            self.tp_size if world is None else world,
            contiguous=contiguous,
        )

    def copy_in(self, param_name: str, tensor: torch.Tensor) -> None:
        param = self._tensors_by_name.get(param_name)
        if param is None:
            try:
                param = self._model.get_parameter(param_name)
            except AttributeError:
                try:
                    param = self._model.get_buffer(param_name)
                except AttributeError as error:
                    raise KeyError(f"no parameter/buffer named {param_name}") from error
            if param is None:
                raise KeyError(f"no parameter/buffer named {param_name}")
            self._tensors_by_name[param_name] = param
        # copy_ streams the H2D transfer and dtype cast in one shot;
        # a prior tensor.to(device=...) would materialize a full device
        # temporary and spike peak HBM on stacked MoE experts.
        self.copy(param.data, tensor, param_name)

    def copy_replicated(self, name: str) -> None:
        self.copy_in(name, self.get_tensor(name))

    def copy_shard(self, name: str, dim: int) -> None:
        self.copy_in(name, self.load_shard(name, dim))

    def pack_gate_up(
        self,
        prefix: str,
        suffix: str = "weight",
        world: int | None = None,
        rank: int | None = None,
    ) -> torch.Tensor:
        gate = self.load_shard(prefix + "gate_proj." + suffix, 0, world=world, rank=rank)
        up = self.load_shard(prefix + "up_proj." + suffix, 0, world=world, rank=rank)
        return torch.cat([gate, up], dim=0)

    def load_projection(
        self,
        source: str,
        destination: str,
        suffixes: Sequence[str],
        shard_dims: Mapping[str, int] | None = None,
        *,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Copy projection tensors, allowing distinct checkpoint and module names."""
        dims = shard_dims or {}
        for suffix in suffixes:
            name = source + "." + suffix
            tensor = self.get_tensor(name)
            dim = dims.get(suffix)
            if dim is not None:
                tensor = self.shard(tensor, dim, world=world, rank=rank)
            self.copy_in(destination + "." + suffix, tensor)
