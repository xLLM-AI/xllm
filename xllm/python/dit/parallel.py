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

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

_WEIGHT_TRANSFER_BATCH_BYTES = 256 * 1024**2


class StateDictShard(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor: ...
    def keys(self) -> list[str]: ...


@dataclass(frozen=True)
class TensorParallel:
    rank: int
    size: int
    reduce: Callable[[torch.Tensor], torch.Tensor]
    gather: Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class SequenceParallel:
    rank: int
    size: int
    gather: Callable[[torch.Tensor], torch.Tensor]
    exchange: Callable[[torch.Tensor, bool], torch.Tensor]


class ParallelLinear(nn.Module):
    def __init__(
        self,
        base: nn.Linear,
        group: TensorParallel,
        *,
        row: bool = False,
        gather: bool = False,
        packed: bool = False,
    ) -> None:
        super().__init__()
        self._group = group
        self._row = row
        self._gather = gather
        self._packed = packed
        axis = 1 if row else 0
        parts = 2 if packed else 1
        shape = list(base.weight.shape)
        if shape[axis] % (group.size * parts):
            raise ValueError(f"Cannot shard DiT projection {shape} across {group.size} ranks")
        shape[axis] //= group.size
        self.weight = nn.Parameter(torch.empty(shape, device="meta"), requires_grad=False)
        bias_size = base.out_features if row else shape[0]
        bias = nn.Parameter(torch.empty(bias_size, device="meta"), requires_grad=False)
        self.register_parameter("bias", bias if base.bias is not None else None)

    def shard(self, value: torch.Tensor, parameter: str) -> torch.Tensor:
        if self._row:
            return value if parameter == "bias" else value.chunk(self._group.size, dim=1)[self._group.rank]
        if self._packed:
            # The converted SwiGLU projection stores up and gate contiguously.
            return torch.cat([part.chunk(self._group.size)[self._group.rank] for part in value.chunk(2)])
        return value.chunk(self._group.size)[self._group.rank]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._row:
            result = self._group.reduce(F.linear(hidden_states, self.weight))
            return result if self.bias is None else result + self.bias
        result = F.linear(hidden_states, self.weight, self.bias)
        return self._group.gather(result) if self._gather else result


class ParallelRMSNorm(nn.Module):
    """Normalize across the full hidden dimension of a TP-sharded projection."""

    def __init__(self, base: nn.RMSNorm, group: TensorParallel) -> None:
        super().__init__()
        self._group = group
        self._eps = base.eps
        self.weight = nn.Parameter(torch.empty(base.weight.numel() // group.size, device="meta"), requires_grad=False)

    def shard(self, value: torch.Tensor, parameter: str) -> torch.Tensor:
        return value.chunk(self._group.size)[self._group.rank]

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        values = states.float()
        variance = self._group.reduce(values.square().mean(dim=-1, keepdim=True)) / self._group.size
        normalized = values * torch.rsqrt(variance + self._eps)
        return (normalized * self.weight.float()).to(states.dtype)


def load_component_weights(
    model: nn.Module,
    shards: Sequence[StateDictShard],
    device: torch.device,
    dtype_for: Callable[[str], torch.dtype],
) -> None:
    missing = set(model.state_dict())
    # Transformers may store a tied embedding only once in safetensors.
    aliases: dict[int, list[str]] = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(name)
    tied_names = {name: names for names in aliases.values() for name in names}
    pending_transfers: list[torch.Tensor] = []
    pending_bytes = 0
    for shard in shards:
        for name in shard.keys():
            if name not in missing:
                continue
            parent, _, key = name.rpartition(".")
            module = model.get_submodule(parent)
            target = getattr(module, key)
            value = shard.get_tensor(name)
            if isinstance(module, (ParallelLinear, ParallelRMSNorm)):
                value = module.shard(value, key)
            if target.shape != value.shape:
                raise ValueError(f"DiT weight {name} has shape {tuple(value.shape)}, expected {tuple(target.shape)}")
            if not value.is_floating_point() and target.is_floating_point():
                raise ValueError(f"DiT weight {name} must be a dense floating point tensor")
            dtype = dtype_for(name) if value.is_floating_point() else target.dtype
            if device.type == "npu" and value.device.type == "cpu":
                # Pack the local shard on the host and enqueue the transfer.
                # Keep the pinned staging buffer alive until its batch completes.
                staging = torch.empty(value.shape, dtype=dtype, device="cpu", pin_memory=True)
                staging.copy_(value)
                value = staging.to(device=device, non_blocking=True)
                pending_transfers.append(staging)
                pending_bytes += staging.nbytes
                if pending_bytes >= _WEIGHT_TRANSFER_BATCH_BYTES:
                    torch.npu.current_stream(device).synchronize()
                    pending_transfers.clear()
                    pending_bytes = 0
            else:
                value = value.to(device=device, dtype=dtype, memory_format=torch.contiguous_format)
            if isinstance(target, nn.Parameter):
                value = nn.Parameter(value, requires_grad=False)
            for alias in tied_names.get(name, [name]):
                parent, _, key = alias.rpartition(".")
                setattr(model.get_submodule(parent), key, value)
                missing.discard(alias)
    if pending_transfers:
        torch.npu.current_stream(device).synchronize()
    if missing:
        raise ValueError(f"Missing DiT weights in {type(model).__name__}: {', '.join(sorted(missing)[:8])}")
    model.eval()
