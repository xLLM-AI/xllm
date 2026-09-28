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
"""W8A8 weight-loading helpers for GLM-5.3-Flash.

This module factors out the checkpoint-tensor lookup / TP sharding / W8A8
projection- and MLP-weight packing used by GLM-5.3-Flash's ``load_weights``.
It is intentionally a standalone helper (not the shared
``xllm.python.models.weight_utils.W8A8WeightLoader`` used by Deepseek-V3):
GLM-5.3-Flash's W8A8 layers need a ``find`` lookup and GLM-specific
``load_w8a8_a`` / ``load_fused_w8a8_a`` / ``load_w8a8_b`` packers that the
shared loader does not provide, while Deepseek-V3's ``load_weights`` relies
on the shared loader's ``load_w8a8_projection`` / ``load_fused_w8a8_projection``
which this helper does not need. Keeping the two classes in separate modules
avoids a name shadow that would break one or the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from xllm_weight_loader import StateDict


class W8A8WeightLoader:
    """Shared W8A8 weight-loading helpers for a model's ``load_weights``.

    Owns the byte-identical checkpoint tensor lookup / TP sharding / W8A8
    projection- and MLP-weight packing used by every W8A8 DSA model. The
    model-specific per-layer loop (which projections/experts to load and in
    what order) stays in each model; only the mechanics live here.
    """

    def __init__(
        self,
        model: nn.Module,
        state_dicts: list[StateDict],
        tp_size: int,
        tp_rank: int,
    ) -> None:
        self._model = model
        self._params_by_name = dict(model.named_parameters())
        self._buffers_by_name = dict(model.named_buffers())
        self._state_dicts = state_dicts
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    def find(self, name: str) -> Optional[StateDict]:
        for sd in self._state_dicts:
            if sd.has(name):
                return sd
        return None

    def load_tensor(self, name: str) -> torch.Tensor:
        sd = self.find(name)
        assert sd is not None, f"checkpoint tensor not found: {name}"
        return sd.get_tensor(name)

    def shard(
        self,
        t: torch.Tensor,
        dim: int,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        world = self.tp_size if world is None else world
        rank = self.tp_rank if rank is None else rank
        if world <= 1:
            return t
        cs = t.size(dim) // world
        return t.narrow(dim, rank * cs, cs).contiguous()

    def copy_in(self, param_name: str, tensor: torch.Tensor) -> None:
        p = self._params_by_name.get(param_name)
        if p is None:
            p = self._buffers_by_name.get(param_name)
        if p is None:
            # resolve_quant / lazy construction (e.g. QLinear._w8a8, bf16
            # experts) may add params AFTER _params_by_name was cached at
            # __init__. Fall back to a live model walk and cache the result.
            live = dict(self._model.named_parameters())
            live.update(dict(self._model.named_buffers()))
            self._params_by_name.update(live)
            self._buffers_by_name.update(live)
            p = self._params_by_name.get(param_name)
            if p is None:
                p = self._buffers_by_name.get(param_name)
        assert p is not None, f"no parameter/buffer named {param_name}"
        p.data.copy_(tensor)  # copy_ streams H2D + dtype/device cast in one shot

    def load_shard(self, name: str, dim: int, world: Optional[int] = None, rank: Optional[int] = None) -> torch.Tensor:
        return self.shard(self.load_tensor(name), dim=dim, world=world, rank=rank)

    def copy_replicated(self, name: str) -> None:
        """Load ``name`` from checkpoint (no shard) and copy into the same-named param."""
        self.copy_in(name, self.load_tensor(name))

    def copy_shard(self, name: str, dim: int) -> None:
        """Load ``name``, shard on ``dim`` by the loader's TP, and copy into the same-named param."""
        self.copy_in(name, self.load_shard(name, dim))

    def pack_gate_up(
        self,
        prefix: str,
        suffix: str = "weight",
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> torch.Tensor:
        gate = self.load_shard(prefix + "gate_proj." + suffix, 0, world=world, rank=rank)
        up = self.load_shard(prefix + "up_proj." + suffix, 0, world=world, rank=rank)
        return torch.cat([gate, up], dim=0)

    def assert_symmetric_int8(self, prefix: str, projs: Sequence[str]) -> None:
        """Assert ``weight_offset`` is all-zero for each ``proj`` under ``prefix``."""
        for proj in projs:
            if self.load_tensor(prefix + proj + ".weight_offset").any():
                raise ValueError(f"{prefix}{proj} requires zero offset (symmetric int8)")

    def load_w8a8_down(
        self,
        prefix: str,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """down_proj (weight sharded on dim 1, replicated weight_scale)."""
        return (
            self.load_shard(prefix + "down_proj.weight", 1, world=world, rank=rank),
            self.load_tensor(prefix + "down_proj.weight_scale"),
        )

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: Optional[int] = None,
        rank: Optional[int] = None,
    ) -> None:
        """Fill fused gate_up + down projections of one dense/shared W8A8 MLP block."""
        for suffix in ("weight", "weight_scale", "weight_offset"):
            self.copy_in(
                prefix + "gate_up_proj." + suffix,
                self.pack_gate_up(prefix, suffix, world=world, rank=rank),
            )
        dw, ds = self.load_w8a8_down(prefix, world=world, rank=rank)
        self.copy_in(prefix + "down_proj.weight", dw)
        self.copy_in(prefix + "down_proj.weight_scale", ds)
        self.copy_in(
            prefix + "down_proj.weight_offset",
            self.load_tensor(prefix + "down_proj.weight_offset"),
        )

    def load_w8a8_a(self, prefix: str, proj: str, shard_dims: Optional[dict] = None) -> None:
        for suffix in ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset"):
            t = self.load_tensor(prefix + proj + "." + suffix)
            dim = (shard_dims or {}).get(suffix)
            if dim is not None:
                t = self.shard(t, dim=dim)
            self.copy_in(prefix + proj + "." + suffix, t)

    def load_fused_w8a8_a(
        self,
        prefix: str,
        target_proj: str,
        source_projs: tuple[str, ...],
    ) -> None:
        for suffix in ("weight", "deq_scale", "quant_bias"):
            tensors = [self.load_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            self.copy_in(
                prefix + target_proj + "." + suffix,
                torch.cat(tensors, dim=0).contiguous(),
            )

        for suffix in ("input_scale", "input_offset"):
            tensors = [self.load_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            reference = tensors[0]
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                names = ", ".join(source_projs)
                raise ValueError(f"{prefix}{names} must share {suffix} for fused W8A8")
            self.copy_in(prefix + target_proj + "." + suffix, reference)

    def load_w8a8_b(self, mlp_pfx: str) -> None:
        gw = self.load_tensor(mlp_pfx + "gate_proj.weight")
        gs = self.load_tensor(mlp_pfx + "gate_proj.weight_scale")
        go = self.load_tensor(mlp_pfx + "gate_proj.weight_offset")
        uw = self.load_tensor(mlp_pfx + "up_proj.weight")
        us = self.load_tensor(mlp_pfx + "up_proj.weight_scale")
        uo = self.load_tensor(mlp_pfx + "up_proj.weight_offset")
        self.copy_in(
            mlp_pfx + "gate_up_proj.weight", torch.cat([self.shard(gw, 0), self.shard(uw, 0)], dim=0).contiguous()
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj.weight_scale", torch.cat([self.shard(gs, 0), self.shard(us, 0)], dim=0).contiguous()
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj.weight_offset",
            torch.cat([self.shard(go, 0), self.shard(uo, 0)], dim=0).contiguous(),
        )
        self.copy_in(mlp_pfx + "down_proj.weight", self.shard(self.load_tensor(mlp_pfx + "down_proj.weight"), dim=1))
        self.copy_in(mlp_pfx + "down_proj.weight_scale", self.load_tensor(mlp_pfx + "down_proj.weight_scale"))
        self.copy_in(mlp_pfx + "down_proj.weight_offset", self.load_tensor(mlp_pfx + "down_proj.weight_offset"))
