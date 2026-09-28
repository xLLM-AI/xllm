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

"""Transparent linear wrapper: fp or W8A8, resolved at load time.

Mirrors libtorch #1426's resolved_weight_quant_method_ pattern in pure Python.
The model graph constructs ``QLinear`` everywhere; whether it actually quantizes
is decided at ``load_weights`` by probing the checkpoint for quant tensors
(``deq_scale`` / ``weight_scale``). Forward is identical either way.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
        kind: str = "static",
        row_parallel: bool = False,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.kind = kind
        self.row_parallel = row_parallel
        self.use_w8a8: bool | None = None
        # fp weight (used iff use_w8a8 is False)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)
        self._w8a8: nn.Module | None = None  # built in Task 2

    def resolve_quant(self, has_quant_tensors: bool) -> None:
        self.use_w8a8 = bool(has_quant_tensors)
        if self.use_w8a8:
            self._build_w8a8()

    def _build_w8a8(self) -> None:
        from xllm.python.models.deepseek_v32 import (
            W8A8DynamicLinear,
            W8A8StaticLinear,
        )

        device = self.weight.device
        cls = W8A8DynamicLinear if self.kind == "dynamic" else W8A8StaticLinear
        if cls is W8A8StaticLinear:
            self._w8a8 = cls(self.in_features, self.out_features, device, row_parallel=self.row_parallel)
        else:
            self._w8a8 = cls(self.in_features, self.out_features, device)

    def load_weight(self, tensor: torch.Tensor) -> None:
        self.weight.data.copy_(tensor.to(dtype=self.weight.dtype, device=self.weight.device))

    def load_w8a8(self, loader, prefix: str, proj: str, shard_dims=None) -> None:
        """Load w8a8 tensors into this QLinear's ``_w8a8`` submodule.

        ``prefix + proj`` is this module's dotted path inside the model (the
        caller passes the path up to the projection attribute, e.g. ``attn`` +
        ``o_proj``). The loader targets ``<prefix+proj>._w8a8.<suffix>`` so the
        int8 weights land on the submodule ``forward`` actually reads, not this
        QLinear's fp ``weight`` slot.
        """
        assert self._w8a8 is not None, "resolve_quant(True) before load_w8a8"
        loader.load_w8a8_into_qlinear(prefix + proj, prefix, proj, shard_dims)

    def process_weights_after_loading(self) -> None:
        if self.use_w8a8:
            self._w8a8.process_weights_after_loading()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_w8a8 is None:
            raise RuntimeError("QLinear.forward before resolve_quant")
        if self.use_w8a8:
            return self._w8a8(x)
        return torch.nn.functional.linear(x, self.weight, self.bias)


class QLinearWeightLoader:
    """Probe-driven fp/w8a8 load router + TP shard, reusing W8A8WeightLoader."""

    def __init__(self, model: nn.Module, state_dicts, tp_size: int, tp_rank: int):
        from xllm.python.models.glm5_next_weight import W8A8WeightLoader

        self._w8 = W8A8WeightLoader(model, state_dicts, tp_size, tp_rank)
        self.tp_size = tp_size
        self.tp_rank = tp_rank

    # expose the underlying helpers glm5_next's load_weights still needs directly
    def find(self, name):
        return self._w8.find(name)

    def load_tensor(self, name):
        return self._w8.load_tensor(name)

    def shard(self, t, dim, world=None, rank=None):
        return self._w8.shard(t, dim, world, rank)

    def copy_in(self, name, tensor):
        self._w8.copy_in(name, tensor)

    def probe_quant(self, prefix: str, proj: str) -> bool:
        """True iff the checkpoint carries w8a8 tensors for ``prefix+proj``."""
        return (
            self._w8.find(f"{prefix}{proj}.deq_scale") is not None
            or self._w8.find(f"{prefix}{proj}.weight_scale") is not None
        )

    def load_fp(self, name: str, dim=None) -> None:
        t = self.load_tensor(name)
        if dim is not None:
            t = self.shard(t, dim=dim)
        self.copy_in(name, t)

    def load_w8a8_into_qlinear(
        self,
        qlinear_name: str,
        prefix: str,
        proj: str,
        shard_dims=None,
    ) -> None:
        """Load static W8A8 projection tensors into a QLinear's ``_w8a8`` module.

        Source checkpoint keys are ``prefix + proj + ".<suffix>"`` (the
        checkpoint names the QLinear directly, e.g. ``attn.o_proj.<suffix>``).
        The tensors are written to the *int8* submodule at qualified name
        ``qlinear_name + "._w8a8.<suffix>"`` (``qlinear_name`` is the QLinear's
        own dotted path, e.g. ``model.layers.0.self_attn.o_proj``). QLinear's
        ``forward`` reads ``self._w8a8``, so that int8 slot is the only one that
        must be populated. Unlike ``load_w8a8_a`` (which targets the module's
        own name), this lands on the ``_w8a8`` path segment.
        """
        for suffix in ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset"):
            key = f"{prefix}{proj}.{suffix}"
            if self.find(key) is None:
                continue  # optional suffix (e.g. dynamic uses weight_scale)
            t = self.load_tensor(key)
            dim = (shard_dims or {}).get(suffix)
            if dim is not None:
                t = self.shard(t, dim=dim)
            self.copy_in(f"{qlinear_name}._w8a8.{suffix}", t)

    def load_w8a8_mlp_into_qlinear(self, mlp_pfx: str) -> None:
        """Load dynamic W8A8 MLP tensors into the QLinear-wrapped ``_w8a8``.

        Mirrors ``W8A8WeightLoader.load_w8a8_b`` (cat gate+up on dim 0, down
        shard dim 1) but writes into ``gate_up_proj._w8a8`` / ``down_proj._w8a8``
        instead of the QLinear fp slots. ``mlp_pfx`` is the dotted MLP path, e.g.
        ``model.layers.0.mlp.``.
        """
        g = mlp_pfx + "gate_proj."
        u = mlp_pfx + "up_proj."
        d = mlp_pfx + "down_proj."
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight",
            torch.cat(
                [self.shard(self.load_tensor(g + "weight"), 0), self.shard(self.load_tensor(u + "weight"), 0)], dim=0
            ).contiguous(),
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight_scale",
            torch.cat(
                [
                    self.shard(self.load_tensor(g + "weight_scale"), 0),
                    self.shard(self.load_tensor(u + "weight_scale"), 0),
                ],
                dim=0,
            ).contiguous(),
        )
        self.copy_in(
            mlp_pfx + "gate_up_proj._w8a8.weight_offset",
            torch.cat(
                [
                    self.shard(self.load_tensor(g + "weight_offset"), 0),
                    self.shard(self.load_tensor(u + "weight_offset"), 0),
                ],
                dim=0,
            ).contiguous(),
        )
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight", self.shard(self.load_tensor(d + "weight"), dim=1))
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight_scale", self.load_tensor(d + "weight_scale"))
        self.copy_in(mlp_pfx + "down_proj._w8a8.weight_offset", self.load_tensor(d + "weight_offset"))
