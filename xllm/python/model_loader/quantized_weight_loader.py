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

"""W8A8 packing and destination adapters for quantized projections."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .model_weight_loader import ModelWeightLoader


class W8A8WeightLoader(ModelWeightLoader):
    """W8A8 projection/MLP weight packing on top of the generic ModelWeightLoader."""

    def w8a8_projection_uses_dynamic_activation(self, prefix: str, proj: str) -> bool:
        """Identify the activation quantization format from checkpoint keys."""
        has_dynamic = self.has(prefix + proj + ".weight_scale")
        has_static = self.has(prefix + proj + ".deq_scale")
        if has_dynamic == has_static:
            formats = "both" if has_dynamic else "neither"
            raise ValueError(f"{prefix}{proj} checkpoint contains {formats} static/dynamic W8A8 formats")
        return has_dynamic

    def load_compatible_w8a8_projection(
        self,
        prefix: str,
        proj: str,
        shard_dims: dict[str, int] | None = None,
        *,
        dynamic_activation: bool,
        world: int | None = None,
        rank: int | None = None,
    ) -> None:
        """Load the W8A8 format already selected by the destination module."""

        suffixes = (
            ("weight", "weight_scale", "weight_offset")
            if dynamic_activation
            else ("weight", "deq_scale", "quant_bias", "input_scale", "input_offset")
        )
        self.load_projection(
            prefix + proj,
            prefix + proj,
            suffixes,
            shard_dims,
            world=world,
            rank=rank,
        )

    def load_fused_w8a8_projection(
        self,
        prefix: str,
        target_proj: str,
        source_projs: tuple[str, ...],
    ) -> None:
        """Concat W8A8 ``source_projs`` into one fused ``target_proj``; asserts shared input scale/offset."""
        for suffix in ("weight", "deq_scale", "quant_bias"):
            tensors = [self.get_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            self.copy_in(prefix + target_proj + "." + suffix, torch.cat(tensors, dim=0))

        for suffix in ("input_scale", "input_offset"):
            tensors = [self.get_tensor(prefix + proj + "." + suffix) for proj in source_projs]
            reference = tensors[0]
            if any(not torch.equal(reference, tensor) for tensor in tensors[1:]):
                names = ", ".join(source_projs)
                raise ValueError(f"{prefix}{names} must share {suffix} for fused W8A8")
            self.copy_in(prefix + target_proj + "." + suffix, reference)

    def assert_symmetric_int8(self, prefix: str, projs: Sequence[str]) -> None:
        """Assert ``weight_offset`` is all-zero for each ``proj`` under ``prefix``."""
        for proj in projs:
            if self.get_tensor(prefix + proj + ".weight_offset").any():
                raise ValueError(f"{prefix}{proj} requires zero offset (symmetric int8)")

    def load_w8a8_down(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """down_proj (weight sharded on dim 1, replicated weight_scale)."""
        return (
            self.load_shard(prefix + "down_proj.weight", 1, world=world, rank=rank),
            self.get_tensor(prefix + "down_proj.weight_scale"),
        )

    def load_w8a8_mlp(
        self,
        prefix: str,
        world: int | None = None,
        rank: int | None = None,
        *,
        destination_suffix: str = "",
    ) -> None:
        """Load one W8A8 MLP, optionally into wrapped projection buffers."""
        for suffix in ("weight", "weight_scale", "weight_offset"):
            self.copy_in(
                prefix + "gate_up_proj" + destination_suffix + "." + suffix,
                self.pack_gate_up(prefix, suffix, world=world, rank=rank),
            )
        dw, ds = self.load_w8a8_down(prefix, world=world, rank=rank)
        self.copy_in(prefix + "down_proj" + destination_suffix + ".weight", dw)
        self.copy_in(prefix + "down_proj" + destination_suffix + ".weight_scale", ds)
        # W8A8DynamicLinear.process_weights_after_loading asserts
        # weight_offset==0 on the down_proj buffer, which stays torch.empty
        # (garbage) unless we copy the checkpoint's offset in. Dense/shared
        # MLPs carry the offset just like gate_up_proj.
        self.copy_in(
            prefix + "down_proj" + destination_suffix + ".weight_offset",
            self.get_tensor(prefix + "down_proj.weight_offset"),
        )


class QLinearWeightLoader(W8A8WeightLoader):
    def probe_quant(self, prefix: str, proj: str) -> bool:
        """True iff the checkpoint carries w8a8 tensors for ``prefix+proj``."""
        return self.has(f"{prefix}{proj}.deq_scale") or self.has(f"{prefix}{proj}.weight_scale")
