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

"""Python TileLang MiniMax-H3 Ref2VA RainFusion interface for the DiT path.

The frozen TileLang DSL kernels own geometry, capacity checks and the two-launch
selector/attention hot path. This module only adapts Ulysses-local Q/K/V to the
packed TND storage extent, holds the per-request workspace and returns the
original sequence view. The model owns the step/layer policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from xllm_kernel.ops.attention.rainfusion import plan

if TYPE_CHECKING:
    from xllm_kernel.ops.attention.rainfusion.tilelang import RainfusionKernels, RainfusionWorkspace

_HEADS = plan.HEADS
_HEAD_DIM = plan.HEAD_DIM
_ROW_WIDTH = plan.ROW_WIDTH


@dataclass(frozen=True)
class MinimaxH3RainfusionPolicy:
    """Native rf_v2 step/layer window; the model owns where it is applied."""

    name: str
    step_begin: int
    step_end: int
    layer_begin: int
    layer_end: int

    def includes_step(self, step: int) -> bool:
        return self.step_begin <= step < self.step_end

    def includes_layer(self, layer: int) -> bool:
        return self.layer_begin <= layer < self.layer_end


SPARSE_MIDDLE_STEPS = MinimaxH3RainfusionPolicy("sparse-middle-steps", 8, 36, 0, 50)


@dataclass
class MinimaxH3RainfusionRequest:
    """One prepared capacity workspace, reused by every active block."""

    geometry: plan.MinimaxH3RainfusionGeometry
    workspace: RainfusionWorkspace
    kernels: RainfusionKernels
    device: torch.device
    record_count: int
    records_padded: int
    output: torch.Tensor
    policy: MinimaxH3RainfusionPolicy
    selector_launches: int = 0
    attention_launches: int = 0
    current_step: int = 0
    sparse_steps: list[int] = field(default_factory=list)

    def begin_step(self, step: int) -> None:
        self.current_step = step

    def attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        live_rows: int,
    ) -> torch.Tensor:
        """Run the JIT selector and attention on packed local Q/K/V."""
        sp_rows = query.size(1)
        if (
            query.shape != key.shape
            or query.shape != value.shape
            or query.size(0) != 1
            or query.size(2) != _HEADS
            or query.size(3) != _HEAD_DIM
        ):
            raise ValueError("MiniMax-H3 RainFusion requires [1, rows, 4, 128] Q/K/V")
        if live_rows != self.geometry.valid_rows:
            raise ValueError("MiniMax-H3 RainFusion live rows differ from the prepared geometry")
        if sp_rows < live_rows or sp_rows > self.geometry.total_rows:
            raise ValueError("MiniMax-H3 RainFusion sequence extent is outside the prepared storage")
        if query.device != key.device or query.device != value.device:
            raise ValueError("MiniMax-H3 RainFusion requires Q/K/V on one device")
        if query.device != self.device:
            raise ValueError("MiniMax-H3 RainFusion request workspace is on a different device")
        if query.dtype is not torch.bfloat16 or key.dtype is not torch.bfloat16 or value.dtype is not torch.bfloat16:
            raise ValueError("MiniMax-H3 RainFusion requires BF16 Q/K/V")

        def pack(tensor: torch.Tensor) -> torch.Tensor:
            if sp_rows == self.geometry.total_rows and tensor.is_contiguous():
                return tensor.view(sp_rows, _ROW_WIDTH)
            storage = torch.empty(
                (self.geometry.total_rows, _HEADS, _HEAD_DIM),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            storage.narrow(0, 0, live_rows).copy_(tensor.squeeze(0).narrow(0, 0, live_rows))
            storage[live_rows:].zero_()
            return storage.view(self.geometry.total_rows, _ROW_WIDTH)

        query_packed = pack(query)
        key_packed = pack(key)
        value_packed = pack(value)
        with torch.npu.device(query.device):
            self.kernels.run_selector(self.geometry, query_packed, key_packed, value_packed, self.workspace)
            self.selector_launches += 1
            self.kernels.run_attention(self.geometry, self.workspace, self.output)
            self.attention_launches += 1
        if not self.sparse_steps or self.sparse_steps[-1] != self.current_step:
            self.sparse_steps.append(self.current_step)
        return self.output.narrow(0, 0, sp_rows).view(query.shape)


def _geometry(
    valid_rows: int, video_start: int, grid_t: int, grid_h: int, grid_w: int
) -> plan.MinimaxH3RainfusionGeometry:
    """Build the native request geometry, including the derived Top-R budget."""
    return plan.MinimaxH3RainfusionGeometry(
        total_rows=plan.align_up(valid_rows, plan.KV_PACK_ROWS),
        valid_rows=valid_rows,
        video_start=video_start,
        grid_t=grid_t,
        grid_h=grid_h,
        grid_w=grid_w,
        topk_r=plan.derive_topk_r(plan.ceildiv(valid_rows, plan.BLOCK)),
    )


class MinimaxH3RainfusionBackend:
    """Prepare and expose the optional JIT RainFusion path."""

    def __init__(self, *, enabled: bool = True) -> None:
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def prepare_specialization(self, device: torch.device) -> None:
        """Compile both production specializations before the model runs."""
        if not self._enabled:
            return
        from xllm_kernel.ops.attention.rainfusion.tilelang import prepare_for_device

        prepare_for_device(device)

    def available(
        self,
        valid_rows: int,
        video_start: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
    ) -> bool:
        """Return whether the qualified capacity and geometry can serve the path."""
        if not self._enabled:
            return False
        return plan.unavailability_reason(_geometry(valid_rows, video_start, grid_t, grid_h, grid_w)) == ""

    def unavailability_reason(
        self,
        valid_rows: int,
        video_start: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
    ) -> str:
        """Return the dense-fallback reason, or "" when servable."""
        if not self._enabled:
            return "disabled"
        return plan.unavailability_reason(_geometry(valid_rows, video_start, grid_t, grid_h, grid_w))

    def prepare(
        self,
        valid_rows: int,
        video_start: int,
        grid_t: int,
        grid_h: int,
        grid_w: int,
        policy: MinimaxH3RainfusionPolicy,
        device: torch.device,
    ) -> MinimaxH3RainfusionRequest | None:
        """Compile once, then allocate the capacity workspace for one request."""
        if not self._enabled:
            return None
        geometry = _geometry(valid_rows, video_start, grid_t, grid_h, grid_w)
        if plan.unavailability_reason(geometry):
            return None
        if device.type not in ("npu", "privateuseone"):
            raise ValueError("MiniMax-H3 RainFusion requires an NPU device")

        from xllm_kernel.ops.attention.rainfusion.tilelang import allocate_workspace, prepare_for_device

        kernels = prepare_for_device(device)
        workspace = allocate_workspace(geometry, device)
        output = torch.zeros(
            (geometry.total_rows, _ROW_WIDTH),
            dtype=torch.bfloat16,
            device=device,
        )
        return MinimaxH3RainfusionRequest(
            geometry=geometry,
            workspace=workspace,
            kernels=kernels,
            device=device,
            record_count=workspace.record_count,
            records_padded=workspace.records_padded,
            output=output,
            policy=policy,
        )


__all__ = [
    "MinimaxH3RainfusionBackend",
    "MinimaxH3RainfusionPolicy",
    "MinimaxH3RainfusionRequest",
    "SPARSE_MIDDLE_STEPS",
]
