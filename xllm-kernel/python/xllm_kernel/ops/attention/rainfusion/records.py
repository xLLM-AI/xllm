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

"""Constant rf_v2 worker-major reorder record table for MiniMax-H3 Ref2VA."""

from __future__ import annotations

from typing import TYPE_CHECKING

from xllm_kernel.ops.attention.rainfusion import plan

if TYPE_CHECKING:
    import torch


def build_reorder_records(
    geometry: plan.MinimaxH3RainfusionGeometry,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the constant int32 [records_padded, 4] worker-major table.

    Worker ``w`` owns the contiguous slot range
    ``[w * records_per_worker, (w + 1) * records_per_worker)`` so each AIV
    worker preloads its own records with one DMA. Unused slots stay zero; the
    active extent is ``record_count`` and never the padded extent.
    """
    import torch

    derived = plan.derive(geometry)
    semantic = plan.semantic_records(geometry, derived)
    count = len(semantic)
    records_per_worker = plan.ceildiv(count, plan.WORKERS)
    padded = plan.WORKERS * records_per_worker
    flat = [0] * (padded * 4)
    for index, (src, dst, rows, flags) in enumerate(semantic):
        slot = (index % plan.WORKERS) * records_per_worker + index // plan.WORKERS
        base = slot * 4
        flat[base : base + 4] = (src, dst, rows, flags)
    table = torch.tensor(flat, dtype=torch.int32).view(padded, 4)
    return table.to(device=device)
