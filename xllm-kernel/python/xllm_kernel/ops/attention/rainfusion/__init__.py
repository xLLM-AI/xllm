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

"""Reusable MiniMax-H3 Ref2VA RainFusion rf_v2 planning and kernel sources.

The ``_tilelang`` modules are the frozen A3 DSL kernels compiled by the isolated
Python TileLang subprocess and loaded from the package cache. ``plan`` and
``records`` expose the proven capacity, geometry and reorder-record semantics
for tests and model-side request preparation.
"""

from __future__ import annotations

from xllm_kernel.ops.attention.rainfusion.plan import (
    ATTENTION_VARIANT_KEY,
    PRODUCTION_DENSE_PREFIX_BLOCKS,
    PRODUCTION_DTYPE,
    PRODUCTION_GRID_H,
    PRODUCTION_GRID_T,
    PRODUCTION_GRID_W,
    PRODUCTION_NUM_BLOCKS,
    PRODUCTION_RECORD_COUNT,
    PRODUCTION_RECORDS_PADDED,
    PRODUCTION_SPARSE_END_BLOCK,
    PRODUCTION_TOPK_R,
    PRODUCTION_TOTAL_ROWS,
    PRODUCTION_VALID_ROWS,
    PRODUCTION_VIDEO_START,
    QUALIFIED_CAPACITY,
    SELECTOR_VARIANT_KEY,
    MinimaxH3RainfusionCapacity,
    MinimaxH3RainfusionDerived,
    MinimaxH3RainfusionGeometry,
    align_up,
    ceildiv,
    derive,
    derive_topk_r,
    is_qualified_profile,
    logical_fits_capacity,
    record_count,
    records_padded,
    semantic_records,
    unavailability_reason,
)
from xllm_kernel.ops.attention.rainfusion.records import build_reorder_records

__all__ = [
    "ATTENTION_VARIANT_KEY",
    "MinimaxH3RainfusionCapacity",
    "MinimaxH3RainfusionDerived",
    "MinimaxH3RainfusionGeometry",
    "PRODUCTION_DENSE_PREFIX_BLOCKS",
    "PRODUCTION_DTYPE",
    "PRODUCTION_GRID_H",
    "PRODUCTION_GRID_T",
    "PRODUCTION_GRID_W",
    "PRODUCTION_NUM_BLOCKS",
    "PRODUCTION_RECORD_COUNT",
    "PRODUCTION_RECORDS_PADDED",
    "PRODUCTION_SPARSE_END_BLOCK",
    "PRODUCTION_TOPK_R",
    "PRODUCTION_TOTAL_ROWS",
    "PRODUCTION_VALID_ROWS",
    "PRODUCTION_VIDEO_START",
    "QUALIFIED_CAPACITY",
    "SELECTOR_VARIANT_KEY",
    "align_up",
    "build_reorder_records",
    "ceildiv",
    "derive",
    "derive_topk_r",
    "is_qualified_profile",
    "logical_fits_capacity",
    "record_count",
    "records_padded",
    "semantic_records",
    "unavailability_reason",
]
