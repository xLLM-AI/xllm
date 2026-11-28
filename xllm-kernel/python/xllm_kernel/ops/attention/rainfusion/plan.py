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

"""SDK-free RainFusion capacity, geometry and reorder-record planning.

These values mirror the frozen native Ascend A3 rf_v2 kernel pair. The module
exposes the proven plan/count semantics only: the model owns the step and layer
policy, the device buffers and the model's execution lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass

BLOCK = 128
HEADS = 4
HEAD_DIM = 128
ROW_WIDTH = HEADS * HEAD_DIM
NUM_CORES = 24
VEC_NUM = 2
WORKERS = NUM_CORES * VEC_NUM
CHUNK_ROWS = 64
M_TILE = 16
PLAN_ALIGN = 8
STACK_BLOCKS = 4
TASK_SLOTS = 3
RING_SLOTS = 2
KV_PACK_ROWS = 256
POOL_ALIGN = 16
N_TILE = 128

FLAG_FIRST_FRAME = 1
FLAG_DENSE = 2
FLAG_TAIL = 4

SPARSITY_PERCENT = 20


@dataclass(frozen=True)
class MinimaxH3RainfusionCapacity:
    """Physical capacity one compiled kernel pair is sized from."""

    total_rows: int
    num_blocks: int
    record_count: int
    records_padded: int
    topk_r: int
    plan_width: int
    q_pool_pad: int
    k_pool_pad: int


QUALIFIED_CAPACITY = MinimaxH3RainfusionCapacity(
    total_rows=61952,
    num_blocks=483,
    record_count=5568,
    records_padded=5568,
    topk_r=97,
    plan_width=488,
    q_pool_pad=480,
    k_pool_pad=496,
)

PRODUCTION_TOTAL_ROWS = 60160
PRODUCTION_VALID_ROWS = 60132
PRODUCTION_VIDEO_START = 22836
PRODUCTION_GRID_T = 37
PRODUCTION_GRID_H = 24
PRODUCTION_GRID_W = 42
PRODUCTION_RECORD_COUNT = 5557
PRODUCTION_TOPK_R = 94
PRODUCTION_NUM_BLOCKS = 470
PRODUCTION_RECORDS_PADDED = 5568
PRODUCTION_DENSE_PREFIX_BLOCKS = 8
PRODUCTION_SPARSE_END_BLOCK = 291
PRODUCTION_DTYPE = "bfloat16"

SELECTOR_VARIANT_KEY = "cap_ctr61952_ctb483_crc5568_ctk97_abi2"
ATTENTION_VARIANT_KEY = "cap_ctr61952_cnb483_crc5568_crp5568_bf16_abi2"


@dataclass(frozen=True)
class MinimaxH3RainfusionGeometry:
    """Logical geometry requested by one packed Ref2VA local-U8 rank."""

    total_rows: int = 0
    valid_rows: int = 0
    video_start: int = 0
    grid_t: int = 0
    grid_h: int = 0
    grid_w: int = 0
    topk_r: int = 0


@dataclass(frozen=True)
class MinimaxH3RainfusionDerived:
    """Host quantities derived from one logical geometry."""

    video_rows: int = 0
    video_end: int = 0
    total_blocks: int = 0
    tail_rows: int = 0
    ff_rows: int = 0
    ff_blocks: int = 0
    video_full_blocks: int = 0
    num_sq: int = 0
    q_pool_pad: int = 0
    k_pool_pad: int = 0
    plan_width: int = 0


def ceildiv(value: int, divisor: int) -> int:
    """Return ceil(value / divisor) for a positive divisor."""
    return (value + divisor - 1) // divisor


def align_up(value: int, alignment: int) -> int:
    """Round value up to a positive alignment."""
    return ceildiv(value, alignment) * alignment


def derive_topk_r(total_blocks: int) -> int:
    """Return the fixed fraction of KV blocks kept by the rf_v2 plan."""
    return (total_blocks * SPARSITY_PERCENT + 99) // 100


def derive(geometry: MinimaxH3RainfusionGeometry) -> MinimaxH3RainfusionDerived:
    """Derive the exact plan geometry, rejecting an unservable request."""
    if (
        geometry.total_rows <= 0
        or geometry.valid_rows <= 0
        or geometry.valid_rows > geometry.total_rows
        or geometry.total_rows % BLOCK
        or geometry.total_rows % KV_PACK_ROWS
        or geometry.grid_t <= 0
        or geometry.grid_h <= 0
        or geometry.grid_w <= 0
        or geometry.video_start < 0
        or geometry.topk_r <= 0
    ):
        raise ValueError("invalid MiniMax-H3 RainFusion dispatch axes")

    video_rows = geometry.grid_t * geometry.grid_h * geometry.grid_w
    total_blocks = ceildiv(geometry.valid_rows, BLOCK)
    tail_rows = geometry.valid_rows - (total_blocks - 1) * BLOCK
    ff_rows = geometry.grid_h * geometry.grid_w
    ff_blocks = ceildiv(ff_rows, BLOCK)
    video_full_blocks = video_rows // BLOCK
    num_sq = video_full_blocks - ff_blocks
    video_end = geometry.video_start + video_rows

    if video_end > geometry.valid_rows or ff_blocks > video_full_blocks or num_sq < 1:
        raise ValueError("invalid MiniMax-H3 RainFusion sparse-video geometry")
    if geometry.topk_r > total_blocks:
        raise ValueError("invalid MiniMax-H3 RainFusion Top-R budget")

    return MinimaxH3RainfusionDerived(
        video_rows=video_rows,
        video_end=video_end,
        total_blocks=total_blocks,
        tail_rows=tail_rows,
        ff_rows=ff_rows,
        ff_blocks=ff_blocks,
        video_full_blocks=video_full_blocks,
        num_sq=num_sq,
        q_pool_pad=align_up(num_sq, M_TILE),
        k_pool_pad=align_up(total_blocks, POOL_ALIGN),
        plan_width=align_up(total_blocks, PLAN_ALIGN),
    )


def semantic_records(
    geometry: MinimaxH3RainfusionGeometry,
    derived: MinimaxH3RainfusionDerived | None = None,
) -> tuple[tuple[int, int, int, int], ...]:
    """Build the rf_v2 [src, dst, rows, flags] spans in semantic order.

    The first frame stays untouched when the latent grid has an H or W
    remainder. Every later frame is its tiled main region followed by that
    frame's H and W remainders; dense context rows are appended after the
    video span. Spans are split into 64-row records with a tail flag.
    """
    if derived is None:
        derived = derive(geometry)

    records: list[tuple[int, int, int, int]] = []

    def add_span(src: int, dst: int, rows: int, flags: int) -> None:
        if rows <= 0:
            return
        offset = 0
        while offset < rows:
            count = min(CHUNK_ROWS, rows - offset)
            record_flags = flags
            if offset + count == rows and rows % CHUNK_ROWS != 0:
                record_flags |= FLAG_TAIL
            records.append((src + offset, dst + offset, count, record_flags))
            offset += count

    remainder_h = geometry.grid_h % 8
    remainder_w = geometry.grid_w % 8
    h_main = geometry.grid_h - remainder_h
    w_main = geometry.grid_w - remainder_w
    dst = 0
    first_rearranged_frame = 0
    if remainder_h or remainder_w:
        add_span(geometry.video_start, dst, derived.ff_rows, FLAG_FIRST_FRAME)
        dst += derived.ff_rows
        first_rearranged_frame = 1

    for frame in range(first_rearranged_frame, geometry.grid_t):
        base = geometry.video_start + frame * derived.ff_rows
        for hn in range(geometry.grid_h // 8):
            for wn in range(w_main // 8):
                for hb in range(8):
                    src = base + (hn * 8 + hb) * geometry.grid_w + wn * 8
                    add_span(src, dst, 8, 0)
                    dst += 8
        if remainder_h:
            add_span(base + h_main * geometry.grid_w, dst, remainder_h * geometry.grid_w, 0)
            dst += remainder_h * geometry.grid_w
        if remainder_w:
            for h in range(h_main):
                add_span(base + h * geometry.grid_w + w_main, dst, remainder_w, 0)
                dst += remainder_w

    add_span(0, dst, geometry.video_start, FLAG_DENSE)
    dst += geometry.video_start
    if geometry.valid_rows > derived.video_end:
        add_span(derived.video_end, dst, geometry.valid_rows - derived.video_end, FLAG_DENSE)
        dst += geometry.valid_rows - derived.video_end

    if dst != geometry.valid_rows:
        raise ValueError(f"MiniMax-H3 RainFusion records cover {dst} rows, expected {geometry.valid_rows}")
    return tuple(records)


def record_count(geometry: MinimaxH3RainfusionGeometry) -> int:
    """Return the active record count that the inverse epilogue iterates."""
    return len(semantic_records(geometry))


def records_padded(geometry: MinimaxH3RainfusionGeometry) -> int:
    """Return the physical [N, 4] table extent, divisible by 48 workers."""
    return WORKERS * ceildiv(record_count(geometry), WORKERS)


def logical_fits_capacity(
    geometry: MinimaxH3RainfusionGeometry,
    derived: MinimaxH3RainfusionDerived | None = None,
    count: int | None = None,
) -> bool:
    """Return whether the logical geometry fits the qualified capacity pair."""
    if derived is None:
        derived = derive(geometry)
    if count is None:
        count = record_count(geometry)
    capacity = QUALIFIED_CAPACITY
    if geometry.total_rows > capacity.total_rows or geometry.valid_rows > geometry.total_rows:
        return False
    if derived.total_blocks > capacity.num_blocks or count > capacity.record_count:
        return False
    padded = WORKERS * ceildiv(count, WORKERS)
    if padded > capacity.records_padded:
        return False
    if geometry.topk_r > capacity.topk_r or geometry.topk_r > derived.total_blocks:
        return False
    if derived.num_sq > capacity.q_pool_pad or derived.total_blocks > capacity.k_pool_pad:
        return False
    if align_up(derived.total_blocks, PLAN_ALIGN) > capacity.plan_width:
        return False
    if ceildiv(derived.total_blocks, N_TILE) > ceildiv(capacity.k_pool_pad, N_TILE):
        return False
    if geometry.total_rows % KV_PACK_ROWS:
        return False
    if ceildiv(geometry.valid_rows, BLOCK) * BLOCK > geometry.total_rows:
        return False
    return True


def is_qualified_profile(geometry: MinimaxH3RainfusionGeometry) -> bool:
    """Return whether the geometry is legal for the rf_v2 production profile."""
    if geometry.total_rows <= 0 or geometry.valid_rows <= 0 or geometry.valid_rows > geometry.total_rows:
        return False
    if geometry.total_rows % BLOCK or geometry.total_rows % KV_PACK_ROWS:
        return False
    total_blocks = ceildiv(geometry.valid_rows, BLOCK)
    if total_blocks * BLOCK > geometry.total_rows:
        return False
    if geometry.grid_t <= 0 or geometry.grid_h <= 0 or geometry.grid_w <= 0:
        return False
    video_rows = geometry.grid_t * geometry.grid_h * geometry.grid_w
    if geometry.video_start < 0 or geometry.video_start + video_rows > geometry.valid_rows:
        return False
    if geometry.topk_r != derive_topk_r(total_blocks) or geometry.topk_r < 1 or geometry.topk_r > total_blocks:
        return False
    return True


def unavailability_reason(geometry: MinimaxH3RainfusionGeometry) -> str:
    """Classify why the capacity pair cannot serve this geometry.

    Returns an empty string when the request is servable. The Python JIT path
    compiles the capacity pair on demand, so a missing build artifact is not a
    separate reason here; a compile failure surfaces when preparation runs.
    """
    if not is_qualified_profile(geometry):
        return "unqualified_geometry"
    video_rows = geometry.grid_t * geometry.grid_h * geometry.grid_w
    ff_blocks = ceildiv(geometry.grid_h * geometry.grid_w, BLOCK)
    if video_rows // BLOCK - ff_blocks < 1:
        return "no_sparse_video_query_blocks"
    try:
        derived = derive(geometry)
        count = record_count(geometry)
    except ValueError:
        return "invalid_sparse_video_geometry"
    if not logical_fits_capacity(geometry, derived, count):
        return "logical_geometry_exceeds_capacity"
    return ""
