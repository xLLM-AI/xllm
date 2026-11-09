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

"""Resolve Full read tables and write slots in one graph-capturable launch."""

import tilelang.language as T
from tilelang import tvm

SHARED_FULL_INDICES_TILE = 128
SHARED_FULL_INDICES_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_shared_full_indices_kernel(tile_size: int) -> tvm.tir.PrimFunc:
    if tile_size != SHARED_FULL_INDICES_TILE:
        raise ValueError("shared Full mapping requires the supported output tile size")

    @T.prim_func
    def shared_full_indices(
        tables_handle: T.handle,
        slots_handle: T.handle,
        rows_handle: T.handle,
        out_tables_handle: T.handle,
        out_slots_handle: T.handle,
        table_elements: T.int32,
        slot_elements: T.int32,
        layers: T.int32,
        capacity: T.int32,
        row_stride: T.int32,
        page_size: T.int32,
        padding_row: T.int32,
        launch_cores: T.int32,
    ):
        tables = T.match_buffer(tables_handle, (T.Select(table_elements > 0, table_elements, 0),), "int32")
        slots = T.match_buffer(slots_handle, (T.Select(slot_elements > 0, slot_elements, 0),), "int32")
        rows = T.match_buffer(rows_handle, ((layers - 1) * row_stride + capacity,), "int32")
        out_tables = T.match_buffer(out_tables_handle, (layers * table_elements,), "int32")
        out_slots = T.match_buffer(out_slots_handle, (layers * slot_elements,), "int32")
        with T.Kernel(launch_cores, is_npu=True) as (core, vid), T.Scope("V"):
            task = T.alloc_var("int32", init=0)
            task = T.Cast("int32", core) * 2 + T.Cast("int32", vid)
            output_ub = T.alloc_ub((tile_size,), "int32")
            logical = T.alloc_var("int32", init=0)
            layer = T.alloc_var("int32", init=0)
            logical_page = T.alloc_var("int32", init=0)
            start = T.alloc_var("int32", init=0)
            row_offset = T.alloc_var("int32", init=0)
            output_offset = T.alloc_var("int32", init=0)
            table_tiles_per_layer = T.ceildiv(table_elements, tile_size)
            slot_tiles_per_layer = T.ceildiv(slot_elements, tile_size)
            table_tiles = layers * table_tiles_per_layer
            slot_tiles = layers * slot_tiles_per_layer
            for group in T.serial(T.ceildiv(table_tiles + slot_tiles, launch_cores * 2)):
                tile = group * launch_cores * 2 + task
                if tile < table_tiles:
                    layer = tile // table_tiles_per_layer
                    start = (tile - layer * table_tiles_per_layer) * tile_size
                    row_offset = layer * row_stride
                    output_offset = layer * table_elements + start
                    count = T.min(T.int32(tile_size), table_elements - start)
                    for lane in T.serial(count):
                        logical = tables[start + lane]
                        if logical > 0:
                            output_ub[lane] = rows[row_offset + logical]
                        elif logical == 0:
                            output_ub[lane] = padding_row
                        else:
                            output_ub[lane] = logical
                    # Scalar gathers populate UB directly. Synchronize only
                    # its handoff to DMA and completion before the next tile.
                    T.set_flag("S", "MTE3", 0)
                    T.wait_flag("S", "MTE3", 0)
                    T.copy(output_ub, out_tables[output_offset : output_offset + count])
                    T.set_flag("MTE3", "S", 0)
                    T.wait_flag("MTE3", "S", 0)
                elif tile < table_tiles + slot_tiles:
                    layer = (tile - table_tiles) // slot_tiles_per_layer
                    start = (tile - table_tiles - layer * slot_tiles_per_layer) * tile_size
                    row_offset = layer * row_stride
                    output_offset = layer * slot_elements + start
                    count = T.min(T.int32(tile_size), slot_elements - start)
                    for lane in T.serial(count):
                        logical = slots[start + lane]
                        if logical >= 0:
                            # Write slot zero addresses warmup scratch. Only
                            # read-table zero uses the immutable padding pair.
                            logical_page = logical // page_size
                            physical = rows[row_offset + logical_page]
                            output_ub[lane] = physical * page_size + logical % page_size
                        else:
                            output_ub[lane] = logical
                    T.set_flag("S", "MTE3", 0)
                    T.wait_flag("S", "MTE3", 0)
                    T.copy(output_ub, out_slots[output_offset : output_offset + count])
                    T.set_flag("MTE3", "S", 0)
                    T.wait_flag("MTE3", "S", 0)

    return shared_full_indices
