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

"""Pack authoritative DSpark prefixes into stable fixed-capacity graph inputs."""

import tilelang.language as T
from tilelang import tvm

DSPARK_PACK_VERIFY_INPUTS_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}

DSPARK_BLOCK_TABLE_TILE = 4096


def build_dspark_pack_verify_inputs_kernel(max_batch_size: int, draft_width: int = 7) -> tvm.tir.PrimFunc:
    """Read widths after TP broadcast; fuse mapping, gathers, metadata and tables."""
    if max_batch_size not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("max_batch_size must be a supported power-of-two bucket")
    if draft_width not in (5, 7):
        raise ValueError("draft_width must be 5 or 7")
    batch_tile = ((max_batch_size + 31) // 32) * 32
    offset_tile = ((max_batch_size + 32) // 32) * 32
    proposal_tile = ((max_batch_size * draft_width + 15) // 16) * 16
    row_tile = 32

    @T.prim_func
    def dspark_pack_verify_inputs(
        anchors_handle: T.handle,
        proposals_handle: T.handle,
        base_positions_handle: T.handle,
        base_lengths_handle: T.handle,
        base_tables_handle: T.handle,
        widths_handle: T.handle,
        offsets_handle: T.handle,
        tokens_handle: T.handle,
        positions_handle: T.handle,
        lengths_handle: T.handle,
        slots_handle: T.handle,
        tables_handle: T.handle,
        query_lengths_handle: T.handle,
        owners_handle: T.handle,
        columns_handle: T.handle,
        batch_size: T.int32,
        capacity: T.int32,
        table_columns: T.int32,
        block_size: T.int32,
        copy_cores: T.int32,
    ):
        # Guarded shape expressions avoid duplicate inferred shape parameters;
        # exact dynamic spans still equal the caller-validated tensor storage.
        anchors = T.match_buffer(anchors_handle, (T.Select(batch_size > 0, batch_size, 0),), "int64")
        proposals = T.match_buffer(proposals_handle, (batch_size * draft_width,), "int64")
        base_positions = T.match_buffer(base_positions_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        base_lengths = T.match_buffer(base_lengths_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        base_tables = T.match_buffer(base_tables_handle, (batch_size * table_columns,), "int32")
        widths = T.match_buffer(widths_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        offsets = T.match_buffer(offsets_handle, (batch_size + 1,), "int32")
        tokens = T.match_buffer(tokens_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        positions = T.match_buffer(positions_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        lengths = T.match_buffer(lengths_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        slots = T.match_buffer(slots_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        tables = T.match_buffer(tables_handle, (capacity * table_columns,), "int32")
        query_lengths = T.match_buffer(query_lengths_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        owners = T.match_buffer(owners_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        columns = T.match_buffer(columns_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        with T.Kernel(copy_cores, is_npu=True) as (core, vid), T.Scope("V"):
            if vid == 0:
                # Kernel IDs are INT64 in this backend. Keep buffer/index
                # arithmetic on the explicit INT32 runtime ABI throughout.
                task = T.alloc_var("int32", init=0)
                task = T.Cast("int32", core)
                start = T.alloc_var("int32", init=0)
                row_count = T.alloc_var("int32", init=0)
                width_ub = T.alloc_ub((batch_tile,), "int32")
                offsets_ub = T.alloc_ub((offset_tile,), "int32")
                anchors_ub = T.alloc_ub((batch_tile,), "int64")
                proposals_ub = T.alloc_ub((proposal_tile,), "int64")
                base_positions_ub = T.alloc_ub((batch_tile,), "int32")
                base_lengths_ub = T.alloc_ub((batch_tile,), "int32")
                tokens_ub = T.alloc_ub((row_tile,), "int32")
                positions_ub = T.alloc_ub((row_tile,), "int32")
                lengths_ub = T.alloc_ub((row_tile,), "int32")
                slots_ub = T.alloc_ub((row_tile,), "int32")
                query_lengths_ub = T.alloc_ub((row_tile,), "int32")
                owners_ub = T.alloc_ub((row_tile,), "int32")
                columns_ub = T.alloc_ub((row_tile,), "int32")
                table_ub = T.alloc_ub((DSPARK_BLOCK_TABLE_TILE,), "int32")
                owner = T.alloc_var("int32", init=0)
                column = T.alloc_var("int32", init=0)
                position = T.alloc_var("int32", init=0)
                total = T.alloc_var("int32", init=0)
                active_rows = T.alloc_var("int32", init=batch_size)
                # A sentinel makes owner+1 safe even when the compound while
                # condition evaluates both operands after the final request.
                T.tile.fill(offsets_ub, 2147483647)
                offsets_ub[0] = 0
                if active_rows > 0:
                    T.copy(widths[0:active_rows], width_ub)
                    T.copy(anchors[0:active_rows], anchors_ub)
                    T.copy(proposals[0 : active_rows * draft_width], proposals_ub)
                    T.copy(base_positions[0:active_rows], base_positions_ub)
                    T.copy(base_lengths[0:active_rows], base_lengths_ub)
                    for row in T.serial(active_rows):
                        total += width_ub[row]
                        offsets_ub[row + 1] = total
                if task == 0:
                    T.copy(offsets_ub, offsets[0 : active_rows + 1])
                # The wrapper caps the mixed-grid count to the device's actual
                # AI core count. Persistent groups cover G without oversubscribing
                # the launch grid or recopying immutable base metadata per group.
                for group in T.serial(T.ceildiv(capacity, copy_cores * T.int32(row_tile))):
                    start = (group * copy_cores + task) * T.int32(row_tile)
                    if start < capacity:
                        row_count = T.min(T.int32(row_tile), capacity - start)
                        T.tile.fill(tokens_ub, 1)
                        T.tile.fill(positions_ub, 0)
                        T.tile.fill(lengths_ub, 1)
                        T.tile.fill(slots_ub, -1)
                        T.tile.fill(query_lengths_ub, 1)
                        T.tile.fill(owners_ub, -1)
                        T.tile.fill(columns_ub, -1)
                        # Assigned groups are monotonic, so seeking advances
                        # through at most B requests across the complete call.
                        while (owner < active_rows) and (start >= offsets_ub[owner + 1]):
                            owner += 1
                        for lane in T.serial(row_count):
                            row = start + lane
                            if row < total:
                                while (owner < active_rows) and (row >= offsets_ub[owner + 1]):
                                    owner += 1
                                column = row - offsets_ub[owner]
                                position = base_positions_ub[owner] + column
                                owners_ub[lane] = owner
                                columns_ub[lane] = column
                                positions_ub[lane] = position
                                lengths_ub[lane] = base_lengths_ub[owner] + column
                                if column == 0:
                                    tokens_ub[lane] = T.Cast("int32", anchors_ub[owner])
                                else:
                                    tokens_ub[lane] = T.Cast("int32", proposals_ub[owner * draft_width + column - 1])
                            else:
                                T.tile.fill(table_ub, 0)
                            # Keep the complete caller-owned row stride without
                            # sizing UB to the model's maximum context length.
                            for table_tile in T.serial((table_columns - 1) // DSPARK_BLOCK_TABLE_TILE + 1):
                                table_start = table_tile * T.int32(DSPARK_BLOCK_TABLE_TILE)
                                table_count = T.min(T.int32(DSPARK_BLOCK_TABLE_TILE), table_columns - table_start)
                                if row < total:
                                    source = owner * table_columns + table_start
                                    T.copy(base_tables[source : source + table_count], table_ub)
                                    # A scalar page read must wait for MTE2. Only
                                    # the tile containing this row's page owns it.
                                    T.barrier_all()
                                    page = position // block_size
                                    if (page >= table_start) and (page < table_start + table_count):
                                        slots_ub[lane] = (
                                            table_ub[page - table_start] * block_size + position % block_size
                                        )
                                destination = row * table_columns + table_start
                                T.copy(table_ub, tables[destination : destination + table_count])
                                # MTE3 must finish before the next tile or row
                                # overwrites this shared scratch buffer.
                                T.barrier_all()
                        T.copy(tokens_ub, tokens[start : start + row_count])
                        T.copy(positions_ub, positions[start : start + row_count])
                        T.copy(lengths_ub, lengths[start : start + row_count])
                        T.copy(slots_ub, slots[start : start + row_count])
                        T.copy(query_lengths_ub, query_lengths[start : start + row_count])
                        T.copy(owners_ub, owners[start : start + row_count])
                        T.copy(columns_ub, columns[start : start + row_count])

    return dspark_pack_verify_inputs
