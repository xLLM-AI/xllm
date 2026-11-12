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

"""Resolve independent slab Conv/SSM rows into persistent GDN metadata."""

import tilelang.language as T
from tilelang import tvm

SHARED_LINEAR_STATE_INDICES_TILE = 128
SHARED_LINEAR_STATE_INDICES_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_shared_linear_state_indices_kernel(tile_size: int) -> tvm.tir.PrimFunc:
    if tile_size != SHARED_LINEAR_STATE_INDICES_TILE:
        raise ValueError("shared Linear mapping requires the supported output tile size")

    @T.prim_func
    def shared_linear_state_indices(
        ids_handle: T.handle,
        rows_handle: T.handle,
        out_handle: T.handle,
        batch_size: T.int32,
        groups: T.int32,
        capacity: T.int32,
        row_stride: T.int32,
        launch_cores: T.int32,
    ):
        ids = T.match_buffer(ids_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        rows = T.match_buffer(rows_handle, ((2 * groups - 1) * row_stride + capacity,), "int32")
        out = T.match_buffer(out_handle, (2 * groups * batch_size,), "int32")
        with T.Kernel(launch_cores, is_npu=True) as (core, vid), T.Scope("V"):
            task = T.alloc_var("int32", init=0)
            task = T.Cast("int32", core) * 2 + T.Cast("int32", vid)
            output_ub = T.alloc_ub((tile_size,), "int32")
            logical = T.alloc_var("int32", init=0)
            layer_group = T.alloc_var("int32", init=0)
            total = 2 * groups * batch_size
            for tile in T.serial(T.ceildiv(total, launch_cores * 2 * tile_size)):
                start = (tile * launch_cores * 2 + task) * tile_size
                if start < total:
                    count = T.min(T.int32(tile_size), total - start)
                    for lane in T.serial(count):
                        index = start + lane
                        layer_group = index // (2 * batch_size)
                        batch = index // 2 % batch_size
                        role = index % 2
                        logical = ids[batch]
                        if logical > 0:
                            output_ub[lane] = rows[(2 * layer_group + role) * row_stride + logical]
                        else:
                            output_ub[lane] = -1
                    # A C1 group holds only two int32 values. Tile across
                    # groups and copy the exact tail rather than storing an
                    # aligned vector beyond the caller-owned metadata.
                    # GM GetValue and UB SetValue use the scalar pipeline;
                    # only the output DMA crosses a pipeline boundary.
                    T.set_flag("S", "MTE3", 0)
                    T.wait_flag("S", "MTE3", 0)
                    T.copy(output_ub, out[start : start + count])
                    T.set_flag("MTE3", "S", 0)
                    T.wait_flag("MTE3", "S", 0)

    return shared_linear_state_indices
