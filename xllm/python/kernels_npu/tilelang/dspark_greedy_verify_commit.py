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

"""Commit ragged greedy verification and mask packed context KV in one launch."""

import tilelang.language as T
from tilelang import tvm

DSPARK_GREEDY_VERIFY_COMMIT_PASS_CONFIGS = {
    # Widths/offsets/owners are produced by the checked packing contract.
    # Every copy and loop is explicitly bounded. Predicated scalar loads in
    # LocalTensor::SetValue produce invalid nested C++ with this codegen.
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": True,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def build_dspark_greedy_verify_commit_kernel(max_batch_size: int, draft_width: int = 7) -> tvm.tir.PrimFunc:
    """Require B <= max_batch_size and G <= (draft_width + 1) * max_batch_size.

    All ranks must agree on proposals, packed target IDs and prefix widths
    before this kernel. Outputs borrow preallocated, non-overlapping storage.
    No output is read by another core; context writers recompute the tiny
    prefix decision from read-only inputs instead of adding a global barrier.
    """
    if max_batch_size not in (1, 2, 4, 8, 16, 32, 64, 128):
        raise ValueError("max_batch_size must be a supported power-of-two bucket")
    if draft_width not in (5, 7):
        raise ValueError("draft_width must be 5 or 7")
    result_width = draft_width + 1
    batch_tile = ((max_batch_size + 31) // 32) * 32
    offset_tile = ((max_batch_size + 32) // 32) * 32
    proposal_tile = ((max_batch_size * draft_width + 15) // 16) * 16
    target_tile = ((max_batch_size * result_width + 7) // 8) * 8
    row_tile = 32

    @T.prim_func
    def dspark_greedy_verify_commit(
        proposals_handle: T.handle,
        target_handle: T.handle,
        widths_handle: T.handle,
        offsets_handle: T.handle,
        owners_handle: T.handle,
        columns_handle: T.handle,
        source_slots_handle: T.handle,
        tokens_handle: T.handle,
        lengths_handle: T.handle,
        context_slots_handle: T.handle,
        batch_size: T.int32,
        capacity: T.int32,
        copy_cores: T.int32,
    ):
        # Select-wrapped dimensions retain the explicit INT32 scalar ABI;
        # direct symbolic dimensions otherwise infer duplicate INT64 args.
        proposals = T.match_buffer(proposals_handle, (T.Select(batch_size > 0, batch_size * draft_width, 0),), "int64")
        target = T.match_buffer(target_handle, (T.Select(capacity > 0, capacity, 0),), "int64")
        widths = T.match_buffer(widths_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        offsets = T.match_buffer(offsets_handle, (T.Select(batch_size >= 0, batch_size + 1, 1),), "int32")
        owners = T.match_buffer(owners_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        columns = T.match_buffer(columns_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        source_slots = T.match_buffer(source_slots_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        tokens = T.match_buffer(tokens_handle, (T.Select(batch_size > 0, batch_size * result_width, 0),), "int64")
        lengths = T.match_buffer(lengths_handle, (T.Select(batch_size > 0, batch_size, 0),), "int32")
        context_slots = T.match_buffer(context_slots_handle, (T.Select(capacity > 0, capacity, 0),), "int32")
        # The wrapper supplies ceil(G/64) in [1,16], and skips G=0. Keeping
        # launch dimensions explicit avoids Ascend's device-only max() in
        # the generated Host launcher. Each core covers two 32-row V tiles.
        with T.Kernel(copy_cores, is_npu=True) as (core, vid), T.Scope("V"):
            task = T.alloc_var("int32", init=0)
            task = T.Cast("int32", core * 2 + vid)
            proposals_ub = T.alloc_ub((proposal_tile,), "int64")
            target_ub = T.alloc_ub((target_tile,), "int64")
            widths_ub = T.alloc_ub((batch_tile,), "int32")
            offsets_ub = T.alloc_ub((offset_tile,), "int32")
            tokens_ub = T.alloc_ub((target_tile,), "int64")
            lengths_ub = T.alloc_ub((batch_tile,), "int32")
            owners_ub = T.alloc_ub((row_tile,), "int32")
            columns_ub = T.alloc_ub((row_tile,), "int32")
            source_slots_ub = T.alloc_ub((row_tile,), "int32")
            context_slots_ub = T.alloc_ub((row_tile,), "int32")
            accepted = T.alloc_var("int32", init=0)
            owner = T.alloc_var("int32", init=0)
            begin = T.alloc_var("int32", init=0)
            if batch_size > 0:
                T.copy(proposals[0 : batch_size * draft_width], proposals_ub)
                T.copy(widths[0:batch_size], widths_ub)
                T.copy(offsets[0 : batch_size + 1], offsets_ub)
            if capacity > 0:
                T.copy(target[0:capacity], target_ub)
            if task == 0 and batch_size > 0:
                # One writer publishes the small result/count arrays without
                # adjacent cores sharing a partial count cache line.
                for request in T.serial(batch_size):
                    begin = offsets_ub[request]
                    accepted = widths_ub[request]
                    for step in T.serial(draft_width):
                        if step < widths_ub[request] - 1:
                            if proposals_ub[request * draft_width + step] != target_ub[begin + step]:
                                accepted = T.min(accepted, step + 1)
                    lengths_ub[request] = accepted
                    for column in T.serial(result_width):
                        tokens_ub[request * result_width + column] = T.Cast("int64", -1)
                        if column < accepted:
                            tokens_ub[request * result_width + column] = target_ub[begin + column]
            start = task * row_tile
            row_count = T.min(T.int32(row_tile), capacity - start)
            if row_count > 0:
                T.copy(owners[start : start + row_count], owners_ub)
                T.copy(columns[start : start + row_count], columns_ub)
                T.copy(source_slots[start : start + row_count], source_slots_ub)
                for lane in T.serial(row_count):
                    context_slots_ub[lane] = -1
                    owner = owners_ub[lane]
                    if owner >= 0 and owner < batch_size:
                        begin = offsets_ub[owner]
                        accepted = widths_ub[owner]
                        for step in T.serial(draft_width):
                            if step < widths_ub[owner] - 1:
                                if proposals_ub[owner * draft_width + step] != target_ub[begin + step]:
                                    accepted = T.min(accepted, step + 1)
                        if columns_ub[lane] >= 0 and columns_ub[lane] < accepted:
                            context_slots_ub[lane] = source_slots_ub[lane]
                T.copy(context_slots_ub, context_slots[start : start + row_count])
            if task == 0 and batch_size > 0:
                # Keep int64 staging live while the Ascend UB planner allocates
                # int32 row tiles, preventing reuse with the wrong element type.
                T.copy(tokens_ub, tokens[0 : batch_size * result_width])
                T.copy(lengths_ub, lengths[0:batch_size])

    return dspark_greedy_verify_commit
