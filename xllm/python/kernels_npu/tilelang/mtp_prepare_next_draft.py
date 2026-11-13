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

"""Prepare repair/current MTP rows without an ACLNN workspace or tiling call."""

import tilelang.language as T
from tilelang import tvm

MTP_PREPARE_NEXT_DRAFT_PASS_CONFIGS = {
    "tl.disable_safe_memory_legalize": True,
    "tl.ascend_auto_sync": False,
    "tl.ascend_memory_planning": True,
    "tl.ascend_auto_cross_core_sync": False,
    "tl.ascend_auto_cv_combine": False,
}


def prepare_vector_only_source(source: str) -> str:
    # This TileLang Ascend version emits a mixed task even for developer-mode
    # vector kernels. No cube instructions or cross-core synchronization exist
    # here; a vector-only task avoids scheduling an idle cube for every call.
    return source.replace("KERNEL_TYPE_MIX_AIC_1_1", "KERNEL_TYPE_AIV_ONLY")


def build_mtp_prepare_next_draft_kernel(preparation_mode: int) -> tvm.tir.PrimFunc:
    """Prepare full hidden (0), compact hidden (1), or a metadata-only prefix (2)."""
    if preparation_mode not in (0, 1, 2):
        raise ValueError("preparation_mode must be full (0), compact (1), or prefix (2)")

    @T.prim_func
    def mtp_prepare_next_draft(
        tokens_handle: T.handle,
        hidden_handle: T.handle,
        placeholder_handle: T.handle,
        base_positions_handle: T.handle,
        base_lengths_handle: T.handle,
        table_handle: T.handle,
        shared_table_handle: T.handle,
        output_tokens_handle: T.handle,
        output_hidden_handle: T.handle,
        output_positions_handle: T.handle,
        output_lengths_handle: T.handle,
        output_slots_handle: T.handle,
        output_shared_slots_handle: T.handle,
        output_logits_indices_handle: T.handle,
        batch_size: T.int32,
        speculative_width: T.int32,
        hidden_size: T.int32,
        table_columns: T.int32,
        block_size: T.int32,
        copy_cores: T.int32,
        use_shared_table: T.int32,
    ):
        tokens = T.match_buffer(tokens_handle, (batch_size * speculative_width,), "int64")
        hidden = T.match_buffer(
            hidden_handle, (batch_size * (2 if preparation_mode == 1 else speculative_width) * hidden_size,), "uint16"
        )
        placeholder = T.match_buffer(placeholder_handle, (T.max(hidden_size, 1),), "uint16")
        base_positions = T.match_buffer(base_positions_handle, (T.max(batch_size, 1),), "int32")
        base_lengths = T.match_buffer(base_lengths_handle, (T.max(batch_size, 1),), "int32")
        table = T.match_buffer(
            table_handle,
            (batch_size * (speculative_width if preparation_mode == 2 else 1) * table_columns,),
            "int32",
        )
        shared_table = T.match_buffer(
            shared_table_handle,
            (batch_size * (speculative_width if preparation_mode == 2 else 1) * table_columns,),
            "int32",
        )
        output_tokens = T.match_buffer(
            output_tokens_handle, (batch_size * (speculative_width if preparation_mode == 2 else 2),), "int32"
        )
        output_hidden = T.match_buffer(output_hidden_handle, (batch_size * 2 * hidden_size,), "uint16")
        output_positions = T.match_buffer(
            output_positions_handle, (batch_size * (speculative_width if preparation_mode == 2 else 2),), "int32"
        )
        output_lengths = T.match_buffer(
            output_lengths_handle,
            (
                T.max(
                    batch_size * (speculative_width if preparation_mode == 2 else (2 if preparation_mode == 1 else 1)),
                    1,
                ),
            ),
            "int32",
        )
        output_slots = T.match_buffer(
            output_slots_handle, (batch_size * (speculative_width if preparation_mode == 2 else 2),), "int32"
        )
        output_shared_slots = T.match_buffer(
            output_shared_slots_handle, (batch_size * (speculative_width if preparation_mode == 2 else 2),), "int32"
        )
        output_logits_indices = T.match_buffer(output_logits_indices_handle, (batch_size,), "int32")
        with T.Kernel(copy_cores, threads=1, is_npu=True) as core:
            accepted = T.alloc_var("int32")
            cache_position = T.alloc_var("int32")
            if preparation_mode == 2:
                # Prefix rows retain the target hidden tensor as a view. Only
                # metadata is written, including both logical and final slots.
                if core == 0:
                    for row in T.serial(batch_size):
                        accepted = 0
                        for step in T.serial(speculative_width):
                            token = tokens[row * speculative_width + step]
                            if token >= 0:
                                accepted += 1
                            output_row = row * speculative_width + step
                            output_tokens[output_row] = T.Cast("int32", T.max(token, 0))
                            output_positions[output_row] = base_positions[row] + step
                            output_lengths[output_row] = base_lengths[row] + step
                            cache_position = base_lengths[row] + step - 1
                            column = T.max(cache_position, 0) // block_size
                            output_slots[output_row] = 0
                            if use_shared_table != 0:
                                output_shared_slots[output_row] = -1
                            if cache_position >= 0 and column < table_columns:
                                table_index = row * speculative_width * table_columns + column
                                block_id = table[table_index]
                                if block_id >= 0:
                                    output_slots[output_row] = block_id * block_size + cache_position % block_size
                                if use_shared_table != 0 and block_id > 0:
                                    page = shared_table[table_index]
                                    if page >= 0:
                                        output_shared_slots[output_row] = (
                                            page * block_size + cache_position % block_size
                                        )
                        output_logits_indices[row] = row * speculative_width + accepted - 1
            else:
                hidden_tile = T.alloc_ub((32768,), "uint16")
                source_row = T.alloc_var("int32")
                selected_index = T.alloc_var("int32")
                if core == 0:
                    for row in T.serial(batch_size):
                        accepted = 0
                        for step in T.serial(speculative_width):
                            if tokens[row * speculative_width + step] >= 0:
                                accepted += 1
                        last_index = T.max(accepted - 1, 0)
                        previous_index = T.max(accepted - 2, 0)
                        output_tokens[row * 2] = T.Cast("int32", tokens[row * speculative_width + previous_index])
                        output_tokens[row * 2 + 1] = T.Cast("int32", tokens[row * speculative_width + last_index])
                        current_position = base_positions[row] + accepted
                        current_length = base_lengths[row] + accepted
                        output_positions[row * 2] = current_position - 1
                        output_positions[row * 2 + 1] = current_position
                        if preparation_mode == 1:
                            output_lengths[row * 2] = current_length - 1
                            output_lengths[row * 2 + 1] = current_length
                        else:
                            output_lengths[row] = current_length
                        for pair in T.serial(2):
                            cache_position = current_position
                            if pair == 0:
                                cache_position = current_position + 1
                                if accepted == speculative_width:
                                    cache_position = current_position - 1
                            column = T.max(cache_position, 0) // block_size
                            output_slots[row * 2 + pair] = 0
                            if use_shared_table != 0:
                                output_shared_slots[row * 2 + pair] = -1
                            if cache_position >= 0 and column < table_columns:
                                block_id = table[row * table_columns + column]
                                if block_id >= 0:
                                    output_slots[row * 2 + pair] = block_id * block_size + cache_position % block_size
                                if use_shared_table != 0 and block_id > 0:
                                    # Read-padding pages must never receive writes.
                                    # Native tables already contain the final FIA page.
                                    page = shared_table[row * table_columns + column]
                                    if page >= 0:
                                        output_shared_slots[row * 2 + pair] = (
                                            page * block_size + cache_position % block_size
                                        )
                # Metadata has one writer; hidden rows are disjoint, 32-byte
                # aligned DMA ranges and can be copied on independent vector cores.
                for group in T.serial(T.ceildiv(batch_size * 2, 32)):
                    output_row = group * 32 + core
                    if output_row < batch_size * 2:
                        row = T.Cast("int32", output_row // 2)
                        pair = T.Cast("int32", output_row % 2)
                        accepted = 0
                        for step in T.serial(speculative_width):
                            if tokens[row * speculative_width + step] >= 0:
                                accepted += 1
                        selected_index = accepted - 2
                        if pair == 1:
                            selected_index += 1
                        selected_index = T.max(selected_index, 0)
                        source_row = row * speculative_width + selected_index
                        if preparation_mode == 1:
                            source_row = output_row
                        for chunk in T.serial(T.ceildiv(hidden_size, 32768)):
                            width = T.min(32768, hidden_size - chunk * 32768)
                            if pair == 0 and accepted <= 1:
                                T.copy(placeholder[chunk * 32768 : chunk * 32768 + width], hidden_tile)
                            else:
                                source = source_row * hidden_size + chunk * 32768
                                T.copy(hidden[source : source + width], hidden_tile)
                            T.set_flag("mte2", "mte3", 0)
                            T.wait_flag("mte2", "mte3", 0)
                            destination = output_row * hidden_size + chunk * 32768
                            T.copy(hidden_tile, output_hidden[destination : destination + width])
                            T.set_flag("mte3", "mte2", 0)
                            T.wait_flag("mte3", "mte2", 0)

    return mtp_prepare_next_draft
