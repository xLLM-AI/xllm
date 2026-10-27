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


import triton
import triton.language as tl


@triton.jit
def tmo_dcp_prepare_indexer_merge_kernel(
    scores_ptr: tl.tensor,
    slots_ptr: tl.tensor,
    sortable_scores_ptr: tl.tensor,
    prepared_slots_ptr: tl.tensor,
    score_rank_stride: tl.tensor,
    score_query_stride: tl.tensor,
    score_col_stride: tl.tensor,
    slot_rank_stride: tl.tensor,
    slot_query_stride: tl.tensor,
    slot_col_stride: tl.tensor,
    output_query_stride: tl.tensor,
    CANDIDATES: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    rank = tl.program_id(0).to(tl.int64)
    query = tl.program_id(1).to(tl.int64)
    columns = tl.arange(0, BLOCK_N).to(tl.int64)
    mask = columns < CANDIDATES
    score = tl.load(
        scores_ptr + rank * score_rank_stride + query * score_query_stride + columns * score_col_stride,
        mask=mask,
        other=0,
    )
    slot = tl.load(
        slots_ptr + rank * slot_rank_stride + query * slot_query_stride + columns * slot_col_stride,
        mask=mask,
        other=-1,
    )
    bits = score.to(tl.int32, bitcast=True)
    valid = (slot >= 0) & ((bits & 0x7FFFFFFF) <= 0x7F800000)
    # Negate through the sign bit to preserve subnormals. Ascending TopK puts
    # canonical positive NaNs after every valid score, including negated -inf.
    # Canonicalization matters: MLU TopK orders negative NaNs before -inf.
    key_bits = tl.where(valid, bits ^ -2147483648, 0x7FC00000)
    key = key_bits.to(tl.float32, bitcast=True)
    output = query * output_query_stride + rank * CANDIDATES + columns
    tl.store(sortable_scores_ptr + output, key, mask)
    tl.store(prepared_slots_ptr + output, tl.where(valid, slot, -1), mask)


@triton.jit
def tmo_dcp_finalize_indexer_merge_kernel(
    prepared_slots_ptr: tl.tensor,
    topk_indices_ptr: tl.tensor,
    slot_mapping_ptr: tl.tensor,
    output_slots_ptr: tl.tensor,
    context_lens_ptr: tl.tensor,
    prepared_query_stride: tl.tensor,
    index_query_stride: tl.tensor,
    mapping_stride: tl.tensor,
    output_query_stride: tl.tensor,
    output_col_stride: tl.tensor,
    lengths_stride: tl.tensor,
    TOPK: tl.constexpr,
    BLOCK_K: tl.constexpr,
) -> None:
    query = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK_K).to(tl.int64)
    mask = columns < TOPK
    index = tl.load(
        topk_indices_ptr + query * index_query_stride + columns,
        mask=mask,
        other=0,
    )
    slot = tl.load(
        prepared_slots_ptr + query * prepared_query_stride + index,
        mask=mask,
        other=-1,
    )
    is_padding = tl.load(slot_mapping_ptr + query * mapping_stride) < 0
    # Sorted TopK keys place every valid slot before the invalid tail.
    valid = mask & ~is_padding & (slot >= 0)
    tl.store(
        output_slots_ptr + query * output_query_stride + columns * output_col_stride,
        tl.where(valid, slot, -1),
        mask=mask,
    )
    tl.store(context_lens_ptr + query * lengths_stride, tl.sum(valid.to(tl.int32)))
