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
"""NPU Triton kernel for GLM-Next compact-tail KPool compression.

The kernel consumes projected K/gate rows, completes every pool covered by
each request, then updates its owned tail. Pool reads finish before tail writes
so speculative spans may cross multiple compression boundaries safely.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _update_compact_kpool_kernel(
    raw_k_ptr,
    gate_ptr,
    valid_ptr,
    positions_ptr,
    pool_cache_ptr,
    tail_cache_ptr,
    tail_ids_ptr,
    block_table_ptr,
    ape_ptr,
    num_pool_blocks,
    num_tail_slots,
    tail_capacity,
    pool_block_size,
    block_table_width,
    raw_k_stride_row,
    raw_k_stride_dim,
    gate_stride_row,
    gate_stride_dim,
    valid_stride,
    positions_stride,
    pool_stride_block,
    pool_stride_slot,
    pool_stride_dim,
    tail_stride_slot,
    tail_stride_kind,
    tail_stride_row,
    tail_stride_dim,
    tail_ids_stride,
    block_table_stride_row,
    block_table_stride_col,
    ape_stride_member,
    ape_stride_dim,
    RATE: tl.constexpr,
    QUERY_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Update one request's compact tail and all completed pools."""
    request_idx = tl.program_id(0)
    dim_offsets = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    query_start = request_idx * QUERY_LEN

    tail_id = tl.load(tail_ids_ptr + request_idx * tail_ids_stride).to(tl.int64)
    request_active = (tail_id > 0) & (tail_id < num_tail_slots)
    safe_tail_id = tl.minimum(tl.maximum(tail_id, 0), num_tail_slots - 1)
    first_position = tl.load(
        positions_ptr + query_start * positions_stride,
        mask=request_active,
        other=-1,
    ).to(tl.int64)
    first_completion_offset = (RATE - 1 - first_position % RATE) % RATE
    # One request may complete several pools (e.g. an 8-row DFlash2 verify
    # with rate 4). Read every pool from the original tail before updating it.
    for pool_idx in tl.static_range((QUERY_LEN + RATE - 1) // RATE):
        completion_offset = first_completion_offset + pool_idx * RATE
        expected_completion_position = first_position + completion_offset
        completion_row = query_start + completion_offset
        has_completion = request_active & (completion_offset < QUERY_LEN)
        completion_position = tl.load(
            positions_ptr + completion_row * positions_stride,
            mask=has_completion,
            other=-1,
        ).to(tl.int64)
        pool_complete = (
            has_completion
            & (completion_position == expected_completion_position)
            & (completion_position >= RATE - 1)
            & ((completion_position + 1) % RATE == 0)
        )

        logit_max = tl.full((BLOCK_D,), float("-inf"), dtype=tl.float32)
        denominator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for member_idx in tl.static_range(RATE):
            member_position = completion_position - (RATE - 1 - member_idx)
            current_offset = member_position - first_position
            safe_current_offset = tl.minimum(tl.maximum(current_offset, 0), QUERY_LEN - 1)
            current_row = query_start + safe_current_offset
            current_position = tl.load(
                positions_ptr + current_row * positions_stride,
                mask=pool_complete,
                other=-1,
            ).to(tl.int64)
            from_current = (
                pool_complete
                & (current_offset >= 0)
                & (current_offset < QUERY_LEN)
                & (current_position == member_position)
            )
            current_valid = tl.load(
                valid_ptr + current_row * valid_stride,
                mask=from_current,
                other=0,
            ).to(tl.int1)
            member_valid = pool_complete & (member_position >= 0) & ((~from_current) | current_valid)
            tail_row = tl.maximum(member_position, 0) % tail_capacity
            tail_gate_address = (
                safe_tail_id * tail_stride_slot
                + tail_stride_kind
                + tail_row * tail_stride_row
                + dim_offsets * tail_stride_dim
            )
            old_gate = tl.load(
                tail_cache_ptr + tail_gate_address,
                mask=member_valid & (~from_current) & dim_mask,
                other=float("-inf"),
            ).to(tl.float32)
            old_marker = tl.load(
                tail_cache_ptr + safe_tail_id * tail_stride_slot + tail_stride_kind + tail_row * tail_stride_row,
                mask=pool_complete & (~from_current),
                other=float("-inf"),
            ).to(tl.float32)
            current_gate = tl.load(
                gate_ptr + current_row * gate_stride_row + dim_offsets * gate_stride_dim,
                mask=member_valid & from_current & dim_mask,
                other=float("-inf"),
            ).to(tl.float32)
            member_valid = member_valid & (from_current | (old_marker != float("-inf")))
            member_gate = tl.where(from_current, current_gate, old_gate)
            pool_complete = pool_complete & member_valid
            ape = tl.load(
                ape_ptr + member_idx * ape_stride_member + dim_offsets * ape_stride_dim,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            logit = member_gate + ape
            next_max = tl.maximum(logit_max, logit)
            rescale = tl.where(logit_max == float("-inf"), 0.0, tl.exp(logit_max - next_max))
            contribution = tl.where(logit == float("-inf"), 0.0, tl.exp(logit - next_max))
            denominator = denominator * rescale + contribution
            logit_max = next_max

        numerator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        for member_idx in tl.static_range(RATE):
            member_position = completion_position - (RATE - 1 - member_idx)
            current_offset = member_position - first_position
            safe_current_offset = tl.minimum(tl.maximum(current_offset, 0), QUERY_LEN - 1)
            current_row = query_start + safe_current_offset
            current_position = tl.load(
                positions_ptr + current_row * positions_stride,
                mask=pool_complete,
                other=-1,
            ).to(tl.int64)
            from_current = (
                pool_complete
                & (current_offset >= 0)
                & (current_offset < QUERY_LEN)
                & (current_position == member_position)
            )
            tail_row = tl.maximum(member_position, 0) % tail_capacity
            tail_key_address = (
                safe_tail_id * tail_stride_slot + tail_row * tail_stride_row + dim_offsets * tail_stride_dim
            )
            tail_gate_address = tail_key_address + tail_stride_kind
            old_key = tl.load(
                tail_cache_ptr + tail_key_address,
                mask=pool_complete & (~from_current) & dim_mask,
                other=0.0,
            ).to(tl.float32)
            old_gate = tl.load(
                tail_cache_ptr + tail_gate_address,
                mask=pool_complete & (~from_current) & dim_mask,
                other=float("-inf"),
            ).to(tl.float32)
            current_key = tl.load(
                raw_k_ptr + current_row * raw_k_stride_row + dim_offsets * raw_k_stride_dim,
                mask=pool_complete & from_current & dim_mask,
                other=0.0,
            ).to(tl.float32)
            current_gate = tl.load(
                gate_ptr + current_row * gate_stride_row + dim_offsets * gate_stride_dim,
                mask=pool_complete & from_current & dim_mask,
                other=float("-inf"),
            ).to(tl.float32)
            member_key = tl.where(from_current, current_key, old_key)
            member_gate = tl.where(from_current, current_gate, old_gate)
            ape = tl.load(
                ape_ptr + member_idx * ape_stride_member + dim_offsets * ape_stride_dim,
                mask=dim_mask,
                other=0.0,
            ).to(tl.float32)
            safe_denominator = tl.where(denominator > 0.0, denominator, 1.0)
            normalized_weight = tl.where(
                pool_complete,
                tl.exp(member_gate + ape - logit_max) / safe_denominator,
                0.0,
            )
            rounded_weight = normalized_weight.to(tl.bfloat16).to(tl.float32)
            rounded_product = (rounded_weight * member_key).to(tl.bfloat16).to(tl.float32)
            numerator += rounded_product

        pool_id = tl.maximum(completion_position, 0) // RATE
        logical_block = pool_id // pool_block_size
        table_in_range = logical_block < block_table_width
        safe_logical_block = tl.minimum(tl.maximum(logical_block, 0), block_table_width - 1)
        physical_block = tl.load(
            block_table_ptr + request_idx * block_table_stride_row + safe_logical_block * block_table_stride_col,
            mask=pool_complete & table_in_range,
            other=-1,
        ).to(tl.int64)
        physical_block_valid = (physical_block >= 0) & (physical_block < num_pool_blocks)
        safe_physical_block = tl.minimum(tl.maximum(physical_block, 0), num_pool_blocks - 1)
        pool_offset = pool_id % pool_block_size
        pool_address = (
            safe_physical_block * pool_stride_block + pool_offset * pool_stride_slot + dim_offsets * pool_stride_dim
        )
        tl.store(
            pool_cache_ptr + pool_address,
            numerator.to(tl.bfloat16),
            mask=pool_complete & table_in_range & physical_block_valid & dim_mask,
        )

    # Compression reads are complete before the circular tail is updated.
    for query_offset in tl.static_range(QUERY_LEN):
        current_row = query_start + query_offset
        current_position = tl.load(
            positions_ptr + current_row * positions_stride,
            mask=request_active,
            other=-1,
        ).to(tl.int64)
        current_valid = tl.load(
            valid_ptr + current_row * valid_stride,
            mask=request_active,
            other=0,
        ).to(tl.int1)
        write_tail = request_active & (current_position >= 0)
        tail_row = tl.maximum(current_position, 0) % tail_capacity
        tail_key_address = safe_tail_id * tail_stride_slot + tail_row * tail_stride_row + dim_offsets * tail_stride_dim
        current_key = tl.load(
            raw_k_ptr + current_row * raw_k_stride_row + dim_offsets * raw_k_stride_dim,
            mask=write_tail & current_valid & dim_mask,
            other=0.0,
        )
        current_gate = tl.load(
            gate_ptr + current_row * gate_stride_row + dim_offsets * gate_stride_dim,
            mask=write_tail & current_valid & dim_mask,
            other=float("-inf"),
        )
        tl.store(
            tail_cache_ptr + tail_key_address,
            current_key,
            mask=write_tail & dim_mask,
        )
        tl.store(
            tail_cache_ptr + tail_key_address + tail_stride_kind,
            current_gate,
            mask=write_tail & dim_mask,
        )


def update_compact_kpool(
    raw_k: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_rows: torch.Tensor,
    positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_ids: torch.Tensor,
    block_table: torch.Tensor,
    query_len: int,
    ape: torch.Tensor,
    rate: int,
) -> None:
    """Update uniform decode/MTP compact KPool state in one Triton launch.

    The current rows are sequence-major with one equal-width span per request.
    Read and write tail ids must already resolve to the same request-owned
    state; prefix restore with distinct ids remains on the torch path. The
    tail capacity bounds the query span, which may complete multiple pools.
    """
    if rate <= 0:
        raise ValueError("compact KPool compression rate must be positive")
    if raw_k.ndim == 0 or gate_scores.ndim == 0:
        raise ValueError("compact KPool raw K and gate tensors must have a feature dimension")
    tensors = (
        raw_k,
        gate_scores,
        valid_rows,
        positions,
        compressed_cache,
        tail_cache,
        tail_ids,
        block_table,
        ape,
    )
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("compact KPool Triton inputs must be contiguous")

    head_dim = raw_k.shape[-1]
    flat_k = raw_k.view(-1, head_dim)
    flat_gate = gate_scores.view(-1, gate_scores.shape[-1])
    flat_valid = valid_rows.view(-1)
    flat_positions = positions.view(-1)
    flat_tail_ids = tail_ids.view(-1)
    num_tokens = flat_k.shape[0]

    if head_dim <= 0:
        raise ValueError("compact KPool head dimension must be positive")
    if num_tokens == 0:
        return
    if query_len <= 0 or num_tokens % query_len != 0:
        raise ValueError("compact KPool Triton update requires uniform request spans")
    num_requests = num_tokens // query_len
    if flat_gate.shape != flat_k.shape:
        raise ValueError("raw K and gate rows must have identical shapes")
    if flat_valid.numel() != num_tokens or flat_positions.numel() != num_tokens:
        raise ValueError("compact KPool update requires one validity and position per token")
    if compressed_cache.ndim != 4 or compressed_cache.shape[2:] != (1, head_dim):
        raise ValueError("compressed KPool cache must have shape [blocks, pools_per_block, 1, dim]")
    if compressed_cache.shape[0] == 0 or compressed_cache.shape[1] == 0:
        raise ValueError("compressed KPool cache must be non-empty")
    if tail_cache.ndim != 4 or tail_cache.shape[1] != 2 or tail_cache.shape[3] != head_dim:
        raise ValueError("compact KPool tail must have shape [slots, 2, capacity, dim]")
    if tail_cache.shape[0] <= 1:
        raise ValueError("compact KPool tail must contain a padding slot and at least one request slot")
    if tail_cache.shape[2] < rate + query_len - 1:
        raise ValueError("compact KPool tail capacity must cover the pool and current query window")
    if block_table.ndim != 2 or block_table.shape[1] == 0:
        raise ValueError("compact KPool update requires a non-empty rank-2 block table")
    if flat_tail_ids.numel() < num_requests or block_table.shape[0] != num_requests:
        raise ValueError("compact KPool metadata must contain one row per request")
    if ape.shape != (rate, head_dim):
        raise ValueError("KPool APE must have shape [rate, dim]")
    if flat_k.dtype != torch.bfloat16 or flat_gate.dtype != torch.bfloat16:
        raise TypeError("compact KPool Triton update requires BF16 raw K and gate rows")
    if compressed_cache.dtype != torch.bfloat16 or tail_cache.dtype != torch.bfloat16:
        raise TypeError("compact KPool Triton update requires BF16 cache tensors")
    if ape.dtype != torch.bfloat16:
        raise TypeError("compact KPool Triton update requires BF16 APE")
    if flat_valid.dtype != torch.bool:
        raise TypeError("compact KPool Triton update validity rows must use bool")
    if flat_positions.dtype not in (torch.int32, torch.int64):
        raise TypeError("compact KPool Triton update positions must use int32 or int64")
    if flat_tail_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("compact KPool Triton update tail ids must use int32 or int64")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("compact KPool Triton update block table must use int32 or int64")
    colocated_tensors = (
        flat_gate,
        flat_valid,
        flat_positions,
        compressed_cache,
        tail_cache,
        flat_tail_ids,
        block_table,
        ape,
    )
    if flat_k.device.type not in ("npu", "privateuseone"):
        raise ValueError("compact KPool Triton update requires NPU tensors")
    if any(tensor.device != flat_k.device for tensor in colocated_tensors):
        raise ValueError("compact KPool Triton inputs must be colocated")

    _launch_compact_kpool(
        raw_k,
        gate_scores,
        valid_rows,
        positions,
        compressed_cache,
        tail_cache,
        tail_ids,
        block_table,
        query_len,
        ape,
        rate,
    )


def _launch_compact_kpool(
    raw_k: torch.Tensor,
    gate_scores: torch.Tensor,
    valid_rows: torch.Tensor,
    positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    tail_cache: torch.Tensor,
    tail_ids: torch.Tensor,
    block_table: torch.Tensor,
    query_len: int,
    ape: torch.Tensor,
    rate: int,
) -> None:
    """Launch after the model/backend have validated invariant inputs."""
    head_dim = raw_k.shape[-1]
    flat_k = raw_k.view(-1, head_dim)
    flat_gate = gate_scores.view(-1, head_dim)
    flat_valid = valid_rows.view(-1)
    flat_positions = positions.view(-1)
    flat_tail_ids = tail_ids.view(-1)
    # The model-side selector only calls this unchecked launcher after
    # _compact_kpool_triton_query_len verifies uniform, exactly covered spans.
    num_requests = flat_k.shape[0] // query_len

    block_d = min(128, triton.next_power_of_2(head_dim))
    grid = (num_requests, triton.cdiv(head_dim, block_d))
    _update_compact_kpool_kernel[grid](
        flat_k,
        flat_gate,
        flat_valid,
        flat_positions,
        compressed_cache,
        tail_cache,
        flat_tail_ids,
        block_table,
        ape,
        compressed_cache.shape[0],
        tail_cache.shape[0],
        tail_cache.shape[2],
        compressed_cache.shape[1],
        block_table.shape[1],
        flat_k.stride(0),
        flat_k.stride(1),
        flat_gate.stride(0),
        flat_gate.stride(1),
        flat_valid.stride(0),
        flat_positions.stride(0),
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        compressed_cache.stride(3),
        tail_cache.stride(0),
        tail_cache.stride(1),
        tail_cache.stride(2),
        tail_cache.stride(3),
        flat_tail_ids.stride(0),
        block_table.stride(0),
        block_table.stride(1),
        ape.stride(0),
        ape.stride(1),
        RATE=rate,
        QUERY_LEN=query_len,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=4,
    )


# Keep the validated entry point public; the model hot path uses the private
# launcher only after enforcing the same contracts upstream.
__all__ = ["update_compact_kpool"]
