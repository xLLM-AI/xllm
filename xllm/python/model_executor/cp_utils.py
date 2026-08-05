# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Context-Parallel (CP) sequence sharding for the Python model executor.

CP splits a single request's tokens along the sequence dimension across the CP
group so each rank runs attention over its own shard. This module owns the pure
index math: it turns per-sequence lengths into the shard/restore indices plus
the packed query/KV gather indices that one FIA call consumes.

Sharding is *zigzag* (head-tail balanced): each sequence is padded up to a
multiple of ``2 * cp_size`` and cut into ``2 * cp_size`` equal chunks; rank ``r``
owns chunk ``r`` (an early, short-prefix segment) paired with chunk
``2 * cp_size - 1 - r`` (a late, long-prefix segment). Under causal attention a
late token attends far more KV than an early one, so pairing a low chunk with
the mirrored high chunk equalizes per-rank attention work — the load imbalance
of a plain contiguous split (rank ``r`` attends ``(r+1)/cp_size`` of the
sequence) is gone.

Attention is computed against the *full* sequence: ``cp_merge_rows`` /
``cp_gather_kv`` all-gather every rank's shard, so each rank reconstructs the
complete global-order KV and then attends its two owned segments over their
exact causal prefixes. Both segments are contiguous real ranges, so a single
FIA call with per-segment ``actual_seq_lengths`` and ``sparse_mode=3``
(right-aligned causal) masks every row exactly, with no custom mask.

The shard/merge functions are deliberately backend-agnostic (plain torch index
ops) so the round-trip ``merge(all_gather(shard(x))) == x`` can be unit-tested on
CPU without NPU or a live process group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch

from xllm.python import ops


@dataclass(frozen=True)
class CpContext:
    """Per-forward CP sharding plan (zigzag head-tail split).

    All index tensors live on the target device and use int64 (torch gather /
    index_select require int64 indices). The ``*_cu_seqlens`` are host int
    lists (FIA ``actual_seq_lengths`` accepts a list), cumulative and without a
    leading 0.
    """

    cp_size: int
    cp_rank: int
    # Number of (padded) rows this rank holds; identical across ranks so
    # all_gather is well-formed. Layout per sequence: [first-half chunk_len
    # rows, second-half chunk_len rows].
    total_local: int
    # [total_local] global row for each local row, or -1 for a padding row.
    shard_index: torch.Tensor
    # [total_local] shard_index with -1 replaced by 0 so it is a valid gather
    # index; pair with shard_valid_mask to zero padding rows afterwards.
    shard_gather_index: torch.Tensor
    # [total_local] True for real tokens, False for padding rows.
    shard_valid_mask: torch.Tensor
    # [total_global] index into the rank-major all-gather output that restores
    # the original global token order.
    restore_index: torch.Tensor
    # [total_real_local] local rows carrying a real token (this rank's queries),
    # packed per (sequence, half) in local order. Selects the FIA query rows
    # from the local packed hidden and, reused as a scatter index, writes the
    # FIA output back into the [total_local] layout (padding rows stay zero).
    query_index: torch.Tensor
    # FIA actual_seq_lengths: real query count of each non-empty (sequence,
    # half) segment, cumulative.
    q_cu_seqlens: List[int]
    # [sum(kv_cu_seqlens)] index into the global-order KV selecting each
    # segment's causal prefix [0, prefix_len), packed in the same segment order
    # as query_index.
    kv_gather_index: torch.Tensor
    # FIA actual_seq_lengths_kv: causal-prefix length of each segment,
    # cumulative. prefix_len == segment_start + query_count, so with
    # sparse_mode=3 query row i attends KV [0, segment_start + i] exactly.
    kv_cu_seqlens: List[int]


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_cp_context(
    seq_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> CpContext:
    """Build a zigzag CP context from per-sequence lengths.

    ``seq_lens`` are the per-request query lengths in the packed batch order.
    Returns index tensors on ``device``.
    """
    if cp_size <= 1:
        raise ValueError("build_cp_context requires cp_size > 1")

    num_chunks = cp_size * 2
    # The two chunk ids this rank owns: an early one and its mirror.
    first_chunk = cp_rank
    second_chunk = num_chunks - 1 - cp_rank

    shard_index: List[int] = []
    query_index: List[int] = []
    q_cu_seqlens: List[int] = []
    kv_gather_index: List[int] = []
    kv_cu_seqlens: List[int] = []
    # restore_index needs the per-seq local segment offset (same on every rank)
    # and the ownership map, so accumulate it in a second pass below.
    chunk_lens: List[int] = []
    local_seg_offsets: List[int] = []

    global_offset = 0
    local_offset = 0
    q_run = 0
    kv_run = 0
    for length in seq_lens:
        length = int(length)
        padded = _ceil_div(length, num_chunks) * num_chunks
        chunk_len = padded // num_chunks
        chunk_lens.append(chunk_len)
        local_seg_offsets.append(local_offset)

        # Emit the two owned segments in local order: first half then second
        # half. For each, the real rows sit at the front (small j) because real
        # position grows with j, so query_index stays front-packed per segment.
        for half, chunk_id in ((0, first_chunk), (1, second_chunk)):
            seg_local_base = local_offset + half * chunk_len
            seg_start = chunk_id * chunk_len  # first real position of segment
            real_count = 0
            for j in range(chunk_len):
                pos_in_seq = seg_start + j
                if pos_in_seq < length:
                    shard_index.append(global_offset + pos_in_seq)
                    query_index.append(seg_local_base + j)
                    real_count += 1
                else:
                    shard_index.append(-1)
            if real_count > 0:
                # Causal prefix ends exactly at the last real query position + 1
                # = seg_start + real_count (segment is a contiguous real range).
                prefix_len = seg_start + real_count
                q_run += real_count
                q_cu_seqlens.append(q_run)
                for p in range(prefix_len):
                    kv_gather_index.append(global_offset + p)
                kv_run += prefix_len
                kv_cu_seqlens.append(kv_run)

        global_offset += length
        local_offset += 2 * chunk_len

    total_local = local_offset
    total_global = global_offset

    # restore_index: for every global (real) row, where it lands in the
    # rank-major all-gather output [cp_size * total_local].
    restore_index: List[int] = []
    global_offset = 0
    for s, length in enumerate(seq_lens):
        length = int(length)
        chunk_len = chunk_lens[s]
        seg_offset = local_seg_offsets[s]
        for pos_in_seq in range(int(length)):
            chunk_id = pos_in_seq // chunk_len
            row_in_chunk = pos_in_seq % chunk_len
            if chunk_id < cp_size:
                owner_rank = chunk_id
                local_pos = seg_offset + row_in_chunk
            else:
                owner_rank = num_chunks - 1 - chunk_id
                local_pos = seg_offset + chunk_len + row_in_chunk
            restore_index.append(owner_rank * total_local + local_pos)
        global_offset += int(length)

    shard_tensor = torch.tensor(shard_index, dtype=torch.int64, device=device)
    valid_mask = shard_tensor >= 0
    gather_index = torch.where(
        valid_mask, shard_tensor, torch.zeros_like(shard_tensor)
    )

    return CpContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        total_local=total_local,
        shard_index=shard_tensor,
        shard_gather_index=gather_index,
        shard_valid_mask=valid_mask,
        restore_index=torch.tensor(
            restore_index, dtype=torch.int64, device=device
        ),
        query_index=torch.tensor(
            query_index, dtype=torch.int64, device=device
        ),
        q_cu_seqlens=q_cu_seqlens,
        kv_gather_index=torch.tensor(
            kv_gather_index, dtype=torch.int64, device=device
        ),
        kv_cu_seqlens=kv_cu_seqlens,
    )


def cp_shard_rows(x: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Select this rank's rows from a global packed tensor ``[T_global, ...]``.

    Padding rows are zeroed. Returns ``[total_local, ...]``.
    """
    local = x.index_select(0, ctx.shard_gather_index)
    if not bool(ctx.shard_valid_mask.all()):
        mask_shape = [ctx.shard_valid_mask.shape[0]] + [1] * (x.dim() - 1)
        local = local * ctx.shard_valid_mask.view(mask_shape).to(local.dtype)
    return local


def cp_shard_positions(positions: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Shard 1-D position ids; padding positions become 0."""
    local = positions.index_select(0, ctx.shard_gather_index)
    return local * ctx.shard_valid_mask.to(local.dtype)


def cp_merge_rows(local: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """Reassemble the global packed tensor from this rank's local shard.

    All-gathers the rank-major shards over the CP group then restores the
    original global token order. Returns ``[T_global, ...]``.
    """
    gathered = ops.cp_all_gather(local, 0, ctx.cp_size)
    return gathered.index_select(0, ctx.restore_index)


def cp_gather_kv(local_kv: torch.Tensor, ctx: CpContext) -> torch.Tensor:
    """All-gather this rank's KV shard back to full global token order.

    ``local_kv`` is ``[total_local, ...]`` (this rank's padded segments of every
    sequence). Returns ``kv_global`` ``[T_global, ...]``: the complete sequence
    in original token order, used both to write the full KV into this rank's
    paged cache (decode stays on the non-CP path and needs every position) and,
    via ``ctx.kv_gather_index``, to select each owned segment's causal prefix.
    """
    gathered = ops.cp_all_gather(local_kv, 0, ctx.cp_size)
    return gathered.index_select(0, ctx.restore_index)
