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
index math: it turns per-sequence lengths into (a) a ``shard_index`` that picks
this rank's rows out of the global packed hidden state and (b) a
``restore_index`` that reassembles the rank-major all-gather output back into the
original global order.

First version uses *contiguous* sharding: each sequence is padded up to a
multiple of ``cp_size`` and split into ``cp_size`` equal contiguous segments;
rank ``r`` owns segment ``r``. Padding rows (when a length is not divisible by
``cp_size``) are marked invalid and zeroed after the gather. Load-balanced
zigzag sharding is a later optimization (see plan P2).

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
    """Per-forward CP sharding plan (contiguous split).

    All index tensors live on the target device and use int64 (torch gather /
    index_select require int64 indices).
    """

    cp_size: int
    cp_rank: int
    # [total_local] global row for each local row, or -1 for a padding row.
    shard_index: torch.Tensor
    # [total_local] same as shard_index but -1 replaced by 0 so it is a valid
    # gather index; pair with shard_valid_mask to zero padding rows afterwards.
    shard_gather_index: torch.Tensor
    # [total_local] True for real tokens, False for padding rows.
    shard_valid_mask: torch.Tensor
    # [total_global] index into the rank-major all-gather output that restores
    # the original global token order.
    restore_index: torch.Tensor
    # Number of (padded) rows this rank holds; identical across ranks so
    # all_gather is well-formed.
    total_local: int
    # [num_seqs] real (unpadded) token count this rank holds per sequence.
    local_seq_lens: torch.Tensor
    # Cumulative per-seq query lengths of this rank's shard (FIA TND
    # actual_seq_lengths; excludes the leading 0). Each entry is seg_len_s.
    q_cu_seqlens: List[int]
    # Cumulative per-seq KV-prefix lengths for this rank (FIA
    # actual_seq_lengths_kv). Each per-seq length is (cp_rank+1)*seg_len_s: the
    # causal prefix rank r needs, since its q segment ends at global position
    # (r+1)*seg_len_s. Paired with sparse_mode=3 (right-aligned causal) this
    # yields exact per-row causal masking with no custom mask.
    kv_cu_seqlens: List[int]
    # [(cp_rank+1)*total_local] index into the rank-major all-gathered KV
    # ([cp_size*total_local] rows) selecting this rank's per-seq causal prefix
    # (segments 0..cp_rank of each sequence, in global position order).
    kv_gather_index: torch.Tensor


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def build_cp_context(
    seq_lens: Sequence[int],
    cp_size: int,
    cp_rank: int,
    device: torch.device,
) -> CpContext:
    """Build a contiguous-split CP context from per-sequence lengths.

    ``seq_lens`` are the per-request query lengths in the packed batch order.
    Returns index tensors on ``device``.
    """
    if cp_size <= 1:
        raise ValueError("build_cp_context requires cp_size > 1")

    shard_index: List[int] = []
    local_seq_lens: List[int] = []

    # Layout of this rank's local shard: concatenation of each sequence's
    # rank-r segment. seg_len_s = padded_len_s / cp_size.
    global_offset = 0
    local_seg_offsets: List[int] = []
    seg_lens: List[int] = []
    running_local = 0
    for length in seq_lens:
        length = int(length)
        padded = _ceil_div(length, cp_size) * cp_size
        seg_len = padded // cp_size
        seg_lens.append(seg_len)
        local_seg_offsets.append(running_local)
        running_local += seg_len

        real_on_rank = 0
        for j in range(seg_len):
            pos_in_seq = cp_rank * seg_len + j
            if pos_in_seq < length:
                shard_index.append(global_offset + pos_in_seq)
                real_on_rank += 1
            else:
                shard_index.append(-1)
        local_seq_lens.append(real_on_rank)
        global_offset += length

    total_local = running_local
    total_global = global_offset

    # restore_index: for every global (unpadded) row, where it lands in the
    # rank-major all-gather output [cp_size * total_local].
    restore_index: List[int] = []
    global_offset = 0
    for s, length in enumerate(seq_lens):
        length = int(length)
        seg_len = seg_lens[s]
        for pos_in_seq in range(length):
            owner_rank = pos_in_seq // seg_len
            local_pos = local_seg_offsets[s] + (pos_in_seq % seg_len)
            restore_index.append(owner_rank * total_local + local_pos)
        global_offset += length

    shard_tensor = torch.tensor(shard_index, dtype=torch.int64, device=device)
    valid_mask = shard_tensor >= 0
    gather_index = torch.where(valid_mask, shard_tensor, torch.zeros_like(shard_tensor))

    # FIA actual_seq_lengths (cumulative, no leading 0) and the causal-prefix
    # KV gather index for this rank's local-q attention.
    q_cu_seqlens: List[int] = []
    kv_cu_seqlens: List[int] = []
    kv_gather_index: List[int] = []
    q_run = 0
    kv_run = 0
    for s in range(len(seq_lens)):
        seg_len = seg_lens[s]
        q_run += seg_len
        q_cu_seqlens.append(q_run)
        kv_run += (cp_rank + 1) * seg_len
        kv_cu_seqlens.append(kv_run)
        base = local_seg_offsets[s]
        for j in range(cp_rank + 1):
            block = j * total_local + base
            for k in range(seg_len):
                kv_gather_index.append(block + k)

    return CpContext(
        cp_size=cp_size,
        cp_rank=cp_rank,
        shard_index=shard_tensor,
        shard_gather_index=gather_index,
        shard_valid_mask=valid_mask,
        restore_index=torch.tensor(
            restore_index, dtype=torch.int64, device=device
        ),
        total_local=total_local,
        local_seq_lens=torch.tensor(
            local_seq_lens, dtype=torch.int32, device=device
        ),
        q_cu_seqlens=q_cu_seqlens,
        kv_cu_seqlens=kv_cu_seqlens,
        kv_gather_index=torch.tensor(
            kv_gather_index, dtype=torch.int64, device=device
        ),
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


def cp_gather_kv(local_kv: torch.Tensor, ctx: CpContext):
    """All-gather this rank's KV shard and derive the two views CP prefill needs.

    ``local_kv`` is ``[total_local, ...]`` (this rank's padded segment of every
    sequence). Returns ``(kv_global, kv_prefix)`` where:

    * ``kv_global`` ``[T_global, ...]`` is the full sequence in original token
      order, for writing the complete KV into this rank's paged cache (decode
      stays on the non-CP path and needs every position).
    * ``kv_prefix`` is the causal prefix this rank's local queries attend over
      (segments ``0..cp_rank`` of each sequence, per-seq contiguous), consumed
      by FIA with ``kv_cu_seqlens`` and right-aligned causal masking.
    """
    gathered = ops.cp_all_gather(local_kv, 0, ctx.cp_size)
    kv_global = gathered.index_select(0, ctx.restore_index)
    kv_prefix = gathered.index_select(0, ctx.kv_gather_index)
    return kv_global, kv_prefix
