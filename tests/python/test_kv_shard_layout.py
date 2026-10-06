# Copyright 2026 The xLLM Authors. All Rights Reserved.
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

"""Unit tests for DCP paged-KV slot localization."""

from __future__ import annotations

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout


def test_localize_slots_does_not_rewrite_worker_logical_slots() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    logical_slots = torch.tensor([-1, 0, 3, 4, 7, 8, 12], dtype=torch.int32)
    original = logical_slots.clone()

    local_slots = layout.localize_slots(logical_slots)

    assert torch.equal(logical_slots, original)
    assert torch.equal(local_slots, torch.tensor([-1, -1, -1, 0, 3, -1, 4], dtype=torch.int32))


def test_local_seq_lens_are_derived_from_global_kv_seq_lens() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    global_seq_lens = torch.tensor([0, 4, 6, 8], dtype=torch.int32)

    assert torch.equal(
        layout.local_seq_lens(global_seq_lens),
        torch.tensor([0, 0, 2, 4], dtype=torch.int32),
    )


def test_indexer_reads_expanded_logical_block_table() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=0,
    )
    logical_blocks = torch.tensor([[3, 7, -1], [0, 2, 4]], dtype=torch.int32)

    assert torch.equal(
        layout.expand_indexer_block_table(logical_blocks),
        torch.tensor([[6, 7, 14, 15, -1, -1], [0, 1, 4, 5, 8, 9]], dtype=torch.int32),
    )


def test_pack_owned_slots_drops_foreign_and_invalid_slots() -> None:
    layout = KVShardLayout(
        physical_block_size=4,
        dcp_size=2,
        dcp_rank=1,
    )
    logical_slots = torch.tensor([-1, 0, 3, 4, 7, 8, 12], dtype=torch.int32)

    packed = layout.pack_owned_slots(logical_slots)

    # Owned slots keep their relative order; foreign and invalid ones end up behind them.
    assert torch.equal(
        packed,
        torch.tensor([0, 3, 4, -1, -1, -1, -1], dtype=torch.int32),
    )


def test_pack_owned_slots_keeps_kpool_tail_inside_the_valid_run() -> None:
    """page=128/DCP=4, top-k prefix 0..2047, kPool tail 2048..2050 (GLM-5.3).

    Rank 0 owns 128 of every 512 logical slots, so 512 prefix slots are local.
    The packed row has to carry all 515 owned slots ahead of the first ``-1``:
    SFA stops scanning at the first ``-1``, so a tail parked behind the padding
    would be dropped.
    """
    layout = KVShardLayout(
        physical_block_size=128,
        dcp_size=4,
        dcp_rank=0,
    )
    logical_slots = torch.cat(
        [
            torch.arange(2048, dtype=torch.int32),
            torch.arange(2048, 2051, dtype=torch.int32),
        ]
    )

    packed = layout.pack_owned_slots(logical_slots)

    assert torch.equal(packed[:512], torch.arange(512, dtype=torch.int32))
    assert torch.equal(packed[512:515], torch.tensor([512, 513, 514], dtype=torch.int32))
    assert bool((packed[515:] < 0).all())
    assert int((packed >= 0).sum()) == 515


def test_graph_padded_zero_block_expands_to_valid_indexer_pages() -> None:
    layout = KVShardLayout(
        physical_block_size=128,
        dcp_size=4,
        dcp_rank=0,
    )
    padded_row = torch.zeros((1, 2), dtype=torch.int32)

    expanded = layout.expand_indexer_block_table(padded_row)

    assert torch.equal(
        expanded,
        torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]], dtype=torch.int32),
    )
