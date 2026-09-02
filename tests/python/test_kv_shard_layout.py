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
