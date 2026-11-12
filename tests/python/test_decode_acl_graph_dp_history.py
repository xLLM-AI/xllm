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

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xllm.python.attention.backend import LayerCache
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _runner(
    dp_rank: int,
    *,
    dp_size: int = 2,
    num_speculative_tokens: int = 0,
    is_spec_draft: bool = False,
) -> DecodeAclGraphRunner:
    runner = DecodeAclGraphRunner(
        nn.Identity(),
        SimpleNamespace(page_size=128, is_mla=False),
        torch.device("cpu"),
        max_batch=4,
        max_model_len=65536,
        dp_size=dp_size,
        dp_rank=dp_rank,
        num_speculative_tokens=num_speculative_tokens,
        is_spec_draft=is_spec_draft,
        enable_index_history_bucketing=True,
    )
    runner.bind_layer_caches([LayerCache(torch.empty(1), torch.empty(1), index=torch.empty(1))])
    return runner


def _metadata(global_lengths: tuple[int, ...], rank: int) -> SimpleNamespace:
    local_length = global_lengths[rank]
    is_dummy = local_length == 0
    local_length = max(local_length, 1)
    page_size = 128
    columns = (local_length + page_size - 1) // page_size
    return SimpleNamespace(
        is_prefill=False,
        is_chunked_prefill=False,
        is_dummy=is_dummy,
        dp_execution_token_counts=(1,) * len(global_lengths),
        dp_is_decode=(1,) * len(global_lengths),
        dp_global_kv_max_seq_lens=global_lengths,
        slot_mapping=torch.tensor([0], dtype=torch.int32),
        new_cache_slots_host_values=[0],
        paged_kv_indptr=torch.tensor([0, columns], dtype=torch.int32),
        paged_kv_indices=torch.arange(columns, dtype=torch.int32),
        paged_kv_last_page_len=torch.tensor([((local_length - 1) % page_size) + 1], dtype=torch.int32),
        block_table=torch.arange(columns, dtype=torch.int32).reshape(1, columns),
        kv_seq_lens=torch.tensor([local_length], dtype=torch.int32),
        kv_seq_lens_host_values=[local_length],
        max_seq_len=local_length,
        q_cu_seq_lens=torch.tensor([0, 1], dtype=torch.int32),
        kv_cu_seq_lens=torch.tensor([0, local_length], dtype=torch.int32),
        linear_state_indices=torch.tensor([0], dtype=torch.int32),
        has_initial_state=torch.tensor([True]),
        expanded_decode_metadata=None,
        multi_block_tables=(),
    )


@pytest.mark.parametrize(
    ("lengths", "num_speculative_tokens", "is_spec_draft", "expected_bucket"),
    [
        ((32768, 0), 0, False, 1),
        ((0, 32769), 0, False, 2),
        ((1024, 32767), 3, False, 2),
        ((0, 32768), 3, True, 3),
    ],
)
def test_dp_raw_index_history_uses_shared_bucket(
    lengths: tuple[int, int], num_speculative_tokens: int, is_spec_draft: bool, expected_bucket: int
) -> None:
    input_ids = torch.tensor([1], dtype=torch.int32)
    for rank in range(len(lengths)):
        runner = _runner(rank, num_speculative_tokens=num_speculative_tokens, is_spec_draft=is_spec_draft)
        prepared = runner.prepare_execution(input_ids, _metadata(lengths, rank))
        assert prepared is not None
        assert prepared.token_capacity_bucket == expected_bucket


def test_tp_only_history_uses_local_length() -> None:
    runner = _runner(0, dp_size=1)
    metadata = _metadata((65536,), 0)
    metadata.dp_global_kv_max_seq_lens = (1,)
    prepared = runner.prepare_execution(torch.tensor([1], dtype=torch.int32), metadata)
    assert prepared is not None
    assert prepared.token_capacity_bucket == 2
