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

from xllm.python.layers.flash_comm1 import FlashComm1Context, FlashComm1MetadataBuilder
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.acl_graph import (
    AclGraphEntry,
    AclGraphRunner,
    StaticGraphAttentionMetadata,
)
from xllm.python.model_executor.runners.decode_acl_graph import DecodeAclGraphRunner


def _metadata(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "num_reqs": 2,
        "num_tokens": 6,
        "num_scheduled_tokens": [4, 2],
        "num_computed_tokens": [10, 20],
        "query_start_loc": [0, 4, 6],
        "is_prefilling": [0, 0],
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("input_tokens", [6, 8])
def test_input_batch_preserves_upstream_request_layout(input_tokens: int) -> None:
    batch = InputBatch.from_runtime(
        torch.arange(input_tokens, dtype=torch.int32),
        torch.arange(input_tokens, dtype=torch.int32),
        _metadata(num_tokens=input_tokens),
    )

    assert batch.num_reqs == 2
    assert batch.num_tokens == input_tokens
    assert batch.num_scheduled_tokens == (4, 2)
    assert batch.num_computed_tokens == (10, 20)
    assert batch.query_start_loc == (0, 4, 6)


def test_bind_graph_inputs_only_changes_graph_owned_fields() -> None:
    batch = InputBatch.from_runtime(
        torch.arange(6, dtype=torch.int32),
        torch.arange(6, dtype=torch.int32),
        _metadata(),
    )
    is_padding = torch.tensor([False, False, False, False, False, False, True, True])

    padded = batch.bind_graph_inputs(
        torch.zeros(8, dtype=torch.int32),
        torch.zeros(8, dtype=torch.int32),
        is_padding,
    )

    assert padded.num_reqs == 2
    assert padded.num_tokens == 6
    assert padded.num_tokens_after_padding == 8
    assert padded.num_scheduled_tokens == (4, 2)
    assert padded.is_padding is is_padding


def test_input_batch_rejects_inconsistent_upstream_fields() -> None:
    with pytest.raises(ValueError, match="query_start_loc disagrees"):
        InputBatch.from_runtime(
            torch.arange(6, dtype=torch.int32),
            torch.arange(6, dtype=torch.int32),
            _metadata(query_start_loc=[0, 3, 6]),
        )


@pytest.mark.parametrize("num_dummy_tokens", [1, 4])
def test_empty_dp_rank_preserves_dummy_execution_rows(num_dummy_tokens: int) -> None:
    batch = InputBatch.from_runtime(
        torch.zeros(num_dummy_tokens, dtype=torch.int32),
        torch.zeros(num_dummy_tokens, dtype=torch.int32),
        _metadata(
            num_reqs=0,
            num_tokens=num_dummy_tokens,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
    )

    assert batch.num_reqs == 0
    assert batch.num_tokens == num_dummy_tokens
    assert batch.is_dummy


def test_graph_binding_keeps_inactive_context_and_refreshes_padding() -> None:
    runner = DecodeAclGraphRunner.__new__(DecodeAclGraphRunner)
    runner.execution_metadata_builders = (FlashComm1MetadataBuilder(tp_size=2, tp_rank=0, min_prefill_tokens=0),)
    entry = AclGraphEntry()
    entry.batch_size = 8
    entry.static_input_ids = torch.zeros(8, dtype=torch.int32)
    entry.static_positions = torch.zeros(8, dtype=torch.int32)
    empty = torch.zeros(8, dtype=torch.int32)
    entry.static_metadata = StaticGraphAttentionMetadata(
        slot_mapping=empty,
        paged_kv_indptr=empty,
        paged_kv_indices=empty,
        paged_kv_last_page_len=empty,
    )
    batch = InputBatch.from_runtime(entry.static_input_ids, entry.static_positions, _metadata(num_tokens=8))
    AclGraphRunner._bind_graph_input_batch(runner, entry, batch)
    padding = entry.is_padding
    assert padding.tolist() == [False] * 6 + [True] * 2
    assert entry.layer_shared_cache == {FlashComm1Context: None}

    dummy = InputBatch.from_runtime(
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        _metadata(
            num_reqs=0,
            num_tokens=1,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
    )
    AclGraphRunner._bind_graph_input_batch(runner, entry, dummy)
    assert entry.is_padding is padding
    assert padding.all()
    assert entry.layer_shared_cache == {FlashComm1Context: None}
