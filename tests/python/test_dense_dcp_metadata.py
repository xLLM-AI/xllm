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

"""CPU tests for builder-owned dense DCP local cache slots."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from xllm.python.attention.backend import LayerCache
from xllm.python.attention.dense_dcp_metadata import DenseDcpMetadata
from xllm.python.attention.dense_dcp_metadata_builder import DenseDcpMetadataBuilder
from xllm.python.model_executor.forward_context import get_execution_context
from xllm.python.model_executor.input_batch import InputBatch
from xllm.python.model_executor.runners.eager import EagerRunner


def _builder() -> DenseDcpMetadataBuilder:
    builder = DenseDcpMetadataBuilder(2, 1)
    builder.bind_layer_caches([LayerCache(torch.empty(8, 4, 1, 8), torch.empty(8, 4, 1, 8))])
    return builder


def _batch(num_tokens: int, capacity: int | None = None) -> InputBatch:
    batch = InputBatch.from_runtime(
        torch.arange(num_tokens, dtype=torch.int32),
        torch.zeros(num_tokens, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=num_tokens,
            num_tokens=num_tokens,
            num_scheduled_tokens=[1] * num_tokens,
            num_computed_tokens=[0] * num_tokens,
            query_start_loc=list(range(num_tokens + 1)),
            is_prefilling=[0] * num_tokens,
        ),
    )
    if capacity is None:
        return batch
    return batch.bind_graph_inputs(
        torch.zeros(capacity, dtype=torch.int32),
        torch.zeros(capacity, dtype=torch.int32),
        torch.arange(capacity) >= num_tokens,
    )


def _metadata(slots: list[int]) -> SimpleNamespace:
    num_tokens = len(slots)
    return SimpleNamespace(
        local_slot_mapping=torch.tensor(slots, dtype=torch.int32),
        has_kv_shard=True,
        kv_split_size=2,
        kv_split_rank=1,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        paged_kv_indptr=torch.arange(num_tokens + 1, dtype=torch.int32),
        paged_kv_indices=torch.zeros(num_tokens, dtype=torch.int32),
        paged_kv_last_page_len=torch.ones(num_tokens, dtype=torch.int32),
        block_table=torch.zeros(num_tokens, 1, dtype=torch.int32),
        kv_seq_lens=torch.ones(num_tokens, dtype=torch.int32),
        kv_seq_lens_host_values=[1] * num_tokens,
        kv_cu_seq_lens=torch.arange(num_tokens + 1, dtype=torch.int32),
        q_cu_seq_lens=torch.arange(num_tokens + 1, dtype=torch.int32),
        linear_state_indices=None,
        expanded_decode_metadata=None,
        multi_block_tables=(),
        new_cache_slots_host_values=list(range(num_tokens)),
        is_prefill=False,
        is_chunked_prefill=False,
        is_spec_verify=False,
        is_mixed=False,
    )


def test_eager_runner_exposes_upstream_slots_without_copying() -> None:
    batch = _batch(2)
    metadata = _metadata([-1, 3])

    def model(input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        dcp_metadata = get_execution_context(DenseDcpMetadata)
        assert dcp_metadata is not None
        assert dcp_metadata.local_slot_mapping is metadata.local_slot_mapping
        return input_ids

    model.execution_metadata_builders = (_builder(),)
    backend = SimpleNamespace(is_mla=False, prepare=MagicMock())
    runner = EagerRunner(model, backend, torch.device("cpu"))
    output = runner.execute(batch.input_ids, batch.positions, metadata, input_batch=batch)

    assert output is batch.input_ids
    backend.prepare.assert_called_once_with(metadata)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("has_kv_shard", False, RuntimeError),
        ("local_slot_mapping", None, RuntimeError),
        ("local_slot_mapping", torch.tensor([0], dtype=torch.int32), RuntimeError),
        ("local_slot_mapping", torch.tensor([[0, 1]], dtype=torch.int32), RuntimeError),
        ("local_slot_mapping", torch.tensor([0.0, 1.0]), TypeError),
        ("kv_split_size", 4, RuntimeError),
        ("kv_split_rank", 0, RuntimeError),
        ("is_spec_verify", True, NotImplementedError),
    ],
)
def test_builder_rejects_invalid_upstream_metadata_before_updating(
    field: str, value: object, error: type[Exception]
) -> None:
    builder = _builder()
    batch = _batch(2, 4)
    metadata = _metadata([-1, 3])
    persistent = builder.allocate_persistent(batch, metadata)
    setattr(metadata, field, value)

    with pytest.raises(error):
        builder.build(batch, metadata)
    with pytest.raises(error):
        builder.allocate_persistent(batch, metadata)
    with pytest.raises(error):
        builder.update_persistent(persistent, batch, metadata)
    assert persistent.local_slot_mapping.tolist() == [-1] * 4


def test_graph_builder_rejects_changed_capacity_and_dtype() -> None:
    builder = _builder()
    metadata = _metadata([0, 1])
    persistent = builder.allocate_persistent(_batch(2, 4), metadata)

    with pytest.raises(RuntimeError, match="capacity"):
        builder.update_persistent(persistent, _batch(2, 8), metadata)
    metadata.local_slot_mapping = metadata.local_slot_mapping.to(torch.int64)
    with pytest.raises(RuntimeError, match="dtype or device"):
        builder.update_persistent(persistent, _batch(2, 4), metadata)
    assert persistent.local_slot_mapping.tolist() == [-1] * 4


def test_graph_allocations_are_independent_for_each_entry() -> None:
    builder = _builder()
    batch = _batch(2, 4)
    metadata = _metadata([0, 1])
    first = builder.allocate_persistent(batch, metadata)
    second = builder.allocate_persistent(batch, metadata)

    builder.update_persistent(first, batch, metadata)

    assert first.local_slot_mapping.data_ptr() != second.local_slot_mapping.data_ptr()
    assert first.local_slot_mapping.tolist() == [0, 1, -1, -1]
    assert second.local_slot_mapping.tolist() == [-1] * 4
    assert first.empty_kv_shards.data_ptr() != second.empty_kv_shards.data_ptr()
    assert first.local_kv_lengths is not second.local_kv_lengths


def test_graph_update_preserves_addresses_and_updates_host_lengths_and_padding() -> None:
    builder = _builder()
    source = _metadata([0, 1])
    persistent = builder.allocate_persistent(_batch(2, 4), source)
    lengths = persistent.local_kv_lengths
    mask = persistent.empty_kv_shards
    slots = persistent.local_slot_mapping
    for kv_lengths, expected in (([5, 9], [1, 4]), ([1], [0]), ([8, 4, 13], [4, 0, 5])):
        source = _metadata(list(range(len(kv_lengths))))
        source.kv_seq_lens_host_values = kv_lengths
        builder.update_persistent(persistent, _batch(len(kv_lengths), 4), source)
        expected += [0] * (4 - len(expected))
        assert persistent.local_kv_lengths is lengths
        assert lengths == expected
        assert persistent.query_seq_ends == [1, 2, 3, 4]
        assert persistent.empty_kv_shards is mask
        assert mask.flatten().tolist() == [length == 0 for length in expected]
        assert persistent.local_slot_mapping is slots
        assert persistent.block_table is None  # Bound to the runner's view during prepare.


def test_eager_builder_aliases_int32_table_and_uses_upstream_host_values() -> None:
    source = _metadata([0, 1])
    source.kv_seq_lens_host_values = [5, 9]
    result = _builder().build(_batch(2), source)
    assert result.block_table is source.block_table
    assert result.local_kv_lengths == [1, 4]


@pytest.mark.parametrize("host_values", [None, [], [5], [-1, 5]])
def test_builder_rejects_missing_or_invalid_host_lengths(host_values: list[int] | None) -> None:
    source = _metadata([0, 1])
    source.kv_seq_lens_host_values = host_values
    with pytest.raises(RuntimeError, match="host KV lengths|non-negative KV length"):
        _builder().build(_batch(2), source)


@pytest.mark.parametrize("field", ["is_prefill", "is_chunked_prefill"])
def test_graph_builder_rejects_prefill_before_mutating_buffers(field: str) -> None:
    builder = _builder()
    source = _metadata([0])
    batch = _batch(1, 2)
    persistent = builder.allocate_persistent(batch, source)
    setattr(source, field, True)
    with pytest.raises(NotImplementedError, match="only decode"):
        builder.update_persistent(persistent, batch, source)
    assert persistent.local_slot_mapping.tolist() == [-1, -1]


def test_dummy_execution_token_does_not_require_a_logical_request() -> None:
    batch = InputBatch.from_runtime(
        torch.zeros(1, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        SimpleNamespace(
            num_reqs=0,
            num_tokens=1,
            num_scheduled_tokens=[],
            num_computed_tokens=[],
            query_start_loc=[0],
            is_prefilling=[],
        ),
    )
    metadata = _metadata([-1])
    builder = _builder()
    assert builder.build(batch, metadata).local_slot_mapping is metadata.local_slot_mapping
    padded_batch = batch.bind_graph_inputs(
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.tensor([False, True, True, True]),
    )
    persistent = builder.allocate_persistent(padded_batch, metadata)
    builder.update_persistent(persistent, padded_batch, metadata)
    assert persistent.local_slot_mapping.tolist() == [-1] * 4
