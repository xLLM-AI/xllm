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

"""CPU tests for SFA DCP graph-prepare metadata."""

from __future__ import annotations

from unittest.mock import MagicMock

import torch

from xllm.python.attention.kv_shard_layout import KVShardLayout
from xllm.python.layers.sfa_dcp import AscendSFADCPMetadataBuilder
from xllm.python.model_executor.forward_context import (
    AclGraphExecutionState,
    ForwardContext,
    copy_into_execution_buffer,
    forward_context,
)


def _cpu_context(execution_state: AclGraphExecutionState | None) -> ForwardContext:
    return ForwardContext(
        attention_backend=MagicMock(),
        device=torch.device("cpu"),
        metadata=MagicMock(),
        layer_caches=[],
        execution_state=execution_state,
    )


def _builder() -> AscendSFADCPMetadataBuilder:
    layout = KVShardLayout(physical_block_size=4, dcp_size=2, dcp_rank=0)
    return AscendSFADCPMetadataBuilder(
        layout=layout,
        device=torch.device("cpu"),
        max_num_reqs=4,
    )


def test_builder_graph_decode_skips_prefill_count() -> None:
    builder = _builder()
    slots = torch.tensor([0, 1], dtype=torch.int32)
    block_table = torch.tensor([[1, 2], [3, 0]], dtype=torch.int32)
    seq_lens = torch.tensor([8, 4], dtype=torch.int32)

    metadata = builder.build(
        slots,
        block_table,
        seq_lens,
        num_reqs=2,
        num_input_tokens=2,
        num_prefills=0,
    )

    assert metadata.num_prefills == 0
    assert metadata.dcp_context.kv_gather_block_ids is None
    assert metadata.dcp_context.kv_gather_block_table is None
    assert torch.equal(metadata.dcp_context.seq_lens, torch.tensor([4, 4], dtype=torch.int32))


def test_copy_into_execution_buffer_reuses_graph_storage() -> None:
    state = AclGraphExecutionState({})
    with forward_context(_cpu_context(state)):
        first = torch.tensor([1, 2, 3], dtype=torch.int32)
        buffer = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), first)
        pointer = buffer.data_ptr()
        second = torch.tensor([4, 5, 6], dtype=torch.int32)
        reused = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (3,)), second)

        assert reused.data_ptr() == pointer
        assert torch.equal(reused, second)


def test_copy_into_execution_buffer_eager_returns_source() -> None:
    source = torch.tensor([1, 2], dtype=torch.int32)
    with forward_context(_cpu_context(None)):
        out = copy_into_execution_buffer(("DCP_LOCAL_SLOTS", (2,)), source)
        assert out.data_ptr() == source.data_ptr()
